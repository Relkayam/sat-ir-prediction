# smoke_m2_scatter.py
# Scatter plots: Model 2 vs naive baseline — actual vs predicted IRD_at_reset
# Retrains Model 2 Condition E and produces:
#   Figure 1: 5-panel per-basin (Model 2 left, naive right)
#   Figure 2: pooled scatter both models

import sys, warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESET_CSV, RANDOM_SEED, FIELD_NAMES
from pipeline.features import MODEL2_FEATURES, TARGET_M2
from plot_style import apply_style, COLORS, FONT, FIELD_COLORS

HELD_OUT = [3203, 4104, 5102, 6303, 7201]
FONT_OVERRIDE = {"title":11,"axis_label":10,"tick":9,
                 "legend":9,"annotation":10}
def _fs(k): return FONT_OVERRIDE.get(k, FONT.get(k,11))

def _metrics(yt, yp):
    mask = (yt>0)&(yp>0)&np.isfinite(yt)&np.isfinite(yp)
    if mask.sum() < 2: return {}
    r2   = r2_score(yt[mask], yp[mask])
    rmse = float(np.sqrt(mean_squared_error(yt[mask],yp[mask])))
    mape = float(np.mean(np.abs((yt[mask]-yp[mask])/yt[mask]))*100)
    return dict(r2=r2,rmse=rmse,mape=mape,n=int(mask.sum()),
                yt=yt[mask],yp=yp[mask])

# ── Load and train ────────────────────────────────────────────────────────────
print("Loading reset dataset...")
df = pd.read_csv(RESET_CSV, parse_dates=["reset_date"])
df = df.loc[:, ~df.columns.duplicated()]
avail = [f for f in MODEL2_FEATURES if f in df.columns]
required = avail + [TARGET_M2,"prev_IRD_at_reset_raw","IRD_at_reset"]

# Condition E: reassign outlier basins
df_e = df.copy()
om = (df_e["basin_role"]=="outlier")&(df_e["split_held_out"]=="excluded")
df_e.loc[om,"split_held_out"] = df_e.loc[om,"split_chrono"]

train = df_e[df_e["split_held_out"]=="train"].dropna(subset=required).reset_index(drop=True)
val   = df_e[df_e["split_held_out"]=="val"].dropna(subset=required).reset_index(drop=True)
test  = df_e[df_e["split_held_out"]=="held_out_test"].dropna(subset=required).reset_index(drop=True)

print(f"  train={len(train)}  val={len(val)}  test={len(test)}")

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
sc  = StandardScaler()
Xtr = sc.fit_transform(train[avail].values)
Xva = sc.transform(val[avail].values)
Xte = sc.transform(test[avail].values)

print("  Training Model 2 Condition E...")
model = LGBMRegressor(
    n_estimators=1000,max_depth=-1,num_leaves=31,
    learning_rate=0.05,subsample=0.8,feature_fraction=0.8,
    min_child_samples=10,reg_alpha=0.1,reg_lambda=1.0,
    random_state=RANDOM_SEED,n_jobs=-1,verbose=-1)
model.fit(Xtr,train[TARGET_M2].values,
          eval_set=[(Xva,val[TARGET_M2].values)],
          callbacks=[early_stopping(50,verbose=False),
                     log_evaluation(period=-1)])
print(f"  best_iter={model.best_iteration_}")

y_pred_log = model.predict(Xte)
prev_ird   = test["prev_IRD_at_reset_raw"].values.astype(float)
test["ird_pred"]  = prev_ird * np.exp(y_pred_log)   # Model 2
test["ird_naive"] = prev_ird                          # Naive
test["ird_true"]  = test["IRD_at_reset"].values.astype(float)

# ── Per-basin results ─────────────────────────────────────────────────────────
print(f"\nPer-basin metrics — IRD space (cm/h):")
print(f"  {'Basin':<8} {'Field':<12} "
      f"{'M2 R²':>8} {'M2 RMSE':>8} {'M2 MAPE%':>9} "
      f"{'Nv R²':>8} {'Nv RMSE':>8} {'Nv MAPE%':>9} {'n':>5}")
print(f"  {'-'*80}")

basin_m2, basin_nv = {}, {}
for bn in HELD_OUT:
    bdf   = test[test["basin_number"]==bn]
    field = FIELD_NAMES.get(int(str(bn)[0]),"")
    yt    = bdf["ird_true"].values.astype(float)
    yp_m2 = bdf["ird_pred"].values.astype(float)
    yp_nv = bdf["ird_naive"].values.astype(float)
    m2 = _metrics(yt,yp_m2)
    nv = _metrics(yt,yp_nv)
    if not m2: continue
    print(f"  {bn:<8} {field:<12} "
          f"{m2['r2']:>+8.3f} {m2['rmse']:>8.3f} {m2['mape']:>8.1f}% "
          f"{nv.get('r2',np.nan):>+8.3f} {nv.get('rmse',np.nan):>8.3f} "
          f"{nv.get('mape',np.nan):>8.1f}% {m2['n']:>5}")
    basin_m2[bn] = dict(**m2, field=field)
    basin_nv[bn] = dict(**nv, field=field)

# Pooled
yt_all  = test["ird_true"].values.astype(float)
yp_all  = test["ird_pred"].values.astype(float)
ynv_all = test["ird_naive"].values.astype(float)
pm2 = _metrics(yt_all, yp_all)
pnv = _metrics(yt_all, ynv_all)
print(f"\n  Pooled — Model 2: R²={pm2['r2']:+.3f}  "
      f"RMSE={pm2['rmse']:.3f}  MAPE={pm2['mape']:.1f}%  n={pm2['n']}")
print(f"  Pooled — Naive  : R²={pnv['r2']:+.3f}  "
      f"RMSE={pnv['rmse']:.3f}  MAPE={pnv['mape']:.1f}%  n={pnv['n']}")

# ── Scatter plots ─────────────────────────────────────────────────────────────
apply_style()

# Figure 1: 5 basins × 2 columns (Model 2 | Naive)
fig, axes = plt.subplots(len(HELD_OUT), 2,
                          figsize=(10, 4.0*len(HELD_OUT)),
                          gridspec_kw={"wspace":0.30,"hspace":0.45})
fig.suptitle(
    "Model 2 vs Naive baseline — actual vs predicted IRD_reset (cm/h)\n"
    "Left: Model 2 (Condition E) | Right: Naive (predict no change)",
    fontsize=_fs("title"), y=1.01)

for row_i, bn in enumerate(HELD_OUT):
    m2  = basin_m2.get(bn,{})
    nv  = basin_nv.get(bn,{})
    if not m2: continue
    field = m2["field"]
    color = FIELD_COLORS.get(field, COLORS["deep_blue"])

    for col_i, (res, label) in enumerate([
        (m2, "Model 2"), (nv, "Naive")
    ]):
        ax = axes[row_i, col_i]
        yt = res["yt"]; yp = res["yp"]
        ax.scatter(yt, yp, s=30, alpha=0.70,
                   color=color if col_i==0 else COLORS["mid_gray"],
                   zorder=3, linewidths=0)
        lo = min(yt.min(),yp.min())*0.85
        hi = max(yt.max(),yp.max())*1.15
        ax.plot([lo,hi],[lo,hi],"k--",lw=1.0,alpha=0.5)
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.annotate(
            f"$R^2$={res['r2']:+.3f}\n"
            f"RMSE={res['rmse']:.3f}\n"
            f"MAPE={res['mape']:.1f}%\n"
            f"n={res['n']}",
            xy=(0.05,0.97), xycoords="axes fraction",
            fontsize=_fs("annotation")-1, va="top",
            bbox=dict(boxstyle="round,pad=0.3",facecolor="white",
                      edgecolor=COLORS["light_gray"],alpha=0.90))
        if col_i==0:
            ax.set_ylabel(f"Basin {bn}\nPredicted (cm/h)",
                          fontsize=_fs("axis_label"))
        if row_i==len(HELD_OUT)-1:
            ax.set_xlabel("Actual IRD_reset (cm/h)",
                          fontsize=_fs("axis_label"))
        ax.set_title(f"Basin {bn} — {field} | {label}",
                     fontsize=_fs("title"), loc="left", pad=4)
        ax.tick_params(labelsize=_fs("tick"))
        ax.grid(True, alpha=0.20)

plt.tight_layout()
plt.show()

# Figure 2: pooled scatter — Model 2 vs Naive side by side
fig2, axes2 = plt.subplots(1,2,figsize=(10,5),
                            gridspec_kw={"wspace":0.30})
fig2.suptitle("Pooled — all 5 held-out basins",
               fontsize=_fs("title"))

for ax, (res_dict, basin_dict, title, color) in zip(axes2, [
    (pm2, basin_m2, "Model 2 (Condition E)", COLORS["deep_blue"]),
    (pnv, basin_nv, "Naive baseline",        COLORS["mid_gray"]),
]):
    for bn in HELD_OUT:
        bd    = basin_dict.get(bn,{})
        if not bd: continue
        fc    = FIELD_COLORS.get(bd["field"], color)
        ax.scatter(bd["yt"], bd["yp"], s=20, alpha=0.60,
                   color=fc, zorder=3, linewidths=0,
                   label=f"{bn} — {bd['field']}")
    all_t = np.concatenate([basin_dict[b]["yt"]
                             for b in HELD_OUT if b in basin_dict])
    all_p = np.concatenate([basin_dict[b]["yp"]
                             for b in HELD_OUT if b in basin_dict])
    lo = min(all_t.min(),all_p.min())*0.85
    hi = max(all_t.max(),all_p.max())*1.15
    ax.plot([lo,hi],[lo,hi],"k--",lw=1.2,alpha=0.5)
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    m = _metrics(all_t, all_p)
    ax.annotate(
        f"Pooled\n$R^2$={m['r2']:+.3f}\n"
        f"RMSE={m['rmse']:.3f} cm/h\n"
        f"MAPE={m['mape']:.1f}%\nn={m['n']}",
        xy=(0.05,0.97), xycoords="axes fraction",
        fontsize=_fs("annotation"), va="top",
        bbox=dict(boxstyle="round,pad=0.35",facecolor="white",
                  edgecolor=COLORS["light_gray"],alpha=0.92))
    ax.set_xlabel("Actual IRD_reset (cm/h)",fontsize=_fs("axis_label"))
    ax.set_ylabel("Predicted IRD_reset (cm/h)",fontsize=_fs("axis_label"))
    ax.set_title(title, fontsize=_fs("title"), loc="left")
    ax.legend(fontsize=_fs("legend")-1, loc="lower right")
    ax.tick_params(labelsize=_fs("tick"))
    ax.grid(True, alpha=0.20)

plt.tight_layout()
plt.show()