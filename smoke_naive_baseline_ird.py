"""
smoke_naive4_linear.py  — FIXED VERSION
Naive 4: causal linear extrapolation within segment.
Works directly on raw IRD (cm/h) — no log space to avoid divergence.
For each event at position k: fit line to IRD values at positions 1..k-1,
extrapolate to position k. Clip predictions to [0.1, 20] cm/h.
Requires >= 2 prior events — NaN for positions 1 and 2.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error

PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
sys.path.insert(0, str(PROJECT_ROOT))

from config import EVENT_CSV, FIELD_NAMES
from plot_style import apply_style, COLORS, FONT, FIELD_COLORS

HELD_OUT = [3203, 4104, 5102, 6303, 7201]
IRD_CLIP_LO = 0.1    # cm/h — physical lower bound
IRD_CLIP_HI = 20.0   # cm/h — physical upper bound

FONT_OVERRIDE = {"title":11,"axis_label":10,"tick":9,
                 "legend":9,"annotation":10}
def _fs(k): return FONT_OVERRIDE.get(k, FONT.get(k, 11))

# ── Load — keep basin_number throughout, no prepare_features ─────────────────
print("Loading data...")
df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
df = df.loc[:, ~df.columns.duplicated()]

TARGET = "IRD_norm_log" if "IRD_norm_log" in df.columns else "IRD_norm"
if TARGET not in df.columns:
    raise ValueError("IRD_norm_log not found")

for col in ["segment_id","IRD_at_reset","IRD",TARGET,"LCT","basin_number"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# We work directly in RAW IRD space — no normalization
# IRD column = raw infiltration rate (cm/h)
ho_all = df[
    (df["row_type"] == "event") &
    (df["basin_number"].isin(HELD_OUT))
][["basin_number","segment_id","opening_valve_date",
   "split_held_out","IRD","IRD_at_reset",TARGET,"LCT"]].copy()

ho_all = ho_all.sort_values(
    ["basin_number","segment_id","opening_valve_date"]
).reset_index(drop=True)

print(f"  Held-out events loaded: {len(ho_all)}")

# ── Causal linear extrapolation in RAW IRD space ──────────────────────────────
print("  Computing causal linear extrapolation in raw IRD space...")

def linear_extrap_ird(group):
    """
    For each event at position i, fit a line to raw IRD values
    at positions 0..i-1 using LCT as x-axis.
    Extrapolate to current LCT.
    Clip to physical bounds.
    """
    group = group.sort_values("opening_valve_date").copy()
    preds = []
    for i in range(len(group)):
        if i < 2:
            preds.append(np.nan)
            continue
        hist_lct = group["LCT"].iloc[:i].values.astype(float)
        hist_ird = group["IRD"].iloc[:i].values.astype(float)
        valid    = np.isfinite(hist_lct) & np.isfinite(hist_ird) & (hist_ird > 0)
        if valid.sum() < 2:
            preds.append(np.nan)
            continue
        coeffs   = np.polyfit(hist_lct[valid], hist_ird[valid], 1)
        curr_lct = float(group["LCT"].iloc[i])
        if not np.isfinite(curr_lct):
            preds.append(np.nan)
            continue
        pred = float(np.polyval(coeffs, curr_lct))
        # Clip to physical bounds — linear extrapolation can go negative
        pred = np.clip(pred, IRD_CLIP_LO, IRD_CLIP_HI)
        preds.append(pred)
    group["ird_pred_linear"] = preds
    return group

ho_all = ho_all.groupby(
    ["basin_number","segment_id"], group_keys=True
).apply(linear_extrap_ird)

# ── Filter to test events ─────────────────────────────────────────────────────
ho_test = ho_all[ho_all["split_held_out"] == "held_out_test"].copy()
ho_test['basin_number'] = ho_test.index.get_level_values("basin_number").astype(int)
ho_test['segment_id'] = ho_test.index.get_level_values("segment_id").astype(int)

# print(ho_test)
# print(ho_test.columns)
# print(ho_test[["basin_number"]])
# exit()
ho_test = ho_test.reset_index(drop=True)

# IRD true — raw IRD column (already in cm/h)
ird_true = ho_test["IRD"].values.astype(float)
ird_pred = ho_test["ird_pred_linear"].values.astype(float)

n_total = len(ho_test)
n_avail = int(np.isfinite(ird_pred).sum())
print(f"\n  Total test events  : {n_total}")
print(f"  With linear pred   : {n_avail} ({100*n_avail/n_total:.1f}%)")

# ── Metrics helper ────────────────────────────────────────────────────────────
def eval_ird(name, yt, yp):
    mask = (yt>0) & (yp>0) & np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() < 2:
        print(f"  {name}: insufficient data ({mask.sum()})")
        return {}
    r2   = r2_score(yt[mask], yp[mask])
    rmse = float(np.sqrt(mean_squared_error(yt[mask], yp[mask])))
    mape = float(np.mean(np.abs((yt[mask]-yp[mask])/yt[mask]))*100)
    n    = int(mask.sum())
    print(f"  {name:<55}: R²={r2:>+.3f}  "
          f"RMSE={rmse:.3f} cm/h  MAPE={mape:.1f}%  n={n}")
    return {"r2":r2,"rmse":rmse,"mape":mape,"n":n,
            "yt":yt[mask],"yp":yp[mask]}

# ── Quick sanity check — show some raw values ─────────────────────────────────
print(f"\n  IRD range (true)  : "
      f"{np.nanmin(ird_true):.2f} – {np.nanmax(ird_true):.2f} cm/h")
print(f"  IRD range (pred)  : "
      f"{np.nanmin(ird_pred):.2f} – {np.nanmax(ird_pred):.2f} cm/h  "
      f"[clipped to {IRD_CLIP_LO}–{IRD_CLIP_HI}]")

# Sample 5 predictions for visual check
print(f"\n  Sample predictions (first 5 events with valid pred):")
print(f"  {'Basin':<8} {'Seg':<6} {'LCT':>6} "
      f"{'IRD_true':>10} {'IRD_pred':>10} {'Error':>8}")
sample = ho_test[np.isfinite(ird_pred)].head(5)
for _, row in sample.iterrows():
    print(f"  {int(row['basin_number']):<8} "
          f"{int(row['segment_id']):<6} "
          f"{row['LCT']:>6.1f} "
          f"{row['IRD']:>10.3f} "
          f"{row['ird_pred_linear']:>10.3f} "
          f"{row['ird_pred_linear']-row['IRD']:>8.3f}")

# ── Overall metrics ───────────────────────────────────────────────────────────
print(f"\nOverall metrics — raw IRD space (cm/h):")
print(f"  {'-'*85}")
print(f"  {'Model 1 Condition E':<55}: "
      f"R²=+0.898  RMSE=0.607 cm/h  MAPE=13.1%  n=4386")
print(f"  {'True persistence (prev IRD)':<55}: "
      f"R²=+0.898  RMSE=0.569 cm/h  MAPE=11.1%  n=2499")
m_lin = eval_ird("Naive 4: causal linear extrap (clipped)",
                  ird_true, ird_pred)

# ── Per-basin ─────────────────────────────────────────────────────────────────
print(f"\nPer-basin breakdown — raw IRD (cm/h):")
print(f"  {'Basin':<8} {'Field':<12} "
      f"{'R²':>8} {'RMSE':>8} {'MAPE%':>7} {'n':>6}")
print(f"  {'-'*55}")

basin_results = {}
for bn in HELD_OUT:
    bdf   = ho_test[ho_test["basin_number"] == bn]
    field = FIELD_NAMES.get(int(str(bn)[0]),"")
    btru  = bdf["IRD"].values.astype(float)
    bprd  = bdf["ird_pred_linear"].values.astype(float)
    bm    = (btru>0)&(bprd>0)&np.isfinite(btru)&np.isfinite(bprd)
    if bm.sum() < 2:
        print(f"  {bn:<8} {field:<12}: insufficient ({bm.sum()})")
        continue
    r2_b   = r2_score(btru[bm], bprd[bm])
    rmse_b = float(np.sqrt(mean_squared_error(btru[bm],bprd[bm])))
    mape_b = float(np.mean(np.abs((btru[bm]-bprd[bm])/btru[bm]))*100)
    print(f"  {bn:<8} {field:<12} "
          f"{r2_b:>+8.3f} {rmse_b:>8.3f} "
          f"{mape_b:>6.1f}% {int(bm.sum()):>6}")
    basin_results[bn] = dict(r2=r2_b, rmse=rmse_b, mape=mape_b,
                              n=int(bm.sum()), field=field,
                              ird_true=btru[bm], ird_pred=bprd[bm])

# ── Position breakdown ────────────────────────────────────────────────────────
ho_test = ho_test.copy()
ho_test["event_pos"] = ho_test.groupby(
    ["basin_number","segment_id"], group_keys=False
).cumcount() + 1

print(f"\nLinear extrap by event position — raw IRD (cm/h):")
print(f"  {'-'*70}")
for label, pmask in [
    ("Pos 1 — first (undefined)",  ho_test["event_pos"]==1),
    ("Pos 2 — second (undefined)", ho_test["event_pos"]==2),
    ("Pos 3–5",                    ho_test["event_pos"].between(3,5)),
    ("Pos 6+",                     ho_test["event_pos"]>=6),
]:
    sub  = ho_test[pmask]
    stru = sub["IRD"].values.astype(float)
    sprd = sub["ird_pred_linear"].values.astype(float)
    sm   = (stru>0)&(sprd>0)&np.isfinite(stru)&np.isfinite(sprd)
    if sm.sum() < 2:
        n_tot = len(sub)
        print(f"  {label:<45}: n={n_tot}  [undefined]")
        continue
    r2_p   = r2_score(stru[sm], sprd[sm])
    rmse_p = float(np.sqrt(mean_squared_error(stru[sm],sprd[sm])))
    mape_p = float(np.mean(np.abs((stru[sm]-sprd[sm])/stru[sm]))*100)
    print(f"  {label:<45}: R²={r2_p:>+.3f}  "
          f"RMSE={rmse_p:.3f} cm/h  MAPE={mape_p:.1f}%  "
          f"n={int(sm.sum())}")

# ── Scatter plots ─────────────────────────────────────────────────────────────
if not basin_results:
    print("\n  No basin results — skipping plots")
else:
    apply_style()

    # Per-basin panels
    fig, axes = plt.subplots(1, len(HELD_OUT), figsize=(16,4),
                              gridspec_kw={"wspace":0.28})
    fig.suptitle(
        "Naive 4: Causal linear extrapolation — actual vs predicted IRD (cm/h)\n"
        "Per-basin scatter | held-out test events | positions ≥ 3",
        fontsize=_fs("title"), y=1.03)

    for ax, bn in zip(axes, HELD_OUT):
        res = basin_results.get(bn,{})
        if not res:
            ax.text(0.5,0.5,"no data",ha="center",va="center",
                    transform=ax.transAxes)
            ax.set_title(f"Basin {bn}",fontsize=_fs("title"),loc="left")
            continue
        color = FIELD_COLORS.get(res["field"], COLORS["deep_blue"])
        ird_t = res["ird_true"]
        ird_p = res["ird_pred"]
        ax.scatter(ird_t, ird_p, s=10, alpha=0.55,
                   color=color, zorder=3, linewidths=0)
        lo = min(np.percentile(ird_t,1), np.percentile(ird_p,1))*0.85
        hi = max(np.percentile(ird_t,99), np.percentile(ird_p,99))*1.15
        ax.plot([lo,hi],[lo,hi],"k--",lw=1.0,alpha=0.5)
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.annotate(
            f"$R^2$={res['r2']:+.3f}\n"
            f"RMSE={res['rmse']:.3f}\n"
            f"MAPE={res['mape']:.1f}%\nn={res['n']}",
            xy=(0.05,0.97), xycoords="axes fraction",
            fontsize=_fs("annotation")-1, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=COLORS["light_gray"], alpha=0.90))
        ax.set_title(f"Basin {bn}",fontsize=_fs("title"),loc="left")
        ax.set_xlabel("Actual IRD (cm/h)",fontsize=_fs("axis_label"))
        if bn==HELD_OUT[0]:
            ax.set_ylabel("Predicted IRD (cm/h)",fontsize=_fs("axis_label"))
        ax.tick_params(labelsize=_fs("tick"))
        ax.grid(True,alpha=0.20)
    plt.tight_layout()
    plt.show()

    # Pooled scatter
    all_t = np.concatenate([r["ird_true"] for r in basin_results.values()])
    all_p = np.concatenate([r["ird_pred"] for r in basin_results.values()])
    fig2, ax2 = plt.subplots(figsize=(5,5))
    for bn in HELD_OUT:
        res = basin_results.get(bn,{})
        if not res: continue
        color = FIELD_COLORS.get(res["field"], COLORS["deep_blue"])
        ax2.scatter(res["ird_true"], res["ird_pred"], s=8, alpha=0.45,
                    color=color, linewidths=0,
                    label=f"{bn} — {res['field']}", zorder=3)
    lo2 = min(np.percentile(all_t,1), np.percentile(all_p,1))*0.85
    hi2 = max(np.percentile(all_t,99), np.percentile(all_p,99))*1.15
    ax2.plot([lo2,hi2],[lo2,hi2],"k--",lw=1.2,alpha=0.5)
    ax2.set_xlim(lo2,hi2); ax2.set_ylim(lo2,hi2)
    pm    = (all_t>0)&(all_p>0)&np.isfinite(all_t)&np.isfinite(all_p)
    r2p   = r2_score(all_t[pm], all_p[pm])
    rmsep = float(np.sqrt(mean_squared_error(all_t[pm],all_p[pm])))
    mapep = float(np.mean(np.abs((all_t[pm]-all_p[pm])/all_t[pm]))*100)
    ax2.annotate(
        f"Pooled\n$R^2$={r2p:+.3f}\n"
        f"RMSE={rmsep:.3f} cm/h\n"
        f"MAPE={mapep:.1f}%\nn={int(pm.sum())}",
        xy=(0.05,0.97), xycoords="axes fraction",
        fontsize=_fs("annotation"), va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor=COLORS["light_gray"], alpha=0.92))
    ax2.set_xlabel("Actual IRD (cm/h)", fontsize=_fs("axis_label"))
    ax2.set_ylabel("Predicted IRD (cm/h)", fontsize=_fs("axis_label"))
    ax2.set_title("Pooled — linear extrap baseline",
                  fontsize=_fs("title"), loc="left")
    ax2.legend(fontsize=_fs("legend")-1, loc="lower right")
    ax2.tick_params(labelsize=_fs("tick"))
    ax2.grid(True, alpha=0.20)
    plt.tight_layout()
    plt.show()

    print(f"\nFinal summary — raw IRD space (cm/h):")
    print(f"  {'Model':<52} {'R²':>7} {'RMSE':>8} {'MAPE%':>7} {'n':>6}")
    print(f"  {'-'*80}")
    print(f"  {'Model 1 Condition E':<52} +0.898    0.607   13.1%   4386")
    print(f"  {'True persistence':<52} +0.898    0.569   11.1%   2499")
    print(f"  {'Naive 4: linear extrap (pooled)':<52} "
          f"{r2p:>+7.3f} {rmsep:>8.3f} {mapep:>6.1f}% "
          f"{int(pm.sum()):>6}")
    print(f"\n  Model 1 vs linear extrap: ΔRMSE = {0.607-rmsep:+.3f} cm/h  "
          f"({'Model 1 better' if 0.607 < rmsep else 'linear better'})")