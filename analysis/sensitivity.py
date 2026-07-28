"""
analysis/sensitivity.py — Engineering sensitivity for Model 1  (v2: ALE)
=======================================================================
Why v2:
  v1 used one-at-a-time (PDP-style) perturbation. With correlated features
  that asks the model OFF-MANIFOLD questions ("raise prev_ALPHA but hold
  cumulative drying fixed" - never happens in reality), so the sign/size are
  unstable. That is the same collinearity pathology that made SHAP hard to
  read here.

  ALE (Accumulated Local Effects) fixes this. For a feature f:
    1. bin f along its REAL range (quantile bins -> equal data per bin);
    2. in each bin, take only rows that fall in it, and move f from the bin's
       lower edge to its upper edge (a LOCAL nudge, within real data);
    3. record the local change in prediction (a difference);
    4. subtract the per-bin mean (centering) and ACCUMULATE across bins.
  Result: an ALE curve f -> delta predicted IRD, with correlated-feature
  effects differenced out. Stays on-manifold, so collinearity no longer
  poisons it.

  We also emit a density-aware SWEEP (predicted IRD vs feature over its range)
  with a data rug, and keep the v1 per-SD number for side-by-side contrast.

Only EXOGENOUS / manipulable features are analysed (weather + schedule);
endogenous/derived/anchor features are excluded (documented below).

Run:  python analysis/sensitivity.py
Outputs: console + output/sensitivity/{ale_curves.png, sensitivity_table.csv}
"""

from __future__ import annotations
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\user\PycharmProjects\sat-ir-prediction")

from models.model1_decay import (
    load_selected_basins, load_data, prepare_splits, train_model,
)
from models.utils import predict, back_transform

import matplotlib.pyplot as plt

DEFAULT_COLOR = "#065A82"

OUT_DIR = r"C:\Users\user\PycharmProjects\sat-ir-prediction\output\sensitivity"

# Analyse the exogenous set. prev_ALPHA (=DrT/Ct) OR prev_DrT - not both.
SAFE_FEATURES = ["prev_TD", "prev_RD", "prev_RW", "prev_ALPHA", "prev_FT"]

FEATURE_UNITS = {
    "prev_TD":    ("prev. drying air temp.", "deg C"),
    "prev_RD":    ("prev. drying radiation", "W m-2"),
    "prev_RW":    ("prev. wetting radiation", "W m-2"),
    "prev_DrT":   ("prev. drying time", "h"),
    "prev_ALPHA": ("prev. drying fraction", "-"),
    "prev_FT":    ("prev. flooding time", "h"),
}

EXCLUDED_NOTE = (
    "Excluded (endogenous / derived / anchor): prev_HL (endogenous with IRD), "
    "all cum_* (path integrals), IRD_at_reset (scale anchor); of {prev_DrT, "
    "prev_ALPHA} only one is analysed to avoid double-counting drying."
)


def _pred_ird(model, scaler, feat_cols, df):
    ird_reset = pd.to_numeric(df["IRD_at_reset"], errors="coerce").values
    pred_norm = predict(model, scaler, feat_cols, df)
    return back_transform(ird_reset, pred_norm)


def ale_1d(test, model, scaler, feat_cols, feature, n_bins=20):
    """1-D ALE curve for `feature`, in IRD units (cm/h).
    Returns (edges, ale_values, bin_counts, bin_centers) or None."""
    base = test.copy()
    valid = base[feat_cols].notna().all(axis=1)
    base = base.loc[valid].reset_index(drop=True)

    x = pd.to_numeric(base[feature], errors="coerce").to_numpy()
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if len(edges) < 3:
        return None
    n_bins = len(edges) - 1
    idx = np.clip(np.digitize(x, edges, right=False) - 1, 0, n_bins - 1)

    local_effects = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx == b
        counts[b] = int(m.sum())
        if counts[b] == 0:
            continue
        lo = base.loc[m].copy(); lo[feature] = edges[b]
        hi = base.loc[m].copy(); hi[feature] = edges[b + 1]
        d = _pred_ird(model, scaler, feat_cols, hi) - _pred_ird(model, scaler, feat_cols, lo)
        local_effects[b] = np.nanmean(d)

    ale = np.cumsum(local_effects)
    ale = np.concatenate([[0.0], ale])
    mid = 0.5 * (ale[:-1] + ale[1:])
    ale -= np.average(mid, weights=np.maximum(counts, 1))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, ale, counts, bin_centers


def per_sd_number(test, model, scaler, feat_cols, feature):
    base = test.copy()
    valid = base[feat_cols].notna().all(axis=1)
    base = base.loc[valid].reset_index(drop=True)
    ird0 = _pred_ird(model, scaler, feat_cols, base)
    sd = float(pd.to_numeric(base[feature], errors="coerce").std())
    bump = base.copy()
    bump[feature] = pd.to_numeric(bump[feature], errors="coerce") + sd
    d = _pred_ird(model, scaler, feat_cols, bump) - ird0
    return float(np.nanmedian(d)), sd


def plot_ale(results, save_path=None):
    feats = [f for f in results if results[f]["ale"] is not None]
    n = len(feats)
    if n == 0:
        print("  [no ALE curves to plot]"); return
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4), squeeze=False)
    for ax, feat in zip(axes[0], feats):
        edges, ale, counts, centers = results[feat]["ale"]
        ax.plot(edges, ale, "-o", color=DEFAULT_COLOR, markersize=3)
        ax.axhline(0, color="gray", lw=0.6, alpha=0.6)
        ax2 = ax.twinx()
        ax2.bar(centers, counts, width=np.diff(edges), alpha=0.12,
                color="gray", align="center")
        ax2.set_yticks([])
        label, unit = FEATURE_UNITS.get(feat, (feat, ""))
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlabel(unit, fontsize=8)
        ax.set_ylabel("ALE on IRD (cm/h)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Accumulated Local Effects - Model 1 (held-out, collinearity-robust)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved figure: {save_path}")
    plt.show()


def main():
    print("=" * 70)
    print("  ENGINEERING SENSITIVITY v2 (ALE) - Model 1, real held-out")
    print("=" * 70)

    held_out = load_selected_basins()
    df, feat_cols = load_data()
    train, val, test = prepare_splits(df, held_out, feat_cols)
    model, scaler = train_model(train, val, feat_cols)
    print("\n  " + EXCLUDED_NOTE + "\n")

    results = {}
    rows = []
    for f in SAFE_FEATURES:
        if f not in test.columns:
            print(f"  [skip] {f} absent"); continue
        ale = ale_1d(test, model, scaler, feat_cols, f)
        d_sd, sd = per_sd_number(test, model, scaler, feat_cols, f)
        results[f] = {"ale": ale}
        ale_range = np.nan if ale is None else float(np.nanmax(ale[1]) - np.nanmin(ale[1]))
        label, unit = FEATURE_UNITS.get(f, (f, ""))
        rows.append({"feature": f, "label": label, "unit": unit, "sd": sd,
                     "ALE_range_cmph": ale_range,
                     "v1_dIRD_per_SD_median": d_sd})

    tbl = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print("=== ALE effect size vs v1 per-SD (contrast) ===")
        print(tbl.to_string(index=False))
        print("\n  Read: ALE_range = span of the ALE curve over the feature's real range")
        print("  (honest effect size). v1_dIRD_per_SD is the naive one-at-a-time number")
        print("  kept only to show how collinearity distorted it.")

    os.makedirs(OUT_DIR, exist_ok=True)
    tbl.to_csv(os.path.join(OUT_DIR, "sensitivity_table.csv"), index=False)
    print(f"\n  Saved table:  {os.path.join(OUT_DIR, 'sensitivity_table.csv')}")

    plot_ale(results, save_path=os.path.join(OUT_DIR, "ale_curves.png"))
    return tbl, results


if __name__ == "__main__":
    main()