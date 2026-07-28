"""
analysis/run_depth_on_heldout.py
================================
Model-vs-naive depth-stratified baseline, on the SELECTED-5 held-out basins,
using your existing Model 1 pipeline verbatim. No synthetic data.

It calls the same functions model1_decay.main() calls:
    load_selected_basins -> load_data -> prepare_splits
    -> train_model -> evaluate_held_out
then feeds the returned held-out `test` frame (which already carries
ird_pred / ird_true / segment_id / LCT / basin_number) into depth_table().

Run in PyCharm:
    python analysis/run_depth_on_heldout.py
"""

from __future__ import annotations
import sys
import pandas as pd

# make sure project root is importable (mirror your other scripts)
sys.path.insert(0, r"C:\Users\user\PycharmProjects\sat-ir-prediction")

from models.model1_decay import (
    load_selected_basins, load_data, prepare_splits,
    train_model, evaluate_held_out,
)
from analysis.depth_baseline import depth_table

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_COLOR = "#065A82"   # model (matches model1_decay)
NAIVE_COLOR   = "#AAAAAA"   # naive-at-lag
DELTA_COLOR   = "#E07B39"   # advantage


def plot_depth(tbl: pd.DataFrame, save_path: str | None = None) -> None:
    """Two-panel depth figure in the paper's house style.
    Left: RMSE vs depth for model and naive-at-lag. Right: delta-RMSE vs depth."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("Model 1 vs naive-at-lag: skill as a function of within-segment "
                 "depth (held-out basins)", fontsize=10, fontweight="bold")

    d = tbl["depth"].to_numpy()
    axL.plot(d, tbl["rmse_naive"], "-o", color=NAIVE_COLOR, markersize=4,
             label="Naive-at-lag n")
    axL.plot(d, tbl["rmse_model"], "-o", color=DEFAULT_COLOR, markersize=4,
             label="Model 1 (one-step)")
    axL.set_xlabel("Within-segment depth n (events)", fontsize=9)
    axL.set_ylabel("RMSE (cm h$^{-1}$)", fontsize=9)
    axL.grid(True, alpha=0.2); axL.legend(fontsize=8, loc="upper left")
    axL.tick_params(labelsize=8)

    axR.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    axR.plot(d, tbl["delta_rmse"], "-o", color=DELTA_COLOR, markersize=4)
    axR.set_xlabel("Within-segment depth n (events)", fontsize=9)
    axR.set_ylabel("delta-RMSE = naive - model (cm h$^{-1}$)", fontsize=9)
    axR.grid(True, alpha=0.2); axR.tick_params(labelsize=8)
    for i in range(0, len(d), max(1, len(d)//6)):
        axR.annotate(f"n={int(tbl['n_events'].iloc[i]):,}",
                     xy=(d[i], tbl["delta_rmse"].iloc[i]),
                     xytext=(0, 6), textcoords="offset points",
                     fontsize=6.5, ha="center", color="gray")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved figure: {save_path}")
    plt.show()


def main(max_depth: int = 15, save_fig: str | None = None):
    print("=" * 70)
    print("  DEPTH BASELINE on SELECTED-5 held-out basins (real data)")
    print("=" * 70)

    # --- your exact pipeline, up to the held-out test frame ---
    held_out = load_selected_basins()
    df, feat_cols = load_data()
    train, val, test = prepare_splits(df, held_out, feat_cols)
    model, scaler = train_model(train, val, feat_cols)
    _pooled, _basin_df, test_df = evaluate_held_out(
        test, model, scaler, feat_cols, held_out
    )

    # --- depth-stratified model vs naive-at-lag, held-out only ---
    tbl = depth_table(
        test_df,
        max_depth=max_depth,
        basin_col="basin_number",
        segment_col="segment_id",
        order_col="LCT",
        ird_col="ird_true",        # truth column produced by evaluate_held_out
        model_pred_col="ird_pred", # Model 1 prediction produced by evaluate_held_out
    )

    with pd.option_context("display.max_columns", None, "display.width", 140):
        print("\n=== Depth-stratified: Model 1 vs naive-at-lag (HELD-OUT 5) ===")
        print(tbl.to_string(index=False))

    # quick read-out
    if "delta_rmse" in tbl.columns:
        d1 = tbl.loc[tbl["depth"] == 1, "delta_rmse"].values
        dN = tbl.loc[tbl["depth"] == tbl["depth"].max(), "delta_rmse"].values
        if len(d1) and len(dN):
            print(f"\n  ΔRMSE at depth 1 : {d1[0]:+.3f} cm/h")
            print(f"  ΔRMSE at depth {int(tbl['depth'].max())}: {dN[0]:+.3f} cm/h")
            trend = "widens" if dN[0] > d1[0] else "flat/narrows"
            print(f"  Model advantage over naive {trend} with depth.")

    # figure (house style). Pass save_fig=r"...\figures\depth_baseline.png" to save.
    try:
        plot_depth(tbl, save_path=save_fig)
    except Exception as e:
        print(f"  [plot skipped: {e}]")

    return tbl


if __name__ == "__main__":
    main()