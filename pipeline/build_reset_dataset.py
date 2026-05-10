"""
pipeline/build_reset_dataset.py — Build reset_dataset.csv for Model 2
======================================================================
Builds one row per reset event from the event_dataset.csv produced by
build_dataset.py. Each row aggregates operational and weather features
from the segment that just ended (segment i-1), and records the target
IRD_norm_log_reset of the next segment (segment i).

Target
------
  IRD_norm_log_reset = log(IRD_at_reset[i] / IRD_at_reset[i-1])

  Dimensionless log-ratio between consecutive reset values.
    = 0   : recovery exactly matches the previous reset
    > 0   : better recovery than last time
    < 0   : worse recovery than last time

  Physical rationale for log-ratio normalization:
    Different basins have fundamentally different hydraulic conductivity
    ceilings (Ks). Basin A may max out at 4 cm/h while basin B maxes out
    at 10 cm/h. Training on raw IRD_at_reset means the model partially
    learns "which basin is this" rather than "how well did this basin
    recover relative to its own potential."

    The log-ratio removes between-basin scale differences and expresses
    recovery relative to the basin's own recent history — consistent with
    Model 1 which also uses a log-ratio target (IRD_norm_log).

    Raw IRD_at_reset was tested and rejected: LogPrevRatio gives better
    MAPE (16.8% vs 21.0%) and is more physically interpretable across
    the heterogeneous basin system (see normalization_analysis.py).

  Back-transform:
    IRD_at_reset[i] = IRD_at_reset[i-1] * exp(IRD_norm_log_reset)

  First reset per basin: no previous reset -> NaN -> excluded from training.

Physical rationale
------------------
Model 2 answers: "Given how this segment was operated, what IRD_norm_log_reset
will the basin achieve after the next reset/tillage?"

The log-ratio target measures recovery relative to the previous reset,
capturing the operational and environmental factors that drive improvement
or degradation in basin performance over consecutive segments.

Cross-segment history features (prev_IRD_at_reset, IRD_direction) are
carried forward from event_dataset.csv where they were computed BEFORE
the quality filter — ensuring the chronological sequence is preserved.
prev_IRD_at_reset serves both as a feature AND as the back-transform
denominator.

Split strategies
----------------
Two splits are computed and stored as separate columns:

  split_chrono — chronological per basin: first 70% -> train,
                 next 15% -> val, last 15% -> test.
                 Realistic for operational planning: model is trained
                 on the past and tested on the future.

  split_random — random per basin (seed=42): same fractions, shuffled.
                 Used for comparison only — chrono is the primary split.

Why chronological?
  IRD_at_reset is autocorrelated. A random split leaks future information
  into training, artificially inflating performance. Chronological split
  gives an honest forward-looking test.

Seasonality encoding
--------------------
  month_sin = sin(2*pi*(month - 4) / 12)  — peak July (+1.0)
  month_cos = cos(2*pi*(month - 4) / 12)  — orthogonal component

Usage
-----
  python -m pipeline.build_reset_dataset
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EVENT_CSV, RESET_CSV, TABLES_DIR,
    TRAIN_FRAC, VAL_FRAC, RANDOM_SEED,
    SEASON_PHASE, FIELD_NAMES,
    HELD_OUT_BASIN_LIST,
)
from pipeline.features import (
    MODEL2_SEGMENT_FEATURES,
    MODEL2_HISTORY_FEATURES,
    MODEL2_SEASON_FEATURES,
    MODEL2_FEATURES,
    TARGET_M2,
)


def _load_outlier_basins() -> set[int]:
    """Load outlier_basins.csv. Returns empty set if not found."""
    from config import OUTLIER_CSV
    if not OUTLIER_CSV.exists():
        return set()
    excluded = set()
    with open(OUTLIER_CSV) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("basin"):
                continue
            try:
                excluded.add(int(line.split(",")[0].strip()))
            except ValueError:
                pass
    return excluded



# ─────────────────────────────────────────────────────────────────────────────
# Seasonality encoding
# ─────────────────────────────────────────────────────────────────────────────

def _month_encoding(month: int) -> tuple[float, float]:
    """
    Circular encoding of calendar month.
    Peak at July (month=7): month_sin = +1.0
    Trough at January (month=1): month_sin = -1.0
    """
    sin_val = float(np.sin(2 * np.pi * (month - SEASON_PHASE) / 12))
    cos_val = float(np.cos(2 * np.pi * (month - SEASON_PHASE) / 12))
    return sin_val, cos_val


# ─────────────────────────────────────────────────────────────────────────────
# Per-basin reset dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_basin_resets(
    bn:             int,
    bdf:            pd.DataFrame,
    held_out_set:   set[int],
    outlier_set:    set[int],
) -> list[dict]:
    """
    Build one row per reset event for a single basin.
    Tags each row with basin_role: 'held_out', 'outlier', or 'clean'.

        Build one row per reset event for a single basin.

    For each reset event (segment i):
      - Target:   IRD_norm_log_reset = log(IRD_at_reset[i] / IRD_at_reset[i-1])
      - Features: aggregated from segment i-1 events
      - History:  prev_IRD_at_reset, IRD_direction from event_dataset.csv

    All segments are used (good and bad) to preserve chronological order.
    Using only good segments would create gaps in the history sequence.

    First reset per basin: no previous reset -> target = NaN -> excluded.

    Returns list of dicts (one per usable reset).
    """

    bdf = bdf.sort_values("opening_valve_date").reset_index(drop=True)

    reset_rows  = bdf[bdf["row_type"] == "reset"].copy()
    all_seg_ids = sorted(s for s in bdf["segment_id"].unique() if s >= 0)

    if len(reset_rows) < 2 or len(all_seg_ids) < 2:
        return []

    # Determine basin role
    if bn in held_out_set:
        basin_role = "held_out"
    elif bn in outlier_set:
        basin_role = "outlier"
    else:
        basin_role = "clean"

    rows = []
    for _, reset_row in reset_rows.iterrows():
        sid = int(reset_row["segment_id"])
        if sid not in all_seg_ids:
            continue

        sid_idx = all_seg_ids.index(sid)
        if sid_idx == 0:
            continue

        prev_sid    = all_seg_ids[sid_idx - 1]
        prev_events = bdf[
            (bdf["segment_id"] == prev_sid) &
            (bdf["row_type"] == "event")
        ]
        if prev_events.empty:
            continue

        row = _build_row(bn, reset_row, prev_events, all_seg_ids)
        if row is not None:
            row["basin_role"] = basin_role
            rows.append(row)

    return rows



def _build_row(
    bn: int,
    reset_row: pd.Series,
    prev_events: pd.DataFrame,
    all_seg_ids: list[int],
) -> dict | None:
    """
    Build one reset-level feature row.

    Target: IRD_norm_log_reset = log(IRD_at_reset[i] / IRD_at_reset[i-1])
    Returns None if target cannot be computed (missing/non-positive values).

    The raw IRD_at_reset[i] is also stored for back-transform and reporting.
    """
    # Current reset's IRD_at_reset (target numerator)
    ird_current = pd.to_numeric(
        reset_row.get("IRD_at_reset", np.nan), errors="coerce"
    )
    # Previous reset's IRD_at_reset (denominator, also a feature)
    ird_prev = pd.to_numeric(
        reset_row.get("prev_IRD_at_reset", np.nan), errors="coerce"
    )

    # Validate — both must be positive and finite for log-ratio
    if not (np.isfinite(ird_current) and ird_current > 0):
        return None
    if not (np.isfinite(ird_prev) and ird_prev > 0):
        return None  # first reset per basin or missing prev

    # Compute log-ratio target
    ird_norm_log_reset = float(np.log(ird_current / ird_prev))

    def _agg(col: str, func: str) -> float:
        if col not in prev_events.columns:
            return np.nan
        s = pd.to_numeric(prev_events[col], errors="coerce").dropna()
        if s.empty:
            return np.nan
        return float(getattr(s, func)())

    def _last(col: str) -> float:
        if col not in prev_events.columns:
            return np.nan
        sorted_ev = prev_events.sort_values("LCT")
        val = pd.to_numeric(sorted_ev[col].iloc[-1], errors="coerce")
        return float(val) if np.isfinite(val) else np.nan

    lct = pd.to_numeric(prev_events.get("LCT", pd.Series()), errors="coerce")

    reset_month = pd.to_datetime(reset_row["opening_valve_date"]).month
    month_sin, month_cos = _month_encoding(reset_month)

    row = {
        # Metadata
        "basin_number":    bn,
        "field_name":      FIELD_NAMES.get(int(str(bn)[0]), str(bn)),
        "segment_id":      int(reset_row["segment_id"]),
        "reset_date":      reset_row["opening_valve_date"],
        "is_good_segment": bool(reset_row.get("is_good_segment", False)),

        # Target — log-ratio (dimensionless)
        TARGET_M2:          ird_norm_log_reset,   # IRD_norm_log_reset

        # Raw IRD values stored for back-transform and reporting
        # Back-transform: IRD_at_reset[i] = prev_IRD_at_reset * exp(TARGET_M2)
        "IRD_at_reset":     float(ird_current),   # actual cm/h (for reporting)
        "prev_IRD_at_reset_raw": float(ird_prev), # denominator for back-transform

        # Aggregated segment i-1 features
        "mean_DrT":   _agg("DrT",   "mean"),
        "mean_FT":    _agg("FT",    "mean"),
        "mean_ALPHA": _agg("ALPHA", "mean"),
        "mean_HL":    _agg("HL",    "mean"),
        "sum_DrT":    _agg("DrT",   "sum"),
        "sum_FT":     _agg("FT",    "sum"),
        "min_DrT":    _agg("DrT",   "min"),
        "max_FT":     _agg("FT",    "max"),
        "n_events":   len(prev_events),
        "total_LCT":  float(lct.max()) if not lct.isna().all() else np.nan,
        "mean_RD":    _agg("RD",    "mean"),
        "mean_TW":    _agg("TW",    "mean"),
        "mean_TD":    _agg("TD",    "mean"),

        # Last-event features
        "last_DrT":   _last("DrT"),
        "last_RD":    _last("RD"),

        # Cross-segment history (from event_dataset.csv, pre-quality-filter)
        "prev_IRD_at_reset":      _to_float(reset_row.get("prev_IRD_at_reset")),
        "prev_prev_IRD_at_reset": _to_float(reset_row.get("prev_prev_IRD_at_reset")),
        "IRD_direction":          _to_float(reset_row.get("IRD_direction")),

        # Seasonality at reset date
        "month_sin": month_sin,
        "month_cos": month_cos,

        # Daily ambient conditions at reset moment
        "DAT": _to_float(reset_row.get("DAT")),
        "DAR": _to_float(reset_row.get("DAR")),
        # Basin role — set in _build_basin_resets
        "basin_role": "clean",  # placeholder, overwritten below
    }

    return row


def _to_float(val) -> float:
    """Safely convert a value to float, returning NaN if not finite."""
    try:
        f = float(val)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Split strategies
# ─────────────────────────────────────────────────────────────────────────────

def add_chronological_split(reset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign train/val/test splits chronologically per basin.
    First 70% of resets -> train, next 15% -> val, last 15% -> test.
    """
    reset_df = reset_df.copy()
    reset_df["split_chrono"] = "test"

    for bn, bdf in reset_df.groupby("basin_number"):
        bdf  = bdf.sort_values("reset_date")
        n    = len(bdf)
        n_tr = max(1, round(n * TRAIN_FRAC))
        n_va = max(1, round(n * VAL_FRAC))
        reset_df.loc[bdf.index[:n_tr],          "split_chrono"] = "train"
        reset_df.loc[bdf.index[n_tr:n_tr+n_va], "split_chrono"] = "val"

    counts = reset_df["split_chrono"].value_counts().to_dict()
    print(f"  Chrono split : {counts}")
    return reset_df


def add_random_split(reset_df: pd.DataFrame) -> pd.DataFrame:
    """Assign train/val/test splits randomly per basin (seed=42)."""
    reset_df = reset_df.copy()
    reset_df["split_random"] = "test"
    rng = np.random.default_rng(RANDOM_SEED)

    for bn, bdf in reset_df.groupby("basin_number"):
        n    = len(bdf)
        idx  = rng.permutation(bdf.index)
        n_tr = max(1, round(n * TRAIN_FRAC))
        n_va = max(1, round(n * VAL_FRAC))
        reset_df.loc[idx[:n_tr],          "split_random"] = "train"
        reset_df.loc[idx[n_tr:n_tr+n_va], "split_random"] = "val"

    counts = reset_df["split_random"].value_counts().to_dict()
    print(f"  Random split : {counts}")
    return reset_df



def add_held_out_split(reset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign split_held_out column for held-out basin generalisation test.

    Logic (mirrors Model 1 condition D/E):
      held_out basins  → all resets tagged 'held_out_test'
                         (ALL resets used as test, not just last 15%)
      outlier basins   → 'excluded'
      clean basins     → same as split_chrono (train/val/test)

    Using ALL resets from held-out basins as test is necessary because
    each held-out basin has only ~45 resets total. The last 15% (chrono)
    would give only ~7 events per basin — too few for reliable metrics.
    Using all resets gives ~45 × 5 = ~225 held-out test events.

    NOTE: held-out basins are NEVER in training.
    """
    reset_df = reset_df.copy()
    reset_df["split_held_out"] = reset_df["split_chrono"]  # default

    held_out_mask = reset_df["basin_role"] == "held_out"
    outlier_mask  = reset_df["basin_role"] == "outlier"

    reset_df.loc[held_out_mask, "split_held_out"] = "held_out_test"
    reset_df.loc[outlier_mask,  "split_held_out"] = "excluded"

    counts = reset_df["split_held_out"].value_counts().to_dict()
    print(f"  Held-out split: {counts}")
    n_ho = int(held_out_mask.sum())
    print(f"  Held-out basins: {sorted(reset_df.loc[held_out_mask, 'basin_number'].unique().tolist())}  ({n_ho} resets → all used as test)")
    return reset_df


# ─────────────────────────────────────────────────────────────────────────────
# Feature coverage report
# ─────────────────────────────────────────────────────────────────────────────



def print_feature_coverage(reset_df: pd.DataFrame) -> None:
    """Print NaN coverage for each Model 2 feature."""
    print("\n  Feature coverage (NaN check):")
    print(f"  {'Feature':<30} {'NaN':>6}  {'%':>6}  {'Status'}")
    print(f"  {'-'*55}")
    for feat in MODEL2_FEATURES:
        if feat not in reset_df.columns:
            print(f"  {feat:<30} {'MISSING':>6}")
            continue
        n_nan = int(reset_df[feat].isna().sum())
        pct   = 100.0 * n_nan / len(reset_df)
        flag  = "  WARNING HIGH" if pct > 20 else ""
        print(f"  {feat:<30} {n_nan:>6}  {pct:>5.1f}%{flag}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    """Build reset_dataset.csv from event_dataset.csv."""
    print("=" * 65)
    print("  BUILD RESET DATASET — pipeline/build_reset_dataset.py")
    print(f"  Target: {TARGET_M2} = log(IRD_at_reset[i] / IRD_at_reset[i-1])")
    print("=" * 65)

    if not EVENT_CSV.exists():
        raise FileNotFoundError(
            f"{EVENT_CSV} not found.\n"
            "Run: python -m pipeline.build_dataset --rebuild"
        )

    df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"  Event dataset : {len(df)} rows  |  "
          f"{df['basin_number'].nunique()} basins")

    # Load outlier and held-out basin sets
    held_out_set = set(HELD_OUT_BASIN_LIST)
    outlier_set  = _load_outlier_basins()
    print(f"  Held-out basins : {sorted(held_out_set)}")
    print(f"  Outlier basins  : {sorted(outlier_set)}")

    # Build reset rows per basin
    print("\n  Building reset rows per basin...")
    all_rows: list[dict] = []
    for bn, bdf in df.groupby("basin_number"):
        rows = _build_basin_resets(
            int(bn), bdf, held_out_set, outlier_set
        )
        all_rows.extend(rows)
        if rows:
            role = rows[0].get("basin_role", "clean")
            print(f"    [{int(bn):>5}]  {len(rows)} reset rows  role={role}")

    if not all_rows:
        raise ValueError("No reset rows built — check event_dataset.csv.")

    reset_df = pd.DataFrame(all_rows).reset_index(drop=True)
    print(f"\n  Reset dataset : {len(reset_df)} rows  |  "
          f"{reset_df['basin_number'].nunique()} basins")
    print(f"  Avg resets/basin : "
          f"{len(reset_df) / reset_df['basin_number'].nunique():.1f}")

    # Role summary
    role_counts = reset_df["basin_role"].value_counts().to_dict()
    print(f"  Basin roles   : {role_counts}")

    # Add splits
    print()
    reset_df = add_chronological_split(reset_df)
    reset_df = add_random_split(reset_df)
    reset_df = add_held_out_split(reset_df)

    # Feature coverage
    print_feature_coverage(reset_df)

    # Target distribution summary
    tgt = reset_df[TARGET_M2].dropna()
    print(f"\n  Target ({TARGET_M2}) distribution:")
    print(f"    mean={tgt.mean():.4f}  median={tgt.median():.4f}  "
          f"std={tgt.std():.4f}  min={tgt.min():.4f}  max={tgt.max():.4f}")
    print(f"    Interpretation: mean~0 means recovery typically matches previous reset")
    print(f"    Positive skew = some resets recover much better than previous")

    # Raw IRD_at_reset stats
    raw_tgt = reset_df["IRD_at_reset"].dropna()
    print(f"\n  Raw IRD_at_reset (cm/h):")
    print(f"    mean={raw_tgt.mean():.2f}  median={raw_tgt.median():.2f}  "
          f"std={raw_tgt.std():.2f}  min={raw_tgt.min():.2f}  "
          f"max={raw_tgt.max():.2f}")

    # Naive baseline
    valid = (
        reset_df[TARGET_M2].notna() &
        (reset_df["IRD_at_reset"] > 0) &
        (reset_df["prev_IRD_at_reset_raw"].notna()) &
        (reset_df["prev_IRD_at_reset_raw"] > 0)
    )
    if valid.sum() > 10:
        from sklearn.metrics import r2_score

        y_true_log  = reset_df.loc[valid, TARGET_M2].values
        y_naive_log = np.zeros(valid.sum())
        r2_naive_log = r2_score(y_true_log, y_naive_log)

        y_true_raw  = reset_df.loc[valid, "IRD_at_reset"].values
        y_naive_raw = reset_df.loc[valid, "prev_IRD_at_reset_raw"].values
        finite = (
            np.isfinite(y_true_raw) & np.isfinite(y_naive_raw) &
            (y_true_raw > 0) & (y_naive_raw > 0)
        )
        r2_naive_raw = r2_score(y_true_raw[finite], y_naive_raw[finite])

        print(f"\n  Naive baseline (predict log-ratio = 0):")
        print(f"    R² (log-ratio space) : {r2_naive_log:.3f}")
        print(f"    R² (raw IRD cm/h)    : {r2_naive_raw:.3f}")
        print(f"    → Model 2 must beat these thresholds to add value.")

    # Held-out split summary
    print(f"\n  Held-out split summary:")
    for split_val in ["train", "val", "test", "held_out_test", "excluded"]:
        n = int((reset_df["split_held_out"] == split_val).sum())
        if n > 0:
            print(f"    {split_val:<15}: {n} resets")

    # Per-field summary
    print(f"\n  Per-field reset counts:")
    field_summary = reset_df.groupby(
        ["field_name", "basin_role"]
    )["basin_number"].count().reset_index()
    field_summary.columns = ["field", "role", "n_resets"]
    for _, row in field_summary.iterrows():
        print(f"    {row['field']:<12}  {row['role']:<10}  {int(row['n_resets'])} resets")

    # Save
    reset_df.to_csv(RESET_CSV, index=False)
    print(f"\n  Saved: {RESET_CSV}  ({len(reset_df)} rows)")

    # Summary table per basin
    summary = reset_df.groupby("basin_number").agg(
        n_resets       = (TARGET_M2,      "count"),
        mean_log_ratio = (TARGET_M2,      "mean"),
        std_log_ratio  = (TARGET_M2,      "std"),
        mean_IRD_reset = ("IRD_at_reset", "mean"),
        field_name     = ("field_name",   "first"),
        basin_role     = ("basin_role",   "first"),
    ).reset_index()
    summary_path = TABLES_DIR / "reset_dataset_summary.xlsx"
    summary.to_excel(summary_path, index=False)
    print(f"  Saved: {summary_path.name}")

    return reset_df


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()