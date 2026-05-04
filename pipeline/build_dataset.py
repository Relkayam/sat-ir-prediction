"""
pipeline/build_dataset.py — Build event_dataset.csv from DuckDB
================================================================
Reads all flooding events for all basins, applies quality filters,
identifies segments, computes features, and saves a single CSV.

Every raw event is KEPT in the CSV with a filter_reason column explaining
why it was excluded (if at all). This full-retention policy ensures:
  - Complete reproducibility: the filter funnel can be reconstructed exactly
  - Supplementary material: excluded events are documented for the paper
  - Sensitivity analysis: re-filtering from the CSV is faster than re-querying

Pipeline (per basin)
--------------------
  Step 1  Read ALL events from DuckDB
  Step 2  Tag quality-filtered events (CIV, Ct, AL, date cutoff)
          Events are KEPT with filter_reason set — not dropped
  Step 3  Identify segments on quality-passed events only
  Step 4  Compute IRD_norm = log(IRD / IRD_at_reset)
  Step 5  Compute HL = CIV / basin_area (cm)
  Step 6  Compute prev_* features (shift within segment)
  Step 7  Compute cum_* features (cumsum within segment)
  Step 7b Compute cross-segment features BEFORE segment quality filter
          (ensures IRD_direction uses true chronological sequence)
  Step 8  Tag segment quality (Pearson r, exponential fit, R²)
  Step 9  Assign 70/15/15 train/val/test split (random by segment)

filter_reason values
--------------------
  ""                    — good event in a good segment (used for training)
  "after_cutoff"        — opening_valve_date >= DATA_CUTOFF
  "quality_filter_CIV"  — CIV < 3000 m³
  "quality_filter_Ct"   — Ct < 20 h
  "quality_filter_AL"   — AL < 5 cm
  "pre_segment"         — before first credible reset in basin
  "too_few_events"      — segment has fewer than MIN_EVENTS_PER_SEGMENT events
  "pearson_r_positive"  — IRD not declining with LCT (r > PEARSON_OUTLIER_THRESHOLD)
  "fit_failed"          — exponential decay fit did not converge
  "r2_below_threshold"  — decay fit R² < MIN_SEGMENT_R2

Usage
-----
  python -m pipeline.build_dataset              # uses cached CSV if exists
  python -m pipeline.build_dataset --rebuild    # forces rebuild from DuckDB
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_CUTOFF, QUALITY_FILTER,
    MIN_EVENTS_PER_SEGMENT, MIN_SEGMENT_R2, PEARSON_OUTLIER_THRESHOLD,
    TRAIN_FRAC, VAL_FRAC, RANDOM_SEED,
    EVENT_CSV, TABLES_DIR, FIELD_NAMES,
    HELD_OUT_BASIN_LIST,
)

# import os
# import sys
# add to path C:\Users\user\PycharmProjects\mek-models-satix-backend\optisat
# import sys
# sys.path.insert(0, r'C:\Users\user\PycharmProjects\mek-models-satix-backend')
#
# import os
# optisat_path = r'C:\Users\user\PycharmProjects\mek-models-satix-backend\optisat'
# print("optisat folder exists:", os.path.exists(optisat_path))
# print("__init__.py exists:", os.path.exists(os.path.join(optisat_path, '__init__.py')))

# Now try the import
from optisat.db.duckdb_manager import DuckDBManager
print("Import successful")
from pipeline.features import PREV_SOURCE_COLS, CUM_SOURCE_COLS, RAW_DB_COLUMNS

from optisat.db.duckdb_manager import DuckDBManager
from optisat.db.paths import SatixPaths

# Tillage pipeline — needed for ceiling reset classification
from optisat.etl.features.tillage_features import (
    read_clean_tillage_events,
    _add_scoring_features,
)
from optisat.etl.features.constants import Constants as _TillageConstants


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Read raw events from DuckDB
# ─────────────────────────────────────────────────────────────────────────────

def _get_basin_area(basin_number: int) -> float:
    """Return basin area in m² from the global DuckDB. Defaults to 1.0 if missing."""
    try:
        with DuckDBManager(
            str(SatixPaths.GLOBAL_DB_PATH)
        ).connect_context() as db:
            res = db.read_dataframe(
                f"SELECT area FROM basins WHERE basin_number = {basin_number}"
            )
        if not res.empty and pd.notna(res["area"].iloc[0]):
            return float(res["area"].iloc[0])
    except Exception:
        pass
    print(f"    [basin {basin_number}] WARNING: area not found — defaulting to 1.0")
    return 1.0


def _read_raw_events(basin_number: int) -> Optional[pd.DataFrame]:
    """
    Read all events for one basin from its DuckDB file.
    Returns None if the file does not exist.
    Only columns in RAW_DB_COLUMNS are fetched (others silently skipped).
    """
    path = SatixPaths.BASIN_DB_DIR / f"basin_{int(basin_number)}.duckdb"
    if not path.exists():
        return None

    with DuckDBManager(str(path)).connect_context() as db:
        schema = db.read_dataframe("PRAGMA table_info('features')")
        available = set(schema["name"].tolist())
        cols = [c for c in RAW_DB_COLUMNS if c in available]
        df = db.read_dataframe(
            f"SELECT {', '.join(cols)} FROM features "
            f"ORDER BY opening_valve_date"
        )

    if df.empty:
        return None

    df["opening_valve_date"] = pd.to_datetime(df["opening_valve_date"])
    return df.sort_values("opening_valve_date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1b — Fill drying-window features with 0 when DrT is zero or near-zero
# ─────────────────────────────────────────────────────────────────────────────

# Threshold below which drying time is treated as zero (hours).
# 6 minutes — covers rounding and timestep artifacts in the IoT data.
_NEAR_ZERO_DRT = 0.1

# Drying-window columns that are physically zero when there is no drying phase.
# When DrT=0 the window has zero width — radiation, temperature, etc. were
# never measured because the basin went straight from drainage to the next flood.
# NaN in these columns does NOT mean missing data — it means the drying phase
# did not exist. Filling with 0 is the correct physical value and prevents
# XGBoost from treating these as random missing values.
_DRYING_WINDOW_COLS = ["RD", "TD", "HD", "WDD", "WSD", "PD"]


def _fill_zero_drying(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill drying-window features with 0 where DrT == 0 or DrT < 0.1h.

    Physical reasoning:
      DrT = 0  →  no drying phase  →  no radiation/temperature exposure
      during drying.  The correct value is 0, not NaN.

    This affects ~5% of good training events (all with DrT=0).
    Verified: 98.2% of RD nulls in good events have DrT=0.
    """
    df = df.copy()
    drt = pd.to_numeric(df.get("DrT", pd.Series(dtype=float)), errors="coerce")
    zero_drt = drt.fillna(0) < _NEAR_ZERO_DRT

    for col in _DRYING_WINDOW_COLS:
        if col in df.columns:
            # Only fill where the value is currently NaN — don't overwrite real data
            missing = df[col].isna()
            df.loc[zero_drt & missing, col] = 0.0

    n_filled = int(zero_drt.sum())
    if n_filled > 0:
        cols_filled = [c for c in _DRYING_WINDOW_COLS if c in df.columns]
        # (silent — called per basin, summary printed at pool level)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Tag quality filter (keep all rows)
# ─────────────────────────────────────────────────────────────────────────────

def _tag_quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tag events that fail quality checks with a filter_reason.
    Events are NOT removed — they stay in the CSV for reproducibility.
    First failing criterion wins (order: cutoff → CIV → Ct → AL).
    Rows that pass all checks get filter_reason = "" (empty string).
    """
    df = df.copy()
    df["filter_reason"] = ""

    # Date cutoff first
    after = df["opening_valve_date"] >= pd.Timestamp(DATA_CUTOFF)
    df.loc[after, "filter_reason"] = "after_cutoff"

    # Quality thresholds — only on rows not already flagged
    for col, threshold in QUALITY_FILTER.items():
        if col not in df.columns:
            continue
        unflagged = df["filter_reason"] == ""
        fails = pd.to_numeric(df[col], errors="coerce").fillna(0) < threshold
        df.loc[unflagged & fails, "filter_reason"] = f"quality_filter_{col}"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Identify segments on quality-passed rows
# ─────────────────────────────────────────────────────────────────────────────

def _identify_segments(df_passed: pd.DataFrame) -> pd.DataFrame:
    """
    Assign segment_id and row_type to quality-passed events.
    A segment starts at a reset event (LCT == 0).
    Events before the first reset are tagged row_type='pre_segment'.

    Returns df_passed with columns added:
      row_type   — 'reset', 'event', or 'pre_segment'
      segment_id — integer (starts at 1); -1 for pre_segment rows
    """
    df = df_passed.copy()
    lct = pd.to_numeric(df["LCT"], errors="coerce")
    reset_mask = lct.fillna(-1) == 0.0

    row_type   = np.full(len(df), "pre_segment", dtype=object)
    seg_id_arr = np.full(len(df), -1, dtype=int)
    seg_counter = 0
    in_segment  = False

    for i in range(len(df)):
        if reset_mask.iloc[i]:
            seg_counter += 1
            in_segment   = True
            row_type[i]  = "reset"
        elif in_segment:
            row_type[i]  = "event"
        if in_segment:
            seg_id_arr[i] = seg_counter

    df["row_type"]   = row_type
    df["segment_id"] = seg_id_arr

    # Tag pre_segment rows with filter_reason
    df.loc[df["row_type"] == "pre_segment", "filter_reason"] = "pre_segment"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Steps 4 & 5 — IRD_norm and HL
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ird_norm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute IRD_norm = log(IRD / IRD_at_reset).
    Stored as 'IRD_norm' in the CSV.
    At a reset event this equals 0 by definition.
    As clogging progresses within a segment, it goes negative.
    """
    df = df.copy()
    ird       = pd.to_numeric(df["IRD"],          errors="coerce")
    ird_reset = pd.to_numeric(df["IRD_at_reset"], errors="coerce").replace(0, np.nan)
    df["IRD_norm"] = np.log(ird / ird_reset)
    return df


def _compute_hl(df: pd.DataFrame, area_m2: float) -> pd.DataFrame:
    """
    Compute hydraulic load: HL = CIV / area_m2 (cm).
    HL is endogenous — high HL is a consequence of high IRD, not a cause.
    It is included as a feature but must NOT be varied in heatmap analysis.
    """
    df = df.copy()
    civ       = pd.to_numeric(df["CIV"], errors="coerce")
    df["HL"]  = civ / area_m2
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — prev_* features (shift within segment)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_prev_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each column in PREV_SOURCE_COLS, compute prev_{col} by shifting
    values by 1 within each segment.
    The first event in each segment gets NaN (no previous event).
    """
    df = df.copy()
    for col in PREV_SOURCE_COLS:
        if col not in df.columns:
            continue
        df[f"prev_{col}"] = np.nan
        for sid, grp in df.groupby("segment_id"):
            if sid < 0:
                continue
            vals = pd.to_numeric(grp[col], errors="coerce").shift(1)
            df.loc[grp.index, f"prev_{col}"] = vals.values
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — cum_* features (cumsum within segment)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_cum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative sum features within each segment.
    CUM_SOURCE_COLS maps cum_col_name → source_col_name.
    Also computes event_count (position in segment, starts at 1).
    """
    df = df.copy()

    for cum_col, src_col in CUM_SOURCE_COLS.items():
        df[cum_col] = np.nan
        if src_col not in df.columns:
            continue
        for sid, grp in df.groupby("segment_id"):
            if sid < 0:
                continue
            vals = pd.to_numeric(grp[src_col], errors="coerce").fillna(0)
            df.loc[grp.index, cum_col] = vals.cumsum().values

    df["event_count"] = np.nan
    for sid, grp in df.groupby("segment_id"):
        if sid < 0:
            continue
        df.loc[grp.index, "event_count"] = np.arange(1, len(grp) + 1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 7b — Cross-segment features (BEFORE segment quality filter)
# ─────────────────────────────────────────────────────────────────────────────

def _add_cross_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add prev_IRD_at_reset, prev_prev_IRD_at_reset, prev_reset_date,
    and IRD_direction to all rows belonging to each segment.

    CRITICAL: this function is called BEFORE the segment quality filter.
    Using only good segments would create gaps in the chronological
    sequence and cause IRD_direction to reference the wrong previous reset.

    IRD_direction = (IRD_at_reset[i-1] - IRD_at_reset[i-2]) / dt
    where dt = days between reset[i-2] and reset[i-1].
    Units: cm/h per day — positive means recovery is improving.

    NaN coverage (expected):
      prev_IRD_at_reset      — NaN for first segment per basin  (~1%)
      prev_prev_IRD_at_reset — NaN for first two segments       (~2%)
      IRD_direction          — NaN for first two segments       (~2%)
    """
    df = df.copy()
    df["prev_IRD_at_reset"]      = np.nan
    df["prev_prev_IRD_at_reset"] = np.nan
    df["prev_reset_date"]        = pd.NaT
    df["IRD_direction"]          = np.nan

    # Build ordered list of (segment_id, IRD_at_reset, reset_date)
    reset_rows = (
        df[df["row_type"] == "reset"]
        .sort_values("opening_valve_date")
        [["segment_id", "IRD_at_reset", "opening_valve_date"]]
        .drop_duplicates(subset="segment_id")
        .reset_index(drop=True)
    )

    if len(reset_rows) < 2:
        return df

    for idx in range(len(reset_rows)):
        sid = int(reset_rows.loc[idx, "segment_id"])

        if idx == 0:
            vals = dict(
                prev_IRD_at_reset      = np.nan,
                prev_prev_IRD_at_reset = np.nan,
                prev_reset_date        = pd.NaT,
                IRD_direction          = np.nan,
            )

        elif idx == 1:
            prev_ird  = reset_rows.loc[idx - 1, "IRD_at_reset"]
            prev_date = reset_rows.loc[idx - 1, "opening_valve_date"]
            vals = dict(
                prev_IRD_at_reset      = float(prev_ird) if pd.notna(prev_ird) else np.nan,
                prev_prev_IRD_at_reset = np.nan,
                prev_reset_date        = prev_date,
                IRD_direction          = np.nan,
            )

        else:
            prev_ird       = float(reset_rows.loc[idx - 1, "IRD_at_reset"])
            prev_prev_ird  = float(reset_rows.loc[idx - 2, "IRD_at_reset"])
            prev_date      = reset_rows.loc[idx - 1, "opening_valve_date"]
            prev_prev_date = reset_rows.loc[idx - 2, "opening_valve_date"]

            if pd.notna(prev_date) and pd.notna(prev_prev_date):
                dt = (prev_date - prev_prev_date).total_seconds() / 86400.0
            else:
                dt = np.nan

            direction = (
                (prev_ird - prev_prev_ird) / dt
                if (np.isfinite(prev_ird) and np.isfinite(prev_prev_ird)
                    and np.isfinite(dt) and dt > 0)
                else np.nan
            )

            vals = dict(
                prev_IRD_at_reset      = prev_ird,
                prev_prev_IRD_at_reset = prev_prev_ird,
                prev_reset_date        = prev_date,
                IRD_direction          = direction,
            )

        mask = df["segment_id"] == sid
        for col, val in vals.items():
            df.loc[mask, col] = val

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Tag segment quality
# ─────────────────────────────────────────────────────────────────────────────

def _decay_model(
    lct: np.ndarray, a: float, b: float, lam: float
) -> np.ndarray:
    """Exponential decay: a·exp(-λ·LCT) + b"""
    return a * np.exp(-lam * lct) + b


def _fit_decay(lct: np.ndarray, inorm: np.ndarray) -> Optional[dict]:
    """
    Fit the exponential decay model to one segment.
    Returns dict with seg_lambda, seg_a, seg_b, seg_r2, or None if fit fails.
    """
    if len(lct) < MIN_EVENTS_PER_SEGMENT:
        return None

    a_i   = float(np.percentile(inorm, 95) - np.percentile(inorm, 5))
    b_i   = float(np.percentile(inorm, 5))
    lam_i = 1.0 / (float(np.median(lct)) + 1e-6)

    try:
        popt, _ = curve_fit(
            _decay_model, lct, inorm,
            p0=[a_i, b_i, lam_i],
            bounds=([-10, -10, 1e-7], [10, 10, 0.5]),
            maxfev=10_000, method="trf",
        )
        pred   = _decay_model(lct, *popt)
        ss_res = float(np.sum((inorm - pred) ** 2))
        ss_tot = float(np.sum((inorm - inorm.mean()) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
        return dict(
            seg_lambda = float(popt[2]),
            seg_a      = float(popt[0]),
            seg_b      = float(popt[1]),
            seg_r2     = float(r2),
        )
    except Exception:
        return None


def _tag_segment_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each segment, apply three quality checks in order:
      1. too_few_events   — fewer than MIN_EVENTS_PER_SEGMENT event rows
      2. pearson_r_positive — IRD not declining with LCT
      3. fit_failed       — exponential decay fit did not converge
      4. r2_below_threshold — fit R² < MIN_SEGMENT_R2

    Adds columns:
      is_good_segment — True if segment passed all checks
      filter_reason   — updated for rows in bad segments
      seg_lambda, seg_a, seg_b, seg_r2 — decay fit parameters (NaN if fit failed)
    """
    df = df.copy()
    df["is_good_segment"] = False

    seg_quality: dict[int, bool] = {}
    seg_reason:  dict[int, str]  = {}
    seg_decay:   dict[int, dict] = {}

    all_seg_ids = [s for s in df["segment_id"].unique() if s >= 0]

    for sid in all_seg_ids:
        grp        = df[df["segment_id"] == sid]
        event_rows = grp[grp["row_type"] == "event"]
        lct        = event_rows["LCT"].values.astype(float)
        inorm      = event_rows["IRD_norm"].values
        valid      = np.isfinite(lct) & np.isfinite(inorm) & (lct > 0)

        # Check 1: enough events
        if valid.sum() < MIN_EVENTS_PER_SEGMENT:
            seg_quality[sid] = False
            seg_reason[sid]  = "too_few_events"
            continue

        # Check 2: IRD must decline with LCT
        pr, _ = pearsonr(lct[valid], inorm[valid])
        if float(pr) > PEARSON_OUTLIER_THRESHOLD:
            seg_quality[sid] = False
            seg_reason[sid]  = "pearson_r_positive"
            continue

        # Check 3 & 4: exponential fit quality
        fit = _fit_decay(lct[valid], inorm[valid])
        if fit is None:
            seg_quality[sid] = False
            seg_reason[sid]  = "fit_failed"
            continue

        if not (np.isfinite(fit["seg_r2"]) and fit["seg_r2"] >= MIN_SEGMENT_R2):
            seg_quality[sid] = False
            seg_reason[sid]  = "r2_below_threshold"
            continue

        seg_quality[sid] = True
        seg_reason[sid]  = ""
        seg_decay[sid]   = fit

    # Apply quality tags to DataFrame
    df["is_good_segment"] = df["segment_id"].map(
        lambda s: seg_quality.get(s, False) if s >= 0 else False
    )

    # Update filter_reason for rows in bad segments (only if not already tagged)
    for sid, reason in seg_reason.items():
        if reason:
            mask = (df["segment_id"] == sid) & (df["filter_reason"] == "")
            df.loc[mask, "filter_reason"] = reason

    # Attach decay fit parameters
    for param in ["seg_lambda", "seg_a", "seg_b", "seg_r2"]:
        df[param] = df["segment_id"].map(
            lambda s, p=param: seg_decay.get(s, {}).get(p, np.nan)
        )

    return df, seg_quality, seg_reason


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Assign train/val/test splits
# ─────────────────────────────────────────────────────────────────────────────

def _assign_splits(
    df: pd.DataFrame,
    seg_quality: dict[int, bool],
    basin_number: int,
) -> pd.DataFrame:
    """
    Assign splits for all evaluation conditions.

    Existing column (V1 compatible):
      split — random 70/15/15 by segment (conditions A, B, C)

    New column (V2):
      split_held_out — for condition D:
        held_out basins  → all good events tagged 'held_out_test'
        other basins     → same random split as 'split' column
        outlier basins   → 'excluded'
    """
    df = df.copy()
    df["split"] = "excluded"
    df["split_held_out"] = "excluded"

    good_segments = sorted(s for s, q in seg_quality.items() if q)
    n_segs = len(good_segments)

    if n_segs == 0:
        return df

    rng      = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(good_segments)
    n_train  = max(1, round(n_segs * TRAIN_FRAC))
    n_val    = max(1, round(n_segs * VAL_FRAC))

    train_segs = set(shuffled[:n_train].tolist())
    val_segs   = set(shuffled[n_train:n_train + n_val].tolist())

    def _get_split(row) -> str:
        if not row["is_good_segment"]:
            return "excluded"
        if row["row_type"] == "reset":
            return "reset"
        sid = row["segment_id"]
        if sid in train_segs:
            return "train"
        if sid in val_segs:
            return "val"
        return "test"

    df["split"] = df.apply(_get_split, axis=1)

    # split_held_out — condition D
    # held_out basins: all good events are test only
    # outlier basins: excluded
    # clean basins: same as split
    if basin_number in HELD_OUT_BASIN_LIST:
        def _get_held_out_split(row) -> str:
            if not row["is_good_segment"]:
                return "excluded"
            if row["row_type"] == "reset":
                return "reset"
            return "held_out_test"
        df["split_held_out"] = df.apply(_get_held_out_split, axis=1)

    # elif basin_number in OUTLIER_BASINS:
    #     df["split_held_out"] = "excluded"

    else:
        # Clean basin — same random split as condition A
        df["split_held_out"] = df["split"]

    return df



# ─────────────────────────────────────────────────────────────────────────────
# Ceiling reset classification (replaces production tillage_driven_reset_search)
# ─────────────────────────────────────────────────────────────────────────────

def _ceiling_reset_search(
    df_passed:  pd.DataFrame,
    basin_number: int,
) -> pd.DataFrame:
    """
    Identify segment boundaries using ALL tillage events as credible resets,
    with smart date correction applied.

    Replaces production tillage_driven_reset_search() which rejected ~30-50%
    of tillage events based on IRD credibility criteria. Feedback loop analysis
    showed that accepting all tillage events (ceiling approach) gives better
    overall model performance — the Pearson r decay filter downstream provides
    the real quality gate.

    Smart date correction (applied before segmentation):
      For each tillage event τ:
        IRD_1 = IRD of last quality event BEFORE τ
        IRD_2 = IRD of first quality event AFTER τ
        Anchor reset to event with max(IRD_1, IRD_2)
        Effective timestamp = that event's opening_valve_date

    Falls back to LCT=0 based segmentation (_identify_segments) if:
      - Basin has no tillage data (no ibud sensor)
      - Tillage data is empty

    Returns df_passed with row_type and segment_id columns added.
    """
    # Try to load tillage data for this basin
    path = SatixPaths.BASIN_DB_DIR / f"basin_{int(basin_number)}.duckdb"
    if not path.exists():
        return _identify_segments(df_passed)

    with DuckDBManager(str(path)).connect_context() as db:
        tillage_df = read_clean_tillage_events(db)

    if tillage_df.empty:
        # No tillage data — fall back to LCT=0 based segmentation
        return _identify_segments(df_passed)

    df = df_passed.copy().sort_values("opening_valve_date").reset_index(drop=True)

    # Add scoring features needed for date correction
    df_scored = _add_scoring_features(df)

    # Normalize timestamps
    df_scored["opening_valve_date"] = pd.to_datetime(
        df_scored["opening_valve_date"]
    ).dt.tz_localize(None).astype("datetime64[ns]")
    tillage_df["timestamp"] = pd.to_datetime(
        tillage_df["timestamp"]
    ).dt.tz_localize(None).astype("datetime64[ns]")

    times_ns = df_scored["opening_valve_date"].values.astype("int64")
    ird_vals = pd.to_numeric(df_scored["IRD"], errors="coerce").to_numpy(float)
    quality  = (
        df_scored["event_quality_ok"].fillna(False).to_numpy(dtype=bool)
        if "event_quality_ok" in df_scored.columns
        else np.ones(len(df_scored), dtype=bool)
    )

    _wait_ns = int(pd.Timedelta(
        days=_TillageConstants.TILLAGE_MAX_WAIT_DAYS
    ).total_seconds() * 1e9)
    _back_ns = int(pd.Timedelta(
        days=_TillageConstants.TILLAGE_BACKWARD_WINDOW_DAYS
    ).total_seconds() * 1e9)

    till_ns = tillage_df["timestamp"].values.astype("int64")

    # ── Smart date correction: find corrected reset timestamps ────────────────
    reset_event_indices: list[int] = []

    for tau_ns in till_ns:
        before_pos = np.where(
            (times_ns >= tau_ns - _back_ns) &
            (times_ns < tau_ns) & quality
        )[0]
        after_pos = np.where(
            (times_ns >= tau_ns) &
            (times_ns <= tau_ns + _wait_ns) & quality
        )[0]

        ird_1 = float(ird_vals[before_pos[-1]])             if len(before_pos) and np.isfinite(ird_vals[before_pos[-1]])             else np.nan
        ird_2 = float(ird_vals[after_pos[0]])             if len(after_pos) and np.isfinite(ird_vals[after_pos[0]])             else np.nan

        # Choose event with max IRD as reset anchor
        if np.isfinite(ird_1) and np.isfinite(ird_2):
            chosen = int(before_pos[-1]) if ird_1 >= ird_2 else int(after_pos[0])
        elif np.isfinite(ird_1):
            chosen = int(before_pos[-1])
        elif np.isfinite(ird_2):
            chosen = int(after_pos[0])
        else:
            continue   # DATA_GAP — skip this tillage event

        reset_event_indices.append(chosen)

    # Deduplicate (two tillage events may map to same flooding event)
    reset_event_indices = sorted(set(reset_event_indices))

    if not reset_event_indices:
        # No valid resets found — fall back to LCT=0 segmentation
        return _identify_segments(df_passed)

    # ── Build segment structure from reset indices ─────────────────────────────
    reset_set   = set(reset_event_indices)
    row_type    = np.full(len(df), "pre_segment", dtype=object)
    seg_id_arr  = np.full(len(df), -1, dtype=int)
    seg_counter = 0
    in_segment  = False

    for i in range(len(df)):
        if i in reset_set:
            seg_counter += 1
            in_segment   = True
            row_type[i]  = "reset"
        elif in_segment:
            row_type[i] = "event"
        if in_segment:
            seg_id_arr[i] = seg_counter

    df["row_type"]   = row_type
    df["segment_id"] = seg_id_arr
    df.loc[df["row_type"] == "pre_segment", "filter_reason"] = "pre_segment"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Per-basin builder — orchestrates all steps
# ─────────────────────────────────────────────────────────────────────────────

def build_basin_dataset(basin_number: int) -> Optional[pd.DataFrame]:
    """
    Build the complete feature dataset for one basin.
    All events are kept; filter_reason documents why each row was excluded.

    Segmentation (Step 3) uses the ceiling reset classification:
    all tillage events are treated as credible resets with smart date
    correction. The Pearson r filter in Step 8 is the primary quality gate.

    Returns DataFrame or None if the basin DuckDB file does not exist.
    """
    # Step 1: read
    df = _read_raw_events(basin_number)
    if df is None:
        return None

    area_m2 = _get_basin_area(basin_number)
    n_raw   = len(df)

    # Step 1b: fill drying-window features with 0 where DrT == 0
    # Must happen before quality filter and feature computation so all
    # downstream steps see physically correct zeros instead of NaN.
    df = _fill_zero_drying(df)

    # Step 2: quality filter tags (all rows kept)
    df = _tag_quality_filter(df)
    df["is_good_segment"] = False

    passed     = df["filter_reason"] == ""
    df_passed  = df[passed].copy().reset_index(drop=True)
    n_passed   = len(df_passed)


    if n_passed < 10:

        # Not enough events to form segments — mark all and return
        df["row_type"]   = "quality_filtered"
        df["segment_id"] = -1
        df["split"]      = "excluded"
        _attach_metadata(df, basin_number, area_m2)
        return df

    # Step 3: segments — ceiling reset classification
    # All tillage events treated as credible resets with smart date correction.
    # Falls back to LCT=0 segmentation if no tillage data available.
    # Rationale: feedback loop analysis showed ceiling approach gives better
    # overall model performance. The Pearson r decay filter (Step 8) provides
    # the real quality gate — rejecting segments with no decay signal.
    df_passed = _ceiling_reset_search(df_passed, basin_number)

    # Steps 4 & 5: targets and HL
    df_passed = _compute_ird_norm(df_passed)
    df_passed = _compute_hl(df_passed, area_m2)

    # Steps 6 & 7: prev and cum features
    df_passed = _compute_prev_features(df_passed)
    df_passed = _compute_cum_features(df_passed)

    # Step 7b: cross-segment features (BEFORE quality filter)
    df_passed = _add_cross_segment_features(df_passed)

    # Step 8: segment quality tags
    df_passed, seg_quality, seg_reason = _tag_segment_quality(df_passed)

    # Step 9: splits
    df_passed = _assign_splits(df_passed, seg_quality, basin_number)

    # Merge quality-filtered rows back in with placeholder columns
    filtered_rows = df[~passed].copy()
    _fill_missing_columns(filtered_rows, df_passed.columns)

    df_full = pd.concat([df_passed, filtered_rows], ignore_index=True)
    df_full = df_full.sort_values("opening_valve_date").reset_index(drop=True)

    _attach_metadata(df_full, basin_number, area_m2)

    # Per-basin summary line
    n_good_segs   = sum(seg_quality.values())
    n_bad_segs    = len(seg_quality) - n_good_segs
    n_good_events = (
        (df_full["row_type"] == "event") & df_full["is_good_segment"]
    ).sum()
    reason_counts = Counter(
        seg_reason[s] for s in seg_reason if not seg_quality[s]
    )
    print(
        f"  [{basin_number:>5}]  "
        f"raw={n_raw:>5}  qpass={n_passed:>5}  "
        f"good_seg={n_good_segs:>3}  bad_seg={n_bad_segs:>3}  "
        f"good_ev={n_good_events:>5}"
        + (f"  excl: {dict(reason_counts)}" if reason_counts else "")
    )
    return df_full


def _fill_missing_columns(df: pd.DataFrame, target_columns) -> None:
    """Add any columns present in target_columns but missing from df, filled with NaN."""
    for col in target_columns:
        if col not in df.columns:
            df[col] = np.nan


def _attach_metadata(df: pd.DataFrame, basin_number: int, area_m2: float) -> None:
    """
    Attach basin-level metadata columns.

    basin_role at build time:
      'held_out' — basin is in HELD_OUT_BASIN_LIST (reserved for condition D/E test)
      'clean'    — all other basins

    NOTE: 'outlier' role is NOT assigned here. Outlier detection is performed
    dynamically by analysis/basin_analysis.py which writes to outlier_basins.csv.
    The model training scripts read that CSV at runtime to resolve basin sets.
    """
    df["basin_number"] = basin_number
    df["facility"]     = int(str(basin_number)[0])
    df["field_name"]   = FIELD_NAMES.get(int(str(basin_number)[0]), str(basin_number))
    df["area_m2"]      = area_m2

    if basin_number in HELD_OUT_BASIN_LIST:
        df["basin_role"] = "held_out"
    else:
        df["basin_role"] = "clean"


# ─────────────────────────────────────────────────────────────────────────────
# All basins
# ─────────────────────────────────────────────────────────────────────────────

def build_all_basins() -> dict[int, pd.DataFrame]:
    """Build datasets for all basins found in the global DuckDB."""
    with DuckDBManager(
        str(SatixPaths.GLOBAL_DB_PATH)
    ).connect_context() as db:
        basins_df = db.read_dataframe(
            "SELECT basin_number FROM basins "
            "WHERE basin_number IS NOT NULL ORDER BY basin_number"
        )
    basin_numbers = [int(bn) for bn in basins_df["basin_number"].tolist()]
    print(f"  Found {len(basin_numbers)} basins in global DB")

    result: dict[int, pd.DataFrame] = {}
    for bn in basin_numbers:
        df = build_basin_dataset(bn)
        if df is not None:
            result[bn] = df

    return result


def pool_basins(all_series: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate all basin DataFrames into one pooled DataFrame."""
    if not all_series:
        return pd.DataFrame()
    return pd.concat(list(all_series.values()), ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Filter funnel table (paper supplementary material)
# ─────────────────────────────────────────────────────────────────────────────

def print_filter_funnel(pooled: pd.DataFrame) -> pd.DataFrame:
    """
    Print and return the filter funnel table.
    Shows how many events were removed at each step and why.
    This table goes into the paper's supplementary material.

    Returns a DataFrame with columns: step, filter_reason, n_events, pct_of_raw
    """
    reason_counts = pooled["filter_reason"].value_counts()

    # after_cutoff events are excluded from reported counts entirely.
    # The dataset is presented as the study-period dataset to the reader.
    after_cutoff_n = int(reason_counts.get("after_cutoff", 0))
    total_raw      = len(pooled) - after_cutoff_n

    step_order = [
        ("Raw events (study period)", None),
        ("Drainage R² < 0.94 (sensor quality)", "quality_filter_IRD_R_squared"),
        ("After CIV filter", "quality_filter_CIV"),
        ("After Ct filter", "quality_filter_Ct"),
        ("After AL filter", "quality_filter_AL"),
        ("Pre-segment (before 1st reset)", "pre_segment"),
        ("Too few events (<4)", "too_few_events"),
        ("No decay signal (Pearson r > -0.05)", "pearson_r_positive"),
        ("Fit failed", "fit_failed"),
        ("R² below threshold (<0.10)", "r2_below_threshold"),
        ("Used for training", ""),
    ]

    rows = []
    for label, reason in step_order:
        if reason is None:
            n = total_raw
        else:
            n = int(reason_counts.get(reason, 0))
        rows.append(dict(
            step          = label,
            filter_reason = reason if reason is not None else "—",
            n_events      = n,
            pct_of_raw    = round(100.0 * n / total_raw, 1) if total_raw > 0 else 0.0,
        ))

    funnel_df = pd.DataFrame(rows)

    print("\n" + "=" * 65)
    print("  FILTER FUNNEL SUMMARY")
    print("=" * 65)
    print(f"  {'Step':<42} {'N':>7}  {'% of raw':>9}")
    print(f"  {'-'*60}")
    for _, row in funnel_df.iterrows():
        marker = "  ✓" if row["filter_reason"] == "" else "  ✗"
        print(
            f"  {row['step']:<42} "
            f"{int(row['n_events']):>7}  "
            f"{row['pct_of_raw']:>8.1f}%"
            f"{marker if row['filter_reason'] in ('', '—') else ''}"
        )
    print("=" * 65)

    # Zero-DrT fill note (for paper footnote)
    if "RD" in pooled.columns and "DrT" in pooled.columns:
        good = pooled[
            (pooled.get("is_good_segment", False) == True) &
            (pooled.get("row_type", "") == "event")
        ] if "is_good_segment" in pooled.columns else pooled
        drt_zero = (
            pd.to_numeric(good.get("DrT", pd.Series()), errors="coerce").fillna(0)
            < _NEAR_ZERO_DRT
        )
        print(f"\n  Note (paper footnote):")
        print(f"    Events with DrT<0.1h in good training set : "
              f"{int(drt_zero.sum())} / {len(good)} "
              f"({100*drt_zero.mean():.1f}%)")
        print(f"    Drying-window features (RD,TD,HD,WDD,WSD,PD) "
              f"filled with 0 for these events.")

    # Save as XLSX for paper
    out_path = TABLES_DIR / "filter_funnel.xlsx"
    funnel_df.to_excel(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    return funnel_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Build event_dataset.csv from DuckDB and print the filter funnel.

    Parameters
    ----------
    force_rebuild : if True, rebuilds from DuckDB even if CSV already exists.

    Returns
    -------
    pooled : the full pooled DataFrame (all basins, all events)
    """
    print("=" * 65)
    print("  BUILD DATASET — pipeline/build_dataset.py")
    print("=" * 65)
    print(f"  QUALITY_FILTER         : {QUALITY_FILTER}")
    print(f"  MIN_EVENTS_PER_SEGMENT : {MIN_EVENTS_PER_SEGMENT}")
    print(f"  MIN_SEGMENT_R2         : {MIN_SEGMENT_R2}")
    print(f"  PEARSON_THRESHOLD      : {PEARSON_OUTLIER_THRESHOLD}")
    print()

    if EVENT_CSV.exists() and not force_rebuild:
        print(f"  Cached CSV found: {EVENT_CSV}")
        print("  Loading from cache (use --rebuild to force rebuild from DuckDB)")
        pooled = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
        print(f"  Loaded {len(pooled)} rows, {pooled['basin_number'].nunique()} basins")
    else:
        print("  Building from DuckDB...")
        all_series = build_all_basins()
        pooled     = pool_basins(all_series)

        # Add IRD_norm_log column (alias used by Model 1)
        if "IRD_norm" in pooled.columns:
            pooled["IRD_norm_log"] = pooled["IRD_norm"]

        # Remove duplicate columns (known issue from segment merging)
        pooled = pooled.loc[:, ~pooled.columns.duplicated()]

        pooled.to_csv(EVENT_CSV, index=False)
        print(f"\n  Saved: {EVENT_CSV}  ({len(pooled)} rows)")

    print_filter_funnel(pooled)

    # Summary statistics for paper
    good_events  = (
        (pooled.get("row_type", "") == "event") &
        pooled.get("is_good_segment", False)
    ).sum()
    total_events = (pooled.get("row_type", "") == "event").sum()
    print(f"\n  Good events (used for training) : {good_events:>7}")
    print(f"  Total event rows (quality passed): {total_events:>7}")
    print(f"  Total rows in CSV (all events)   : {len(pooled):>7}")
    print(f"  Basins                           : {pooled['basin_number'].nunique():>7}")

    return pooled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build event_dataset.csv from DuckDB"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild from DuckDB even if event_dataset.csv already exists"
    )
    args = parser.parse_args()
    main(force_rebuild=args.rebuild)
