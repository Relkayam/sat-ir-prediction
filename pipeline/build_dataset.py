"""
pipeline/build_dataset.py — Build event_dataset.csv from DuckDB
================================================================
Reads all flooding events for all basins, applies the four SENSOR-quality
filters, identifies segments, computes features, and saves a single CSV.

Every raw event is KEPT in the CSV with a filter_reason column.

PIPELINE ORDER (fixed)
----------------------
  1. Read raw events
  2. Fill zero drying
  3. Segmentation on ALL raw events (ceiling reset search)
     → assigns segment_id, row_type, IRD_at_reset to every event
  4. Compute IRD_norm on ALL events (needs IRD_at_reset from step 3)
  5. Compute HL on ALL events
  6. Tag quality filter (marks filter_reason — does NOT drop rows)
  7. Split into passed / filtered subsets
  8. Compute prev_* and cum_* features on passed events only
     (lag features must not cross quality-filtered events)
  9. Fit per-segment decay (DIAGNOSTIC ONLY — see note), assign splits
 10. Reassemble full df and attach metadata

Rationale for this order
-------------------------
  Segmentation must run before quality filtering because the reset
  anchor event (highest IRD after tillage) may itself fail a quality
  filter (e.g. low IRD_R_squared on the first flooding after tillage).
  Filtering first caused basins with poor-quality reset events to lose
  their entire segment structure — IRD_at_reset was NaN for all rows.

WHAT FILTERS THE MODEL DATA (and what does NOT)
-----------------------------------------------
  Condition E-full uses EVERY event that passes the four sensor-quality
  filters defined in QUALITY_FILTER (IRD_R² ≥ 0.94, CIV ≥ 3000, Ct ≥ 20,
  AL ≥ 5) and is not a pre-segment row. There is NO segment-level quality
  gate of any kind: no is_good_segment flag, no Pearson screen, no decay-R²
  threshold. The per-segment exponential decay fit computed in step 9
  (seg_lambda / seg_a / seg_b / seg_r2) is a DIAGNOSTIC retained only for
  the Figure 2 overlays and the fitted-decay-rate range in the Discussion.
  It does not exclude, select, or weight any event.

basin_role values
-----------------
  'clean' — all basins. Held-out selection at bootstrap runtime.

Usage
-----
  python -m pipeline.build_dataset              # uses cached CSV if exists
  python -m pipeline.build_dataset --rebuild    # forces rebuild from DuckDB
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_CUTOFF, QUALITY_FILTER,
    MIN_EVENTS_PER_SEGMENT,
    TRAIN_FRAC, RANDOM_SEED,
    EVENT_CSV, TABLES_DIR, FIELD_NAMES,
)

from optisat.db.duckdb_manager import DuckDBManager
from optisat.db.paths import SatixPaths
from optisat.etl.features.tillage_features import (
    read_clean_tillage_events,
    _add_scoring_features,
)
from optisat.etl.features.constants import Constants as _TillageConstants

from pipeline.features import PREV_SOURCE_COLS, CUM_SOURCE_COLS, RAW_DB_COLUMNS

print("Imports OK")

_NEAR_ZERO_DRT      = 0.1
_DRYING_WINDOW_COLS = ["RD", "TD", "HD", "WDD", "WSD", "PD"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Read raw events from DuckDB
# ─────────────────────────────────────────────────────────────────────────────

def _get_basin_area(basin_number: int) -> float:
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
    path = SatixPaths.BASIN_DB_DIR / f"basin_{int(basin_number)}.duckdb"
    if not path.exists():
        return None
    with DuckDBManager(str(path)).connect_context() as db:
        schema    = db.read_dataframe("PRAGMA table_info('features')")
        available = set(schema["name"].tolist())
        cols      = [c for c in RAW_DB_COLUMNS if c in available]
        df        = db.read_dataframe(
            f"SELECT {', '.join(cols)} FROM features "
            f"WHERE opening_valve_date < '{DATA_CUTOFF}' "
            f"ORDER BY opening_valve_date"
        )
    if df.empty:
        return None
    df["opening_valve_date"] = pd.to_datetime(df["opening_valve_date"])
    return df.sort_values("opening_valve_date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Fill drying-window features with 0 when DrT is zero
# ─────────────────────────────────────────────────────────────────────────────

def _fill_zero_drying(df: pd.DataFrame) -> pd.DataFrame:
    df       = df.copy()
    drt      = pd.to_numeric(df.get("DrT", pd.Series(dtype=float)), errors="coerce")
    zero_drt = drt.fillna(0) < _NEAR_ZERO_DRT
    for col in _DRYING_WINDOW_COLS:
        if col in df.columns:
            df.loc[zero_drt & df[col].isna(), col] = 0.0
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Segmentation on ALL raw events
# ─────────────────────────────────────────────────────────────────────────────

def _identify_segments(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback segmentation using LCT == 0 as reset indicator.
    Used when no DuckDB file exists or tillage table is empty.
    Operates on full raw dataframe (all events, no quality filter applied).
    """
    df         = df_raw.copy()
    lct        = pd.to_numeric(df["LCT"], errors="coerce")
    reset_mask = lct.fillna(-1) == 0.0
    row_type   = np.full(len(df), "pre_segment", dtype=object)
    seg_id_arr = np.full(len(df), -1, dtype=int)
    seg_counter = 0
    in_segment  = False
    for i in range(len(df)):
        if reset_mask.iloc[i]:
            seg_counter += 1; in_segment = True; row_type[i] = "reset"
        elif in_segment:
            row_type[i] = "event"
        if in_segment:
            seg_id_arr[i] = seg_counter
    df["row_type"]   = row_type
    df["segment_id"] = seg_id_arr
    # pre_segment rows will get filter_reason assigned in step 6
    return df


def _ceiling_reset_search(df_raw: pd.DataFrame, basin_number: int) -> pd.DataFrame:
    """
    Segment the full raw event stream using tillage timestamps.

    Runs on ALL raw events (before quality filtering) so that:
    - Reset anchor events are found even if they fail quality filters
    - IRD_at_reset is populated for every event in the segment
    - Quality filtering in step 6 can then mark individual events
      without destroying the segment structure

    The internal `quality` mask (event_quality_ok) still guides which
    events are preferred as reset anchors — it just no longer gates
    which events are available for the search.
    """
    path = SatixPaths.BASIN_DB_DIR / f"basin_{int(basin_number)}.duckdb"
    if not path.exists():
        return _identify_segments(df_raw)

    with DuckDBManager(str(path)).connect_context() as db:
        tillage_df = read_clean_tillage_events(db)
    if tillage_df.empty:
        return _identify_segments(df_raw)

    df        = df_raw.copy().sort_values("opening_valve_date").reset_index(drop=True)
    df_scored = _add_scoring_features(df)
    df_scored["opening_valve_date"] = (
        pd.to_datetime(df_scored["opening_valve_date"])
        .dt.tz_localize(None).astype("datetime64[ns]"))
    tillage_df["timestamp"] = (
        pd.to_datetime(tillage_df["timestamp"])
        .dt.tz_localize(None).astype("datetime64[ns]"))

    times_ns = df_scored["opening_valve_date"].values.astype("int64")
    ird_vals = pd.to_numeric(df_scored["IRD"], errors="coerce").to_numpy(float)

    # event_quality_ok guides anchor preference but all events are candidates
    quality  = (df_scored["event_quality_ok"].fillna(False).to_numpy(dtype=bool)
                if "event_quality_ok" in df_scored.columns
                else np.ones(len(df_scored), dtype=bool))

    _wait_ns = int(pd.Timedelta(days=_TillageConstants.TILLAGE_MAX_WAIT_DAYS
                                ).total_seconds() * 1e9)
    _back_ns = int(pd.Timedelta(days=_TillageConstants.TILLAGE_BACKWARD_WINDOW_DAYS
                                ).total_seconds() * 1e9)
    till_ns  = tillage_df["timestamp"].values.astype("int64")

    reset_event_indices: list[int] = []
    for tau_ns in till_ns:
        # Prefer quality-passing events as anchors; fall back to any event
        before_q = np.where(
            (times_ns >= tau_ns - _back_ns) & (times_ns < tau_ns) & quality)[0]
        after_q  = np.where(
            (times_ns >= tau_ns) & (times_ns <= tau_ns + _wait_ns) & quality)[0]

        # Fallback: any event in window (not just quality-passing)
        before_any = np.where(
            (times_ns >= tau_ns - _back_ns) & (times_ns < tau_ns))[0]
        after_any  = np.where(
            (times_ns >= tau_ns) & (times_ns <= tau_ns + _wait_ns))[0]

        before_pos = before_q if len(before_q) > 0 else before_any
        after_pos  = after_q  if len(after_q)  > 0 else after_any

        ird_1 = (float(ird_vals[before_pos[-1]])
                 if len(before_pos) and np.isfinite(ird_vals[before_pos[-1]])
                 else np.nan)
        ird_2 = (float(ird_vals[after_pos[0]])
                 if len(after_pos) and np.isfinite(ird_vals[after_pos[0]])
                 else np.nan)

        if np.isfinite(ird_1) and np.isfinite(ird_2):
            chosen = int(before_pos[-1]) if ird_1 >= ird_2 else int(after_pos[0])
        elif np.isfinite(ird_1):
            chosen = int(before_pos[-1])
        elif np.isfinite(ird_2):
            chosen = int(after_pos[0])
        else:
            continue
        reset_event_indices.append(chosen)

    reset_event_indices = sorted(set(reset_event_indices))
    if not reset_event_indices:
        return _identify_segments(df_raw)

    reset_set   = set(reset_event_indices)
    row_type    = np.full(len(df), "pre_segment", dtype=object)
    seg_id_arr  = np.full(len(df), -1, dtype=int)
    seg_counter = 0
    in_segment  = False
    for i in range(len(df)):
        if i in reset_set:
            seg_counter += 1; in_segment = True; row_type[i] = "reset"
        elif in_segment:
            row_type[i] = "event"
        if in_segment:
            seg_id_arr[i] = seg_counter
    df["row_type"]   = row_type
    df["segment_id"] = seg_id_arr
    return df



# ─────────────────────────────────────────────────────────────────────────────
# Step 3b — Recompute LCT where NaN (from segmentation results)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_lct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill LCT (hours since tillage reset) where NaN, from segmentation results.

    LCT = (opening_valve_date - reset_date).total_seconds() / 3600.
    At the reset event: LCT = 0.

    Only fills NaN values — database values are preserved for normal basins.
    """
    df = df.copy()

    for sid, grp in df.groupby("segment_id"):
        if sid < 0:
            continue

        grp = grp.sort_values("opening_valve_date")
        reset_rows = grp[grp["row_type"] == "reset"]
        if reset_rows.empty:
            continue

        reset_date = reset_rows["opening_valve_date"].iloc[0]
        hours = (
            grp["opening_valve_date"] - reset_date
        ).dt.total_seconds() / 3600.0

        # Only fill NaN LCT values — preserve database values where available
        lct_col   = pd.to_numeric(grp["LCT"], errors="coerce")
        needs_fill = lct_col.isna()
        if needs_fill.any():
            df.loc[grp.index[needs_fill], "LCT"] = hours.values[needs_fill.values]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3c — Fill IRD_at_reset where NaN (from segmentation results)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ird_at_reset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill IRD_at_reset from the reset event's IRD value where NaN.

    IRD_at_reset is pre-computed by optisat for most basins, but is NaN for
    basins where the reset anchor was quality-filtered in the old pipeline.
    Only fills NaN — database values are preserved.
    """
    df = df.copy()

    for sid, grp in df.groupby("segment_id"):
        if sid < 0:
            continue

        reset_rows = grp[grp["row_type"] == "reset"]
        if reset_rows.empty:
            continue

        seg_mask = df["segment_id"] == sid

        # Only fill if IRD_at_reset is NaN for this segment
        if df.loc[seg_mask, "IRD_at_reset"].notna().any():
            continue  # already populated — trust the database value

        reset_ird = pd.to_numeric(reset_rows["IRD"].iloc[0], errors="coerce")
        if not np.isfinite(reset_ird):
            continue

        df.loc[seg_mask, "IRD_at_reset"] = reset_ird

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Steps 4 & 5 — IRD_norm and HL (on all events, after segmentation)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ird_norm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute IRD_norm = ln(IRD / IRD_at_reset) for ALL events.
    IRD_at_reset is populated from the full-data segmentation step,
    so this works even for events that will later be quality-filtered.
    """
    df = df.copy()
    ird       = pd.to_numeric(df["IRD"],          errors="coerce")
    ird_reset = pd.to_numeric(df["IRD_at_reset"], errors="coerce").replace(0, np.nan)
    df["IRD_norm"] = np.log(ird / ird_reset)
    return df


def _compute_hl(df: pd.DataFrame, area_m2: float) -> pd.DataFrame:
    df = df.copy()
    df["HL"] = pd.to_numeric(df["CIV"], errors="coerce") / area_m2
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Tag quality filter (marks filter_reason — does NOT drop rows)
# ─────────────────────────────────────────────────────────────────────────────

def _tag_quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark filter_reason for events that fail the four SENSOR-quality thresholds.
    pre_segment rows already have filter_reason = 'pre_segment' from
    segmentation; this step adds quality flags to the remaining rows.
    Does NOT remove any rows — all events remain in the dataframe.

    NOTE: This is the ONLY step that excludes events from the model data.
    There is no segment-level quality gate anywhere in this pipeline.
    """
    df = df.copy()

    # Initialise filter_reason if not already set by segmentation
    if "filter_reason" not in df.columns:
        df["filter_reason"] = ""
    else:
        # Keep pre_segment labels; clear everything else for re-evaluation
        df["filter_reason"] = df["filter_reason"].where(
            df["filter_reason"] == "pre_segment", "")

    # after_cutoff — applied to all rows regardless of segment status
    after = df["opening_valve_date"] >= pd.Timestamp(DATA_CUTOFF)
    df.loc[after & (df["filter_reason"] == ""), "filter_reason"] = "after_cutoff"

    # sensor / operational quality filters
    for col, threshold in QUALITY_FILTER.items():
        if col not in df.columns:
            continue
        unflagged = df["filter_reason"] == ""
        fails     = pd.to_numeric(df[col], errors="coerce").fillna(0) < threshold
        df.loc[unflagged & fails, "filter_reason"] = f"quality_filter_{col}"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Steps 7 & 8 — prev_* and cum_* features (on quality-passing events only)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_prev_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lag features within each segment, on quality-passing events only.
    This prevents a quality-filtered event from corrupting the lag for the
    next good event in the same segment.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ ⚠ IMPORTANT — segment_id IS NOT GLOBALLY UNIQUE.                          │
    │                                                                           │
    │ segment_id is a PER-BASIN counter that restarts at 1 for every basin     │
    │ (see _ceiling_reset_search / _identify_segments). The groupby below is   │
    │ therefore correct ONLY because build_basin_dataset() calls this function │
    │ on ONE basin at a time, BEFORE pool_basins() concatenates all 50 basins. │
    │                                                                           │
    │ NEVER call _compute_prev_features (or _compute_cum_features) on the       │
    │ POOLED dataframe: segment 1 of basin 3203 and segment 1 of basin 4401    │
    │ would silently merge and the lag would jump across basins. If you ever   │
    │ need a global key, build one as f"{basin_number}_{segment_id}" first.    │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    df = df.copy()
    for col in PREV_SOURCE_COLS:
        if col not in df.columns:
            continue
        df[f"prev_{col}"] = np.nan
        for sid, grp in df.groupby("segment_id"):
            if sid < 0:
                continue
            df.loc[grp.index, f"prev_{col}"] = (
                pd.to_numeric(grp[col], errors="coerce").shift(1).values)
    return df


def _compute_cum_features(df: pd.DataFrame) -> pd.DataFrame:
    # ⚠ Same per-basin segment_id caveat as _compute_prev_features above:
    #   this MUST run per-basin (pre-pooling), never on the pooled frame.
    df = df.copy()
    for cum_col, src_col in CUM_SOURCE_COLS.items():
        df[cum_col] = np.nan
        if src_col not in df.columns:
            continue
        for sid, grp in df.groupby("segment_id"):
            if sid < 0:
                continue
            df.loc[grp.index, cum_col] = (
                pd.to_numeric(grp[src_col], errors="coerce").fillna(0).cumsum().values)
    df["event_count"] = np.nan
    for sid, grp in df.groupby("segment_id"):
        if sid < 0:
            continue
        df.loc[grp.index, "event_count"] = np.arange(1, len(grp) + 1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 8b — Cross-segment features
# ─────────────────────────────────────────────────────────────────────────────

def _add_cross_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prev_IRD_at_reset"]      = np.nan
    df["prev_prev_IRD_at_reset"] = np.nan
    df["prev_reset_date"]        = pd.NaT
    df["IRD_direction"]          = np.nan

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
            vals = dict(prev_IRD_at_reset=np.nan, prev_prev_IRD_at_reset=np.nan,
                        prev_reset_date=pd.NaT, IRD_direction=np.nan)
        elif idx == 1:
            prev_ird = reset_rows.loc[idx-1, "IRD_at_reset"]
            vals = dict(
                prev_IRD_at_reset      = float(prev_ird) if pd.notna(prev_ird) else np.nan,
                prev_prev_IRD_at_reset = np.nan,
                prev_reset_date        = reset_rows.loc[idx-1, "opening_valve_date"],
                IRD_direction          = np.nan,
            )
        else:
            prev_ird       = float(reset_rows.loc[idx-1, "IRD_at_reset"])
            prev_prev_ird  = float(reset_rows.loc[idx-2, "IRD_at_reset"])
            prev_date      = reset_rows.loc[idx-1, "opening_valve_date"]
            prev_prev_date = reset_rows.loc[idx-2, "opening_valve_date"]
            dt = ((prev_date - prev_prev_date).total_seconds() / 86400.0
                  if pd.notna(prev_date) and pd.notna(prev_prev_date) else np.nan)
            direction = (
                (prev_ird - prev_prev_ird) / dt
                if np.isfinite(prev_ird) and np.isfinite(prev_prev_ird)
                   and np.isfinite(dt) and dt > 0 else np.nan)
            vals = dict(prev_IRD_at_reset=prev_ird,
                        prev_prev_IRD_at_reset=prev_prev_ird,
                        prev_reset_date=prev_date,
                        IRD_direction=direction)
        mask = df["segment_id"] == sid
        for col, val in vals.items():
            df.loc[mask, col] = val
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Per-segment decay fit  (DIAGNOSTIC ONLY — does NOT gate any event)
# ─────────────────────────────────────────────────────────────────────────────

def _decay_model(lct, a, b, lam):
    return a * np.exp(-lam * lct) + b


def _fit_decay(lct: np.ndarray, inorm: np.ndarray) -> Optional[dict]:
    if len(lct) < MIN_EVENTS_PER_SEGMENT:
        return None
    a_i   = float(np.percentile(inorm, 95) - np.percentile(inorm, 5))
    b_i   = float(np.percentile(inorm, 5))
    lam_i = 1.0 / (float(np.median(lct)) + 1e-6)
    try:
        popt, _ = curve_fit(_decay_model, lct, inorm,
                            p0=[a_i, b_i, lam_i],
                            bounds=([-10,-10,1e-7],[10,10,0.5]),
                            maxfev=10_000, method="trf")
        pred   = _decay_model(lct, *popt)
        ss_res = float(np.sum((inorm - pred)**2))
        ss_tot = float(np.sum((inorm - inorm.mean())**2))
        r2     = 1.0 - ss_res/ss_tot if ss_tot > 1e-12 else np.nan
        return dict(seg_lambda=float(popt[2]), seg_a=float(popt[0]),
                    seg_b=float(popt[1]),     seg_r2=float(r2))
    except Exception:
        return None


def _fit_segment_decay(
    df:          pd.DataFrame,
    passed_mask: pd.Series,
) -> pd.DataFrame:
    """
    Fit a per-segment exponential decay to normalized IRD (η vs LCT) on
    quality-passing event rows, and broadcast the fit parameters
    (seg_lambda, seg_a, seg_b, seg_r2) to ALL rows of each segment.

    ───────────────────────────────────────────────────────────────────────────
    DIAGNOSTIC ONLY. These parameters are NOT used to filter, gate, select, or
    weight any event for model training or evaluation. There is no
    'is_good_segment' flag and no segment is ever excluded on the basis of fit
    quality. The fit exists solely to support:
      - the dashed exponential overlays in Figure 2, and
      - the fitted-decay-rate (seg_lambda) range reported in the Discussion.
    If Figure 2 ever refits independently and does not read seg_lambda from the
    CSV, this entire step (and _fit_decay / _decay_model / curve_fit) can be
    deleted without affecting any model result.
    ───────────────────────────────────────────────────────────────────────────
    """
    df        = df.copy()
    df_passed = df[passed_mask]
    seg_decay: dict[int, dict] = {}

    for sid in [s for s in df["segment_id"].unique() if s >= 0]:
        grp        = df_passed[df_passed["segment_id"] == sid]
        event_rows = grp[grp["row_type"] == "event"]
        lct        = event_rows["LCT"].values.astype(float)
        inorm      = event_rows["IRD_norm"].values
        valid      = np.isfinite(lct) & np.isfinite(inorm) & (lct > 0)
        if valid.sum() < MIN_EVENTS_PER_SEGMENT:
            continue
        fit = _fit_decay(lct[valid], inorm[valid])
        if fit is not None:
            seg_decay[sid] = fit

    for param in ["seg_lambda", "seg_a", "seg_b", "seg_r2"]:
        df[param] = df["segment_id"].map(
            lambda s, p=param: seg_decay.get(s, {}).get(p, np.nan))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 10 — Assign splits
# ─────────────────────────────────────────────────────────────────────────────

def _assign_splits(
    df:          pd.DataFrame,
    passed_mask: pd.Series,
) -> pd.DataFrame:
    """
    Random 80/20 train/val split BY SEGMENT, applied to ALL segments that
    contain at least one quality-passing event.

    Splitting at the segment level (not the event level) prevents leakage
    between consecutive events of the same segment. Condition E-full uses every
    event that passes the four sensor-quality filters — there is no
    is_good_segment gate.

      reset rows           → split == 'reset'
      non-quality-passing  → split == 'excluded'
      passing event rows   → split in {'train', 'val'} by segment
    """
    df = df.copy()
    df["split"] = "excluded"

    # All segments with at least one quality-passing event
    all_segments = sorted([s for s in df["segment_id"].unique() if s >= 0])
    if not all_segments:
        return df

    rng      = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(all_segments)
    n_train  = max(1, round(len(shuffled) * TRAIN_FRAC))
    train_segs = set(shuffled[:n_train].tolist())
    val_segs   = set(shuffled[n_train:].tolist())

    def _get_split(row) -> str:
        if not passed_mask.loc[row.name]:
            return "excluded"
        if row["row_type"] == "reset":
            return "reset"
        sid = row["segment_id"]
        if sid < 0:
            return "excluded"
        if sid in train_segs: return "train"
        if sid in val_segs:   return "val"
        return "excluded"

    df["split"] = df.apply(_get_split, axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

def _attach_metadata(df: pd.DataFrame, basin_number: int, area_m2: float) -> None:
    df["basin_number"] = basin_number
    df["facility"]     = int(str(basin_number)[0])
    df["field_name"]   = FIELD_NAMES.get(int(str(basin_number)[0]), str(basin_number))
    df["area_m2"]      = area_m2
    df["basin_role"]   = "clean"


# ─────────────────────────────────────────────────────────────────────────────
# Per-basin builder
# ─────────────────────────────────────────────────────────────────────────────

def build_basin_dataset(basin_number: int) -> Optional[pd.DataFrame]:
    df = _read_raw_events(basin_number)
    if df is None:
        return None

    area_m2 = _get_basin_area(basin_number)
    n_raw   = len(df)

    # ── Step 2: fill zero-drying features ────────────────────────────────────
    df = _fill_zero_drying(df)

    # ── Step 3: segmentation on ALL raw events ────────────────────────────────
    # Must run before quality filtering so reset anchors are never filtered out
    df = _ceiling_reset_search(df, basin_number)

    # ── Step 3b: recompute LCT from segmentation results ─────────────────────
    df = _compute_lct(df)

    # ── Step 3c: fill IRD_at_reset where NaN ─────────────────────────────────
    df = _compute_ird_at_reset(df)

    # ── Steps 4 & 5: IRD_norm and HL on ALL events ───────────────────────────
    df = _compute_ird_norm(df)
    df = _compute_hl(df, area_m2)

    # ── Step 6: quality filter (marks filter_reason, keeps all rows) ─────────
    df = _tag_quality_filter(df)

    # passed_mask: events that pass all four sensor-quality filters
    # (pre_segment rows are also excluded via their filter_reason)
    passed_mask = df["filter_reason"] == ""
    n_passed    = int(passed_mask.sum())

    if n_passed < 10:
        # Too few quality-passing events to be useful — mark whole basin
        df["split"] = "excluded"
        _attach_metadata(df, basin_number, area_m2)
        print(f"  [{basin_number:>5}]  raw={n_raw:>5}  qpass={n_passed:>5}  "
              f"SKIPPED (insufficient quality-passing events)")
        return df

    # ── Steps 7 & 8: lag features on quality-passing events only ─────────────
    # See the per-basin segment_id warning inside _compute_prev_features:
    # this MUST run on a single-basin frame, which is the case here.
    df_passed = df[passed_mask].copy().reset_index(drop=True)
    df_passed = _compute_prev_features(df_passed)
    df_passed = _compute_cum_features(df_passed)
    df_passed = _add_cross_segment_features(df_passed)

    # Merge computed features back into full df using pd.concat.
    # Strategy: keep all non-passed rows from df, replace passed rows with df_passed.
    df_not_passed = df[~passed_mask].copy()

    for col in df_passed.columns:
        if col not in df_not_passed.columns:
            if pd.api.types.is_datetime64_any_dtype(df_passed[col]):
                df_not_passed[col] = pd.NaT
            else:
                df_not_passed[col] = np.nan

    for col in df_not_passed.columns:
        if col not in df_passed.columns:
            if pd.api.types.is_datetime64_any_dtype(df_not_passed[col]):
                df_passed[col] = pd.NaT
            else:
                df_passed[col] = np.nan

    df = pd.concat([df_passed, df_not_passed], ignore_index=True)
    df = df.sort_values("opening_valve_date").reset_index(drop=True)

    # Rebuild passed_mask after concat + sort (index was reset)
    passed_mask = df["filter_reason"] == ""

    # ── Step 9: per-segment decay fit (DIAGNOSTIC ONLY — gates nothing) ──────
    df = _fit_segment_decay(df, passed_mask)

    # ── Step 10: assign splits ───────────────────────────────────────────────
    df = _assign_splits(df, passed_mask)

    _attach_metadata(df, basin_number, area_m2)

    n_segments     = int(len([s for s in df["segment_id"].unique() if s >= 0]))
    n_passed_event = int(((df["row_type"] == "event") & passed_mask).sum())
    print(
        f"  [{basin_number:>5}]  raw={n_raw:>5}  qpass={n_passed:>5}  "
        f"segments={n_segments:>3}  passed_events={n_passed_event:>5}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# All basins
# ─────────────────────────────────────────────────────────────────────────────

def build_all_basins() -> dict[int, pd.DataFrame]:
    with DuckDBManager(str(SatixPaths.GLOBAL_DB_PATH)).connect_context() as db:
        basins_df = db.read_dataframe(
            "SELECT basin_number FROM basins "
            "WHERE basin_number IS NOT NULL ORDER BY basin_number")
    basin_numbers = [int(bn) for bn in basins_df["basin_number"].tolist()]
    print(f"  Found {len(basin_numbers)} basins in global DB")
    result: dict[int, pd.DataFrame] = {}
    for bn in basin_numbers:
        df = build_basin_dataset(bn)
        if df is not None:
            result[bn] = df
    return result


def pool_basins(all_series: dict[int, pd.DataFrame]) -> pd.DataFrame:
    # NOTE: prev_*/cum_* features are already computed per-basin inside
    # build_basin_dataset BEFORE this concat — see the segment_id warning in
    # _compute_prev_features. Do not (re)compute any segment lag here.
    if not all_series:
        return pd.DataFrame()
    return pd.concat(list(all_series.values()), ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Filter funnel
# ─────────────────────────────────────────────────────────────────────────────

def print_filter_funnel(pooled: pd.DataFrame) -> pd.DataFrame:
    reason_counts  = pooled["filter_reason"].value_counts()
    after_cutoff_n = int(reason_counts.get("after_cutoff", 0))
    total_raw      = len(pooled) - after_cutoff_n

    # Only the four SENSOR filters + pre_segment exclude events now.
    # There is no segment-level quality gate, so no decay-screening rows here.
    step_order = [
        ("Raw events (study period)",                None),
        ("Drainage R² < 0.94 (sensor quality)",     "quality_filter_IRD_R_squared"),
        ("CIV < 3,000 m³",                          "quality_filter_CIV"),
        ("Ct < 20 h",                               "quality_filter_Ct"),
        ("AL < 5 cm",                               "quality_filter_AL"),
        ("Pre-segment (before 1st reset)",          "pre_segment"),
        ("Passed events (used by the models)",      ""),
    ]

    rows = []
    for label, reason in step_order:
        n = total_raw if reason is None else int(reason_counts.get(reason, 0))
        rows.append(dict(
            step          = label,
            filter_reason = "—" if reason is None else reason,
            n_events      = n,
            pct_of_raw    = round(100.0 * n / total_raw, 1) if total_raw > 0 else 0.0,
        ))

    funnel_df = pd.DataFrame(rows)
    print("\n" + "="*65)
    print("  FILTER FUNNEL SUMMARY")
    print("="*65)
    print(f"  {'Step':<45} {'N':>7}  {'% of raw':>9}")
    print(f"  {'-'*63}")
    for _, row in funnel_df.iterrows():
        print(f"  {row['step']:<45} {int(row['n_events']):>7}  "
              f"{row['pct_of_raw']:>8.1f}%")
    print("="*65)

    out_path = TABLES_DIR / "filter_funnel.xlsx"
    funnel_df.to_excel(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    return funnel_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(force_rebuild: bool = False) -> pd.DataFrame:
    print("="*65)
    print("  BUILD DATASET — pipeline/build_dataset.py")
    print("="*65)
    print(f"  QUALITY_FILTER         : {QUALITY_FILTER}")
    print(f"  MIN_EVENTS_PER_SEGMENT : {MIN_EVENTS_PER_SEGMENT} (decay-fit guard only)")
    print(f"  DATA_CUTOFF            : {DATA_CUTOFF} (enforced at SQL read level)")
    print(f"  Split                  : 80% train / 20% val by segment (no internal test)")
    print(f"  Pipeline order         : segment → IRD_norm → quality_filter → features")
    print(f"  Segment quality gate   : NONE — all sensor-passing events are used")
    print(f"  NOTE: No fixed held-out basins — selection at bootstrap runtime")
    print()

    if EVENT_CSV.exists() and not force_rebuild:
        print(f"  Cached CSV found: {EVENT_CSV}")
        print("  Loading from cache (use --rebuild to force rebuild)")
        pooled = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
        print(f"  Loaded {len(pooled)} rows, {pooled['basin_number'].nunique()} basins")
    else:
        print("  Building from DuckDB...")
        all_series = build_all_basins()
        pooled     = pool_basins(all_series)
        if "IRD_norm" in pooled.columns:
            pooled["IRD_norm_log"] = pooled["IRD_norm"]
        pooled = pooled.loc[:, ~pooled.columns.duplicated()]
        pooled.to_csv(EVENT_CSV, index=False)
        print(f"\n  Saved: {EVENT_CSV}  ({len(pooled)} rows)")

    print_filter_funnel(pooled)

    passed_events = int(((pooled.get("row_type", "") == "event") &
                         (pooled.get("filter_reason", "") == "")).sum())
    total_events  = int((pooled.get("row_type", "") == "event").sum())
    print(f"\n  Passed events (used by the models): {passed_events:>7}")
    print(f"  Total event rows                  : {total_events:>7}")
    print(f"  Total rows in CSV (all events)    : {len(pooled):>7}")
    print(f"  Basins                            : "
          f"{pooled['basin_number'].nunique():>7}")
    return pooled


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    main(force_rebuild=args.rebuild)