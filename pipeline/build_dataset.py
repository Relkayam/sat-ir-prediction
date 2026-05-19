"""
pipeline/build_dataset.py — Build event_dataset.csv from DuckDB
================================================================
Reads all flooding events for all basins, applies quality filters,
identifies segments, computes features, and saves a single CSV.

Every raw event is KEPT in the CSV with a filter_reason column.

basin_role values
-----------------
  'clean'   — all basins (no fixed held-out basins)

  Held-out basin selection is performed at runtime by the bootstrap
  (experiments/run_bootstrap.py), not at build time.
  Outlier detection is performed by analysis/basin_analysis.py.

Usage
-----
  python -m pipeline.build_dataset              # uses cached CSV if exists
  python -m pipeline.build_dataset --rebuild    # forces rebuild from DuckDB
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATA_CUTOFF, QUALITY_FILTER,
    MIN_EVENTS_PER_SEGMENT, MIN_SEGMENT_R2, PEARSON_OUTLIER_THRESHOLD,
    TRAIN_FRAC, VAL_FRAC, RANDOM_SEED,
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
            f"ORDER BY opening_valve_date"
        )
    if df.empty:
        return None
    df["opening_valve_date"] = pd.to_datetime(df["opening_valve_date"])
    return df.sort_values("opening_valve_date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1b — Fill drying-window features with 0 when DrT is zero
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
# Step 2 — Tag quality filter
# ─────────────────────────────────────────────────────────────────────────────

def _tag_quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["filter_reason"] = ""
    after = df["opening_valve_date"] >= pd.Timestamp(DATA_CUTOFF)
    df.loc[after, "filter_reason"] = "after_cutoff"
    for col, threshold in QUALITY_FILTER.items():
        if col not in df.columns:
            continue
        unflagged = df["filter_reason"] == ""
        fails     = pd.to_numeric(df[col], errors="coerce").fillna(0) < threshold
        df.loc[unflagged & fails, "filter_reason"] = f"quality_filter_{col}"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def _identify_segments(df_passed: pd.DataFrame) -> pd.DataFrame:
    df         = df_passed.copy()
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
    df.loc[df["row_type"] == "pre_segment", "filter_reason"] = "pre_segment"
    return df


def _ceiling_reset_search(df_passed: pd.DataFrame, basin_number: int) -> pd.DataFrame:
    path = SatixPaths.BASIN_DB_DIR / f"basin_{int(basin_number)}.duckdb"
    if not path.exists():
        return _identify_segments(df_passed)
    with DuckDBManager(str(path)).connect_context() as db:
        tillage_df = read_clean_tillage_events(db)
    if tillage_df.empty:
        return _identify_segments(df_passed)

    df        = df_passed.copy().sort_values("opening_valve_date").reset_index(drop=True)
    df_scored = _add_scoring_features(df)
    df_scored["opening_valve_date"] = (
        pd.to_datetime(df_scored["opening_valve_date"])
        .dt.tz_localize(None).astype("datetime64[ns]"))
    tillage_df["timestamp"] = (
        pd.to_datetime(tillage_df["timestamp"])
        .dt.tz_localize(None).astype("datetime64[ns]"))

    times_ns = df_scored["opening_valve_date"].values.astype("int64")
    ird_vals = pd.to_numeric(df_scored["IRD"], errors="coerce").to_numpy(float)
    quality  = (df_scored["event_quality_ok"].fillna(False).to_numpy(dtype=bool)
                if "event_quality_ok" in df_scored.columns
                else np.ones(len(df_scored), dtype=bool))
    _wait_ns = int(pd.Timedelta(days=_TillageConstants.TILLAGE_MAX_WAIT_DAYS).total_seconds() * 1e9)
    _back_ns = int(pd.Timedelta(days=_TillageConstants.TILLAGE_BACKWARD_WINDOW_DAYS).total_seconds() * 1e9)
    till_ns  = tillage_df["timestamp"].values.astype("int64")

    reset_event_indices: list[int] = []
    for tau_ns in till_ns:
        before_pos = np.where((times_ns >= tau_ns - _back_ns) & (times_ns < tau_ns) & quality)[0]
        after_pos  = np.where((times_ns >= tau_ns) & (times_ns <= tau_ns + _wait_ns) & quality)[0]
        ird_1 = (float(ird_vals[before_pos[-1]]) if len(before_pos) and np.isfinite(ird_vals[before_pos[-1]]) else np.nan)
        ird_2 = (float(ird_vals[after_pos[0]])   if len(after_pos)  and np.isfinite(ird_vals[after_pos[0]])   else np.nan)
        if np.isfinite(ird_1) and np.isfinite(ird_2):
            chosen = int(before_pos[-1]) if ird_1 >= ird_2 else int(after_pos[0])
        elif np.isfinite(ird_1): chosen = int(before_pos[-1])
        elif np.isfinite(ird_2): chosen = int(after_pos[0])
        else: continue
        reset_event_indices.append(chosen)

    reset_event_indices = sorted(set(reset_event_indices))
    if not reset_event_indices:
        return _identify_segments(df_passed)

    reset_set   = set(reset_event_indices)
    row_type    = np.full(len(df), "pre_segment", dtype=object)
    seg_id_arr  = np.full(len(df), -1, dtype=int)
    seg_counter = 0; in_segment = False
    for i in range(len(df)):
        if i in reset_set:
            seg_counter += 1; in_segment = True; row_type[i] = "reset"
        elif in_segment:
            row_type[i] = "event"
        if in_segment:
            seg_id_arr[i] = seg_counter
    df["row_type"]   = row_type
    df["segment_id"] = seg_id_arr
    df.loc[df["row_type"] == "pre_segment", "filter_reason"] = "pre_segment"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Steps 4 & 5 — IRD_norm and HL
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ird_norm(df: pd.DataFrame) -> pd.DataFrame:
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
# Steps 6 & 7 — prev_* and cum_* features
# ─────────────────────────────────────────────────────────────────────────────

def _compute_prev_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in PREV_SOURCE_COLS:
        if col not in df.columns:
            continue
        df[f"prev_{col}"] = np.nan
        for sid, grp in df.groupby("segment_id"):
            if sid < 0: continue
            df.loc[grp.index, f"prev_{col}"] = (
                pd.to_numeric(grp[col], errors="coerce").shift(1).values)
    return df


def _compute_cum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for cum_col, src_col in CUM_SOURCE_COLS.items():
        df[cum_col] = np.nan
        if src_col not in df.columns: continue
        for sid, grp in df.groupby("segment_id"):
            if sid < 0: continue
            df.loc[grp.index, cum_col] = (
                pd.to_numeric(grp[src_col], errors="coerce").fillna(0).cumsum().values)
    df["event_count"] = np.nan
    for sid, grp in df.groupby("segment_id"):
        if sid < 0: continue
        df.loc[grp.index, "event_count"] = np.arange(1, len(grp) + 1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 7b — Cross-segment features
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
            prev_ird  = reset_rows.loc[idx-1, "IRD_at_reset"]
            vals = dict(
                prev_IRD_at_reset      = float(prev_ird) if pd.notna(prev_ird) else np.nan,
                prev_prev_IRD_at_reset = np.nan,
                prev_reset_date        = reset_rows.loc[idx-1, "opening_valve_date"],
                IRD_direction          = np.nan,
            )
        else:
            prev_ird      = float(reset_rows.loc[idx-1, "IRD_at_reset"])
            prev_prev_ird = float(reset_rows.loc[idx-2, "IRD_at_reset"])
            prev_date     = reset_rows.loc[idx-1, "opening_valve_date"]
            prev_prev_date= reset_rows.loc[idx-2, "opening_valve_date"]
            dt = ((prev_date - prev_prev_date).total_seconds() / 86400.0
                  if pd.notna(prev_date) and pd.notna(prev_prev_date) else np.nan)
            direction = (
                (prev_ird - prev_prev_ird) / dt
                if np.isfinite(prev_ird) and np.isfinite(prev_prev_ird)
                   and np.isfinite(dt) and dt > 0 else np.nan)
            vals = dict(prev_IRD_at_reset=prev_ird, prev_prev_IRD_at_reset=prev_prev_ird,
                        prev_reset_date=prev_date, IRD_direction=direction)
        mask = df["segment_id"] == sid
        for col, val in vals.items():
            df.loc[mask, col] = val
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Segment quality
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


def _tag_segment_quality(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, bool], dict[int, str]]:
    df = df.copy()
    df["is_good_segment"] = False
    seg_quality: dict[int, bool] = {}
    seg_reason:  dict[int, str]  = {}
    seg_decay:   dict[int, dict] = {}

    for sid in [s for s in df["segment_id"].unique() if s >= 0]:
        grp        = df[df["segment_id"] == sid]
        event_rows = grp[grp["row_type"] == "event"]
        lct        = event_rows["LCT"].values.astype(float)
        inorm      = event_rows["IRD_norm"].values
        valid      = np.isfinite(lct) & np.isfinite(inorm) & (lct > 0)

        if valid.sum() < MIN_EVENTS_PER_SEGMENT:
            seg_quality[sid] = False; seg_reason[sid] = "too_few_events"; continue
        pr, _ = pearsonr(lct[valid], inorm[valid])
        if float(pr) > PEARSON_OUTLIER_THRESHOLD:
            seg_quality[sid] = False; seg_reason[sid] = "pearson_r_positive"; continue
        fit = _fit_decay(lct[valid], inorm[valid])
        if fit is None:
            seg_quality[sid] = False; seg_reason[sid] = "fit_failed"; continue
        if not (np.isfinite(fit["seg_r2"]) and fit["seg_r2"] >= MIN_SEGMENT_R2):
            seg_quality[sid] = False; seg_reason[sid] = "r2_below_threshold"; continue
        seg_quality[sid] = True; seg_reason[sid] = ""; seg_decay[sid] = fit

    df["is_good_segment"] = df["segment_id"].map(
        lambda s: seg_quality.get(s, False) if s >= 0 else False)
    for sid, reason in seg_reason.items():
        if reason:
            df.loc[(df["segment_id"]==sid) & (df["filter_reason"]==""), "filter_reason"] = reason
    for param in ["seg_lambda", "seg_a", "seg_b", "seg_r2"]:
        df[param] = df["segment_id"].map(
            lambda s, p=param: seg_decay.get(s, {}).get(p, np.nan))
    return df, seg_quality, seg_reason


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Assign splits (chrono random — no held-out column)
# ─────────────────────────────────────────────────────────────────────────────

def _assign_splits(
    df:          pd.DataFrame,
    seg_quality: dict[int, bool],
) -> pd.DataFrame:
    """
    Assign random 70/15/15 train/val/test split by segment.
    No held-out column — held-out selection happens at bootstrap runtime.
    """
    df = df.copy()
    df["split"] = "excluded"

    good_segments = sorted(s for s, q in seg_quality.items() if q)
    if not good_segments:
        return df

    rng      = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(good_segments)
    n_train  = max(1, round(len(shuffled) * TRAIN_FRAC))
    n_val    = max(1, round(len(shuffled) * VAL_FRAC))
    train_segs = set(shuffled[:n_train].tolist())
    val_segs   = set(shuffled[n_train:n_train+n_val].tolist())

    def _get_split(row) -> str:
        if not row["is_good_segment"]: return "excluded"
        if row["row_type"] == "reset": return "reset"
        sid = row["segment_id"]
        if sid in train_segs: return "train"
        if sid in val_segs:   return "val"
        return "test"

    df["split"] = df.apply(_get_split, axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Per-basin builder
# ─────────────────────────────────────────────────────────────────────────────

def _fill_missing_columns(df: pd.DataFrame, target_columns) -> None:
    for col in target_columns:
        if col not in df.columns:
            df[col] = np.nan


def _attach_metadata(df: pd.DataFrame, basin_number: int, area_m2: float) -> None:
    df["basin_number"] = basin_number
    df["facility"]     = int(str(basin_number)[0])
    df["field_name"]   = FIELD_NAMES.get(int(str(basin_number)[0]), str(basin_number))
    df["area_m2"]      = area_m2
    df["basin_role"]   = "clean"   # all basins are clean — no fixed held-out


def build_basin_dataset(basin_number: int) -> Optional[pd.DataFrame]:
    df = _read_raw_events(basin_number)
    if df is None:
        return None

    area_m2 = _get_basin_area(basin_number)
    n_raw   = len(df)

    df = _fill_zero_drying(df)
    df = _tag_quality_filter(df)
    df["is_good_segment"] = False

    passed    = df["filter_reason"] == ""
    df_passed = df[passed].copy().reset_index(drop=True)
    n_passed  = len(df_passed)

    if n_passed < 10:
        df["row_type"]   = "quality_filtered"
        df["segment_id"] = -1
        df["split"]      = "excluded"
        _attach_metadata(df, basin_number, area_m2)
        return df

    df_passed = _ceiling_reset_search(df_passed, basin_number)
    df_passed = _compute_ird_norm(df_passed)
    df_passed = _compute_hl(df_passed, area_m2)
    df_passed = _compute_prev_features(df_passed)
    df_passed = _compute_cum_features(df_passed)
    df_passed = _add_cross_segment_features(df_passed)
    df_passed, seg_quality, seg_reason = _tag_segment_quality(df_passed)
    df_passed = _assign_splits(df_passed, seg_quality)

    filtered_rows = df[~passed].copy()
    _fill_missing_columns(filtered_rows, df_passed.columns)

    df_full = pd.concat([df_passed, filtered_rows], ignore_index=True)
    df_full = df_full.sort_values("opening_valve_date").reset_index(drop=True)
    _attach_metadata(df_full, basin_number, area_m2)

    n_good_segs   = sum(seg_quality.values())
    n_bad_segs    = len(seg_quality) - n_good_segs
    n_good_events = int(((df_full["row_type"]=="event") & df_full["is_good_segment"]).sum())
    reason_counts = Counter(seg_reason[s] for s in seg_reason if not seg_quality[s])
    print(
        f"  [{basin_number:>5}]  raw={n_raw:>5}  qpass={n_passed:>5}  "
        f"good_seg={n_good_segs:>3}  bad_seg={n_bad_segs:>3}  "
        f"good_ev={n_good_events:>5}"
        + (f"  excl: {dict(reason_counts)}" if reason_counts else "")
    )
    return df_full


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

    step_order = [
        ("Raw events (study period)",                None),
        ("Drainage R² < 0.94 (sensor quality)",     "quality_filter_IRD_R_squared"),
        ("CIV < 3,000 m³",                          "quality_filter_CIV"),
        ("Ct < 20 h",                               "quality_filter_Ct"),
        ("AL < 5 cm",                               "quality_filter_AL"),
        ("Pre-segment (before 1st reset)",          "pre_segment"),
        ("Too few events (<4)",                     "too_few_events"),
        ("No decay signal (Pearson r > −0.05)",     "pearson_r_positive"),
        ("Fit failed",                              "fit_failed"),
        ("R² below threshold (<0.10)",              "r2_below_threshold"),
        ("Good events (passed all quality filters)",""),
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
    print(f"  MIN_EVENTS_PER_SEGMENT : {MIN_EVENTS_PER_SEGMENT}")
    print(f"  MIN_SEGMENT_R2         : {MIN_SEGMENT_R2}")
    print(f"  PEARSON_THRESHOLD      : {PEARSON_OUTLIER_THRESHOLD}")
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

    good_events  = int(((pooled.get("row_type","")=="event") &
                        pooled.get("is_good_segment", False)).sum())
    total_events = int((pooled.get("row_type","")=="event").sum())
    print(f"\n  Good events (passed all quality filters): {good_events:>7}")
    print(f"  Total event rows (quality passed)        : {total_events:>7}")
    print(f"  Total rows in CSV (all events)           : {len(pooled):>7}")
    print(f"  Basins                                   : "
          f"{pooled['basin_number'].nunique():>7}")
    return pooled


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    main(force_rebuild=args.rebuild)