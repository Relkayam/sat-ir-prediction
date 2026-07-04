"""
pipeline/build_reset_dataset.py — Build reset_dataset.csv for Model 2  (v2)
============================================================================
REWRITE RATIONALE (2026-07, Roy Elkayam)
-----------------------------------------
The v1 script had two independent, undocumented definitions of "the
previous segment" for the same row:
  1. build_dataset.py's _add_cross_segment_features chained prev_IRD_at_reset
     only across reset rows that PASSED the 4 sensor-quality filters.
  2. This script picked "the previous segment" for every OTHER feature
     (mean_DrT, mean_RD, sum_RW, DAR, ...) using all_seg_ids[idx-1] --
     the immediately preceding segment_id, full stop, regardless of
     whether ITS reset passed quality filtering.
Whenever an intervening segment's reset failed quality filtering, (1) and
(2) silently pointed at two DIFFERENT segments in the same row. Confirmed
via smoke_reset_chain_mismatch.py: match rate 100% when the true previous
segment's reset passed quality, ~62% when it didn't (2.1% of all segments
affected). A separate, LARGER, still up-for-fixing-at-the-source issue:
build_dataset.py's own segmentation (_ceiling_reset_search) marks a row
'reset' for every tillage timestamp with no credibility check, so many of
its "resets" never earned a new ceiling and just inherited the previous
one -- confirmed via smoke_reset_ird_mismatch.py: 0% of zero-delta resets
have a ceiling that matches their own independently-fitted IRD (vs 93.5%
for non-zero-delta resets). own IRD is essentially NEVER NaN
(smoke_reset_nan_ird.py: 0.0%), so this is NOT a failed-fit/IRD_MAX issue --
it's a wrong-anchor issue.

STRATEGY (this rewrite)
------------------------
  1. Per basin (never pooled -- segment_id is not globally unique, see the
     warning in build_dataset.py's _compute_prev_features):
  2. Build ONE ceiling "spine" dataframe: every row_type=='reset' row,
     UNCONDITIONALLY -- no quality gating, nothing dropped. Columns:
     segment_id, reset_date, IRD_at_reset (the assigned ceiling, straight
     from event_dataset.csv, never recomputed), own_IRD (that reset event's
     own independently-fitted IRD), own_reset_passed_quality (the 4 sensor
     filters), is_self_derived (own_IRD == IRD_at_reset, within tolerance --
     the reliability signal; a proxy for the real upstream credible-reset
     flag, which tillage_features.py computes but never persists -- see
     note in the handoff to Roy).
  3. prev_IRD_at_reset / prev_prev_IRD_at_reset / the target delta / the
     delta history / IRD_direction are ALL computed by .shift(1)/.shift(2)
     on this ONE spine. There is no second adjacency definition left to
     drift out of sync with the first -- that's the actual fix for the
     chain-mismatch bug, not a patch on top of it.
  4. The other features (mean_DrT, mean_RD, sum_RW, DAR, n_events, ...) are
     aggregated from the SAME previous segment's event rows (spine.iloc[i-1]),
     guaranteed consistent with the target's "previous" by construction.
     If that previous segment has zero event rows, these are NaN -- the row
     is still kept.
  5. NOTHING is dropped for "insufficient data" anymore. The first segment
     of a basin gets a row with target/prev_* = NaN (nothing to compare to)
     instead of being silently omitted. Completeness over convenience --
     Roy: "the dataset should be complete... allow NaN."
  6. Permanent invariant checks (_assert_consistency), run every build, not
     a one-off smoke: reset_dataset's IRD_at_reset must match
     event_dataset's value for that segment; prev_IRD_at_reset_raw must
     match event_dataset's value for the TRUE chronologically-previous
     segment (ignoring quality filtering). This is exactly the drift that
     went unnoticed before -- an assertion catches it forever.

split_chrono is still the only split baked into the CSV -- per-basin
chronological TRAIN/VAL, for LightGBM early stopping only. Test = held-out
basins, chosen at bootstrap runtime (unchanged from v1).

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
    TRAIN_FRAC,
    SEASON_PHASE, FIELD_NAMES,
)
from pipeline.features import MODEL2_FEATURES, TARGET_M2

_NEAR_ZERO_DRT = 0.1

# Tolerance for "this reset's own IRD equals its assigned ceiling" (Roy:
# tolerance is fine as-is). Confirmed via smoke_reset_ird_mismatch.py.
SELF_DERIVED_TOL = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_float(val) -> float:
    try:
        f = float(val)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _month_encoding(month: int) -> tuple[float, float]:
    # NOTE: month_sin/month_cos are NOT in the final MODEL2_FEATURES (Set-D
    # dropped month encoding in favour of radiation features). Kept only as
    # candidate columns for the feature-selection comparison (Table S3).
    sin_val = float(np.sin(2 * np.pi * (month - SEASON_PHASE) / 12))
    cos_val = float(np.cos(2 * np.pi * (month - SEASON_PHASE) / 12))
    return sin_val, cos_val


def _agg(events: pd.DataFrame, col: str, func: str) -> float:
    if events.empty or col not in events.columns:
        return np.nan
    s = pd.to_numeric(events[col], errors="coerce").dropna()
    return float(getattr(s, func)()) if not s.empty else np.nan


def _std(events: pd.DataFrame, col: str) -> float:
    if events.empty or col not in events.columns:
        return np.nan
    s = pd.to_numeric(events[col], errors="coerce").dropna()
    return float(s.std()) if len(s) > 1 else (0.0 if len(s) == 1 else np.nan)


def _frac(events: pd.DataFrame, col: str, threshold: float) -> float:
    if events.empty or col not in events.columns:
        return np.nan
    s = pd.to_numeric(events[col], errors="coerce").dropna()
    return float((s < threshold).mean()) if not s.empty else np.nan


def _last(events: pd.DataFrame, col: str) -> float:
    if events.empty or col not in events.columns or "LCT" not in events.columns:
        return np.nan
    sorted_ev = events.sort_values("LCT")
    val = pd.to_numeric(sorted_ev[col].iloc[-1], errors="coerce")
    return float(val) if np.isfinite(val) else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — the ceiling spine (single source of truth), per basin
# ─────────────────────────────────────────────────────────────────────────────

def _build_ceiling_spine(bdf: pd.DataFrame) -> pd.DataFrame:
    """
    ONE row per segment_id, UNCONDITIONAL. This is the single source of
    truth for the ceiling sequence -- nothing here decides what counts as
    a "real" reset; reliability is attached as columns, not used to drop
    rows. bdf must already be a SINGLE basin's slice of event_dataset.csv.

    DESIGN DECISION (Roy, this session): IRD_at_reset is now ALWAYS this
    segment's own reset event's independently-fitted IRD ("own_IRD"), full
    stop -- NOT the CSV's original IRD_at_reset column. The CSV's original
    value was forward-filled by tillage_features.py's credible-reset search
    (its own drainage-response threshold + a 15-day minimum spacing) -- a
    DIFFERENT segmentation than build_dataset.py's own (which segments on
    every cleaned tillage timestamp, no credibility gate). Per Roy's article
    design, every tillage-triggered reset is a real segment boundary, "good
    or bad" -- so the ceiling must come from THIS segmentation's own event,
    never the other one's. Confirmed as the plateau/zero-delta mechanism via
    smoke_reset_ird_mismatch.py + the visual test (event IRD peaks not
    appearing in IRD_at_reset). The CSV's original value is kept as
    IRD_at_reset_db_original for before/after comparison only -- unused below.
    """
    resets = bdf[(bdf["row_type"] == "reset") & (bdf["segment_id"] >= 0)].copy()
    resets = resets.sort_values("opening_valve_date")
    resets = resets.drop_duplicates(subset="segment_id", keep="first")

    spine = pd.DataFrame({
        "segment_id":   resets["segment_id"].astype(int).to_numpy(),
        "reset_date":   pd.to_datetime(resets["opening_valve_date"]).to_numpy(),
        "own_IRD":      pd.to_numeric(resets["IRD"], errors="coerce").to_numpy(),
        "IRD_at_reset_db_original": pd.to_numeric(
            resets["IRD_at_reset"], errors="coerce").to_numpy(),
        "own_reset_passed_quality": (
            resets.get("filter_reason", pd.Series("", index=resets.index))
            .fillna("").eq("").to_numpy()
        ),
    }).sort_values("segment_id").reset_index(drop=True)

    # THE ceiling, now always self-derived by construction.
    spine["IRD_at_reset"] = spine["own_IRD"]

    with np.errstate(divide="ignore", invalid="ignore"):
        log_gap = np.log(spine["IRD_at_reset_db_original"] / spine["own_IRD"])
    # diagnostic only -- shows where the OLD (DB-sourced) ceiling would have
    # disagreed with this segment's own reset. Not used for anything downstream.
    spine["db_ceiling_was_inherited"] = np.abs(log_gap) >= SELF_DERIVED_TOL
    spine["is_self_derived"] = True  # always true now, kept for schema stability

    # ---- THE single adjacency definition. Everything downstream shifts
    # off of THIS array -- nothing else independently re-derives "previous".
    spine["prev_IRD_at_reset"]      = spine["IRD_at_reset"].shift(1)
    spine["prev_prev_IRD_at_reset"] = spine["IRD_at_reset"].shift(2)
    spine["prev_reset_date"]        = spine["reset_date"].shift(1)
    spine["prev_prev_reset_date"]   = spine["reset_date"].shift(2)
    spine["prev_is_self_derived"]      = spine["is_self_derived"].shift(1)
    spine["prev_reset_passed_quality"] = spine["own_reset_passed_quality"].shift(1)

    with np.errstate(divide="ignore", invalid="ignore"):
        spine[TARGET_M2] = np.log(spine["IRD_at_reset"] / spine["prev_IRD_at_reset"])
    spine["prev_delta"]      = spine[TARGET_M2].shift(1)
    spine["prev_prev_delta"] = spine[TARGET_M2].shift(2)

    dt_days = (spine["prev_reset_date"] - spine["prev_prev_reset_date"]).dt.total_seconds() / 86400.0
    with np.errstate(divide="ignore", invalid="ignore"):
        direction = (spine["prev_IRD_at_reset"] - spine["prev_prev_IRD_at_reset"]) / dt_days
    spine["IRD_direction"] = direction.where(dt_days > 0)

    rho_prev, rho_prev_prev = spine["prev_IRD_at_reset"], spine["prev_prev_IRD_at_reset"]
    with np.errstate(divide="ignore", invalid="ignore"):
        spine["IRD_trend"] = (rho_prev - rho_prev_prev) / rho_prev_prev.replace(0, np.nan)

    return spine


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — per-segment row: spine fields + aggregate features from the SAME
# previous segment (guaranteed, since both come from spine adjacency)
# ─────────────────────────────────────────────────────────────────────────────

def _build_basin_resets(bn: int, bdf: pd.DataFrame) -> list[dict]:
    bdf = bdf.sort_values("opening_valve_date").reset_index(drop=True)
    spine = _build_ceiling_spine(bdf)
    if spine.empty:
        return []

    rows: list[dict] = []
    for i in range(len(spine)):
        s = spine.iloc[i]

        if i == 0:
            prev_events = bdf.iloc[0:0]  # empty -- no previous segment at all
        else:
            prev_sid = int(spine.iloc[i - 1]["segment_id"])
            prev_events = bdf[(bdf["segment_id"] == prev_sid) & (bdf["row_type"] == "event")]

        reset_month = pd.to_datetime(s["reset_date"]).month
        month_sin, month_cos = _month_encoding(reset_month)
        lct = pd.to_numeric(prev_events.get("LCT", pd.Series(dtype=float)), errors="coerce")

        rows.append({
            # Metadata
            "basin_number": bn,
            "field_name":   FIELD_NAMES.get(int(str(bn)[0]), str(bn)),
            "segment_id":   int(s["segment_id"]),
            "reset_date":   s["reset_date"],
            "basin_role":   "clean",

            # Target -- NaN for the first segment (nothing to compare to).
            # Kept, not dropped: completeness over convenience.
            TARGET_M2: _to_float(s[TARGET_M2]),

            # Ceiling + reliability -- the spine, unconditional. IRD_at_reset
            # is ALWAYS this segment's own fitted IRD now (see _build_ceiling_
            # spine docstring). IRD_at_reset_db_original / db_ceiling_was_
            # inherited are diagnostic-only, for before/after comparison
            # against the old (optisat-sourced) ceiling.
            "IRD_at_reset":            _to_float(s["IRD_at_reset"]),
            "prev_IRD_at_reset_raw":   _to_float(s["prev_IRD_at_reset"]),
            "IRD_at_reset_db_original": _to_float(s["IRD_at_reset_db_original"]),
            "db_ceiling_was_inherited": bool(s["db_ceiling_was_inherited"]),
            "own_reset_passed_quality":      bool(s["own_reset_passed_quality"]),
            "prev_reset_passed_quality":     (bool(s["prev_reset_passed_quality"])
                                               if pd.notna(s["prev_reset_passed_quality"]) else pd.NA),
            "is_self_derived":               bool(s["is_self_derived"]),
            "prev_is_self_derived":          (bool(s["prev_is_self_derived"])
                                               if pd.notna(s["prev_is_self_derived"]) else pd.NA),

            # Candidate seasonal encoding (not in final feature set)
            "month_sin": month_sin,
            "month_cos": month_cos,
            "total_LCT": float(lct.max()) if not lct.isna().all() else np.nan,

            # Operational features (from the previous segment's events; NaN if none)
            "mean_DrT":      _agg(prev_events, "DrT", "mean"),
            "sum_DrT":       _agg(prev_events, "DrT", "sum"),
            "max_DrT":       _agg(prev_events, "DrT", "max"),
            "min_DrT":       _agg(prev_events, "DrT", "min"),
            "std_DrT":       _std(prev_events, "DrT"),
            "last_DrT":      _last(prev_events, "DrT"),
            "frac_zero_DrT": _frac(prev_events, "DrT", _NEAR_ZERO_DRT),
            "mean_FT":       _agg(prev_events, "FT", "mean"),
            "sum_FT":        _agg(prev_events, "FT", "sum"),
            "max_FT":        _agg(prev_events, "FT", "max"),
            "mean_ALPHA":    _agg(prev_events, "ALPHA", "mean"),
            "min_ALPHA":     _agg(prev_events, "ALPHA", "min"),
            "mean_HL":       _agg(prev_events, "HL", "mean"),
            "n_events":      len(prev_events),

            # Radiation -- drying phase
            "mean_RD": _agg(prev_events, "RD", "mean"),
            "max_RD":  _agg(prev_events, "RD", "max"),
            "sum_RD":  _agg(prev_events, "RD", "sum"),
            "last_RD": _last(prev_events, "RD"),

            # Radiation -- wetting phase
            "mean_RW": _agg(prev_events, "RW", "mean"),
            "sum_RW":  _agg(prev_events, "RW", "sum"),

            # Temperature
            "mean_TD": _agg(prev_events, "TD", "mean"),
            "min_TD":  _agg(prev_events, "TD", "min"),
            "max_TD":  _agg(prev_events, "TD", "max"),
            "mean_TW": _agg(prev_events, "TW", "mean"),
            "max_TW":  _agg(prev_events, "TW", "max"),

            # Ambient conditions at reset date
            "DAT": _to_float(bdf.loc[bdf["segment_id"] == int(s["segment_id"]),
                                      "DAT"].iloc[0] if (bdf["segment_id"] == int(s["segment_id"])).any() else np.nan),
            "DAR": _to_float(bdf.loc[bdf["segment_id"] == int(s["segment_id"]),
                                      "DAR"].iloc[0] if (bdf["segment_id"] == int(s["segment_id"])).any() else np.nan),

            # Cross-segment history -- all from the spine, one definition
            "prev_IRD_at_reset":      _to_float(s["prev_IRD_at_reset"]),
            "prev_prev_IRD_at_reset": _to_float(s["prev_prev_IRD_at_reset"]),
            "IRD_direction":          _to_float(s["IRD_direction"]),
            "prev_delta":             _to_float(s["prev_delta"]),
            "prev_prev_delta":        _to_float(s["prev_prev_delta"]),
            "IRD_trend":              _to_float(s["IRD_trend"]),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# split_chrono — per-basin chronological TRAIN/VAL (early stopping only)
# ─────────────────────────────────────────────────────────────────────────────

def add_chronological_split(reset_df: pd.DataFrame) -> pd.DataFrame:
    reset_df = reset_df.copy()
    reset_df["split_chrono"] = "val"
    for bn, bdf in reset_df.groupby("basin_number"):
        bdf = bdf.sort_values("reset_date")
        n = len(bdf)
        n_tr = max(1, round(n * TRAIN_FRAC))
        reset_df.loc[bdf.index[:n_tr], "split_chrono"] = "train"
    print(f"  split_chrono: {reset_df['split_chrono'].value_counts().to_dict()}")
    return reset_df


# ─────────────────────────────────────────────────────────────────────────────
# Permanent consistency assertions -- run every build, not a one-off smoke
# ─────────────────────────────────────────────────────────────────────────────

def _assert_consistency(reset_df: pd.DataFrame, event_df: pd.DataFrame) -> None:
    """
    Guards against the exact chain-mismatch bug this rewrite fixes ever
    silently reappearing. Fails loudly (AssertionError) rather than
    producing a CSV that looks fine but isn't.

    NOTE: checks against event_dataset's own IRD column (the reset event's
    independently-fitted rate) -- NOT the CSV's original IRD_at_reset column,
    since we now deliberately diverge from that (see _build_ceiling_spine).
    """
    ceiling_map: dict[tuple[int, int], float] = {}
    prev_seg_map: dict[tuple[int, int], int] = {}

    for bn, bdf in event_df.groupby("basin_number"):
        resets = (bdf[(bdf["row_type"] == "reset") & (bdf["segment_id"] >= 0)]
                  .sort_values("opening_valve_date")
                  .drop_duplicates(subset="segment_id", keep="first"))
        seg_ids = sorted(resets["segment_id"].astype(int).tolist())
        ceiling = dict(zip(resets["segment_id"].astype(int),
                            pd.to_numeric(resets["IRD"], errors="coerce")))
        for sid, val in ceiling.items():
            ceiling_map[(int(bn), int(sid))] = val
        for idx in range(1, len(seg_ids)):
            prev_seg_map[(int(bn), seg_ids[idx])] = seg_ids[idx - 1]

    n_checked = n_bad_ceiling = n_bad_prev = 0
    for _, row in reset_df.iterrows():
        key = (int(row["basin_number"]), int(row["segment_id"]))
        n_checked += 1

        expected_ceiling = ceiling_map.get(key)
        if expected_ceiling is not None and np.isfinite(expected_ceiling):
            if not np.isclose(row["IRD_at_reset"], expected_ceiling, atol=1e-9):
                n_bad_ceiling += 1

        prev_sid = prev_seg_map.get(key)
        if prev_sid is not None:
            expected_prev = ceiling_map.get((int(row["basin_number"]), prev_sid))
            if (expected_prev is not None and np.isfinite(expected_prev)
                    and np.isfinite(row["prev_IRD_at_reset_raw"])):
                if not np.isclose(row["prev_IRD_at_reset_raw"], expected_prev, atol=1e-9):
                    n_bad_prev += 1

    print(f"  [consistency check] rows checked: {n_checked}  "
          f"ceiling mismatches: {n_bad_ceiling}  prev-ceiling mismatches: {n_bad_prev}")
    assert n_bad_ceiling == 0, (
        f"{n_bad_ceiling} rows have IRD_at_reset != event_dataset's value for "
        "that segment -- the ceiling spine is not the single source of truth "
        "anymore. Fix before proceeding."
    )
    assert n_bad_prev == 0, (
        f"{n_bad_prev} rows have prev_IRD_at_reset_raw != event_dataset's true "
        "chronologically-previous segment. The chain-mismatch bug this rewrite "
        "fixes has reappeared. Fix before proceeding."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    print("=" * 65)
    print("  BUILD RESET DATASET (v2) — pipeline/build_reset_dataset.py")
    print(f"  Target  : {TARGET_M2}")
    print(f"  Features: {MODEL2_FEATURES}")
    print("  Ceiling spine: ALL reset rows, unconditional. Reliability is a "
          "column (is_self_derived / *_passed_quality), never a filter here.")
    print("=" * 65)

    if not EVENT_CSV.exists():
        raise FileNotFoundError(
            f"{EVENT_CSV} not found.\nRun: python -m pipeline.build_dataset --rebuild"
        )

    df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"\n  Event dataset : {len(df):,} rows  {df['basin_number'].nunique()} basins")

    print("\n  Building reset rows per basin (spine-first, allowing NaN)...")
    all_rows: list[dict] = []
    for bn, bdf in df.groupby("basin_number"):
        rows = _build_basin_resets(int(bn), bdf)
        all_rows.extend(rows)
        if rows:
            print(f"    [{int(bn):>5}]  {len(rows):>4} resets")

    if not all_rows:
        raise ValueError("No reset rows built — check event_dataset.csv.")

    reset_df = pd.DataFrame(all_rows).reset_index(drop=True)
    print(f"\n  Reset dataset : {len(reset_df)} rows  {reset_df['basin_number'].nunique()} basins "
          f"(v1 dropped the first segment of every basin + any segment with "
          f"an empty previous segment; v2 keeps them with NaN target/features)")

    print()
    reset_df = add_chronological_split(reset_df)

    n_changed = int(reset_df["db_ceiling_was_inherited"].sum())
    print(f"\n  BEFORE/AFTER: {n_changed}/{len(reset_df)} "
          f"({100*n_changed/len(reset_df):.1f}%) segments now have a DIFFERENT "
          f"ceiling than the old (optisat DB-sourced) IRD_at_reset column --")
    print("  i.e. this is the fraction of segments where the old value was "
          "inherited from tillage_features.py's separate credibility-gated "
          "segmentation, not this segmentation's own reset event.")
    print("  is_self_derived is now trivially True for every row by "
          "construction (kept for schema stability, not informative anymore).")

    tgt = reset_df[TARGET_M2].dropna()
    print(f"\n  Target ({TARGET_M2}), non-NaN only:")
    print(f"    mean={tgt.mean():.4f}  std={tgt.std():.4f}  "
          f"|δ|<0.10: {(tgt.abs()<0.10).mean()*100:.1f}%  "
          f"|δ|≥0.10: {(tgt.abs()>=0.10).mean()*100:.1f}%")

    print("\n  Running permanent consistency assertions...")
    _assert_consistency(reset_df, df)
    print("  OK — ceiling spine and adjacency are internally consistent.")

    reset_df.to_csv(RESET_CSV, index=False)
    print(f"\n  Saved: {RESET_CSV}  ({len(reset_df)} rows)")
    print(f"  Columns ({len(reset_df.columns)}): {list(reset_df.columns)}")

    basin_summary = reset_df.groupby("basin_number").agg(
        n_resets           = (TARGET_M2,      "count"),
        mean_log_ratio     = (TARGET_M2,      "mean"),
        std_log_ratio      = (TARGET_M2,      "std"),
        mean_IRD_reset     = ("IRD_at_reset", "mean"),
        frac_self_derived  = ("is_self_derived", "mean"),
        field_name         = ("field_name",   "first"),
    ).reset_index()
    basin_summary.to_excel(TABLES_DIR / "reset_dataset_summary.xlsx", index=False)
    print(f"  Saved: reset_dataset_summary.xlsx")

    return reset_df


if __name__ == "__main__":
    main()