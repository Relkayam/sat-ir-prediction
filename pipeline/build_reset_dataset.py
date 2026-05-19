"""
pipeline/build_reset_dataset.py — Build reset_dataset.csv for Model 2
======================================================================
Builds one row per reset event from event_dataset.csv.

All basins have basin_role == 'clean'.
Held-out basin selection happens at bootstrap runtime — not here.

Only split_chrono is baked into the CSV (chronological 70/15/15 per basin).
All other splits computed at runtime by the bootstrap and model scripts.

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
    TRAIN_FRAC, VAL_FRAC,
    SEASON_PHASE, FIELD_NAMES,
)
from pipeline.features import MODEL2_FEATURES, TARGET_M2

_NEAR_ZERO_DRT = 0.1


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
    sin_val = float(np.sin(2 * np.pi * (month - SEASON_PHASE) / 12))
    cos_val = float(np.cos(2 * np.pi * (month - SEASON_PHASE) / 12))
    return sin_val, cos_val


# ─────────────────────────────────────────────────────────────────────────────
# Per-basin reset row builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_basin_resets(bn: int, bdf: pd.DataFrame) -> list[dict]:
    """
    Build one row per reset event for a single basin.
    All segments used to preserve chronological sequence.
    First reset: no previous reset → target = NaN → excluded from training.
    """
    bdf = bdf.sort_values("opening_valve_date").reset_index(drop=True)
    reset_rows  = bdf[bdf["row_type"] == "reset"].copy()
    all_seg_ids = sorted(s for s in bdf["segment_id"].unique() if s >= 0)

    if len(reset_rows) < 2 or len(all_seg_ids) < 2:
        return []

    rows = []
    for _, reset_row in reset_rows.iterrows():
        sid = int(reset_row["segment_id"])
        if sid not in all_seg_ids:
            continue
        sid_idx = all_seg_ids.index(sid)
        if sid_idx == 0:
            continue
        prev_sid    = all_seg_ids[sid_idx - 1]
        prev_events = bdf[(bdf["segment_id"]==prev_sid) & (bdf["row_type"]=="event")]
        if prev_events.empty:
            continue
        row = _build_row(bn, reset_row, prev_events)
        if row is not None:
            rows.append(row)
    return rows


def _build_row(bn: int, reset_row: pd.Series, prev_events: pd.DataFrame) -> dict | None:
    ird_current = pd.to_numeric(reset_row.get("IRD_at_reset", np.nan), errors="coerce")
    ird_prev    = pd.to_numeric(reset_row.get("prev_IRD_at_reset", np.nan), errors="coerce")
    if not (np.isfinite(ird_current) and ird_current > 0):
        return None
    if not (np.isfinite(ird_prev) and ird_prev > 0):
        return None

    target = float(np.log(ird_current / ird_prev))

    def _agg(col: str, func: str) -> float:
        if col not in prev_events.columns: return np.nan
        s = pd.to_numeric(prev_events[col], errors="coerce").dropna()
        return float(getattr(s, func)()) if not s.empty else np.nan

    def _std(col: str) -> float:
        if col not in prev_events.columns: return np.nan
        s = pd.to_numeric(prev_events[col], errors="coerce").dropna()
        return float(s.std()) if len(s) > 1 else 0.0

    def _frac(col: str, threshold: float) -> float:
        if col not in prev_events.columns: return np.nan
        s = pd.to_numeric(prev_events[col], errors="coerce").dropna()
        return float((s < threshold).mean()) if not s.empty else np.nan

    def _last(col: str) -> float:
        if col not in prev_events.columns: return np.nan
        sorted_ev = prev_events.sort_values("LCT")
        val = pd.to_numeric(sorted_ev[col].iloc[-1], errors="coerce")
        return float(val) if np.isfinite(val) else np.nan

    lct = pd.to_numeric(prev_events.get("LCT", pd.Series(dtype=float)), errors="coerce")
    reset_month          = pd.to_datetime(reset_row["opening_valve_date"]).month
    month_sin, month_cos = _month_encoding(reset_month)

    return {
        # Metadata
        "basin_number":    bn,
        "field_name":      FIELD_NAMES.get(int(str(bn)[0]), str(bn)),
        "segment_id":      int(reset_row["segment_id"]),
        "reset_date":      reset_row["opening_valve_date"],
        "is_good_segment": bool(reset_row.get("is_good_segment", False)),
        "basin_role":      "clean",

        # Target
        TARGET_M2: target,

        # Back-transform denominator (NOT a model feature)
        "IRD_at_reset":          float(ird_current),
        "prev_IRD_at_reset_raw": float(ird_prev),

        # Current model features
        "month_sin":  month_sin,
        "month_cos":  month_cos,
        "total_LCT":  float(lct.max()) if not lct.isna().all() else np.nan,

        # Operational features
        "mean_DrT":      _agg("DrT",   "mean"),
        "sum_DrT":       _agg("DrT",   "sum"),
        "max_DrT":       _agg("DrT",   "max"),
        "min_DrT":       _agg("DrT",   "min"),
        "std_DrT":       _std("DrT"),
        "last_DrT":      _last("DrT"),
        "frac_zero_DrT": _frac("DrT",  _NEAR_ZERO_DRT),
        "mean_FT":       _agg("FT",    "mean"),
        "sum_FT":        _agg("FT",    "sum"),
        "max_FT":        _agg("FT",    "max"),
        "mean_ALPHA":    _agg("ALPHA", "mean"),
        "min_ALPHA":     _agg("ALPHA", "min"),
        "mean_HL":       _agg("HL",    "mean"),
        "n_events":      len(prev_events),

        # Radiation — drying phase
        "mean_RD":  _agg("RD", "mean"),
        "max_RD":   _agg("RD", "max"),
        "sum_RD":   _agg("RD", "sum"),
        "last_RD":  _last("RD"),

        # Radiation — wetting phase
        "mean_RW":  _agg("RW", "mean"),
        "sum_RW":   _agg("RW", "sum"),

        # Temperature
        "mean_TD":  _agg("TD", "mean"),
        "min_TD":   _agg("TD", "min"),
        "max_TD":   _agg("TD", "max"),
        "mean_TW":  _agg("TW", "mean"),
        "max_TW":   _agg("TW", "max"),

        # Ambient conditions at reset date
        "DAT": _to_float(reset_row.get("DAT")),
        "DAR": _to_float(reset_row.get("DAR")),

        # Cross-segment history (raw — scale-dependent)
        "prev_IRD_at_reset":      _to_float(reset_row.get("prev_IRD_at_reset")),
        "prev_prev_IRD_at_reset": _to_float(reset_row.get("prev_prev_IRD_at_reset")),
        "IRD_direction":          _to_float(reset_row.get("IRD_direction")),

        # Scale-free delta features — computed post-hoc in add_delta_features()
        "prev_delta":      np.nan,
        "prev_prev_delta": np.nan,
        "IRD_trend":       np.nan,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scale-free delta features
# ─────────────────────────────────────────────────────────────────────────────

def add_delta_features(reset_df: pd.DataFrame) -> pd.DataFrame:
    """Compute prev_delta, prev_prev_delta, IRD_trend within each basin."""
    reset_df = reset_df.copy()
    for bn, bdf in reset_df.groupby("basin_number"):
        bdf  = bdf.sort_values("reset_date")
        prev_d      = pd.to_numeric(bdf[TARGET_M2], errors="coerce").shift(1)
        prev_prev_d = pd.to_numeric(bdf[TARGET_M2], errors="coerce").shift(2)
        rho_prev      = pd.to_numeric(bdf["prev_IRD_at_reset_raw"], errors="coerce")
        rho_prev_prev = rho_prev.shift(1)
        trend = (rho_prev - rho_prev_prev) / rho_prev_prev.replace(0, np.nan)
        reset_df.loc[bdf.index, "prev_delta"]      = prev_d.values
        reset_df.loc[bdf.index, "prev_prev_delta"] = prev_prev_d.values
        reset_df.loc[bdf.index, "IRD_trend"]       = trend.values

    print(f"  Delta features: "
          f"prev_delta={reset_df['prev_delta'].notna().sum()}  "
          f"prev_prev_delta={reset_df['prev_prev_delta'].notna().sum()}  "
          f"IRD_trend={reset_df['IRD_trend'].notna().sum()}")
    return reset_df


# ─────────────────────────────────────────────────────────────────────────────
# split_chrono
# ─────────────────────────────────────────────────────────────────────────────

def add_chronological_split(reset_df: pd.DataFrame) -> pd.DataFrame:
    """Chronological 70/15/15 per basin. Only split stored in CSV."""
    reset_df                 = reset_df.copy()
    reset_df["split_chrono"] = "test"
    for bn, bdf in reset_df.groupby("basin_number"):
        bdf  = bdf.sort_values("reset_date")
        n    = len(bdf)
        n_tr = max(1, round(n * TRAIN_FRAC))
        n_va = max(1, round(n * VAL_FRAC))
        reset_df.loc[bdf.index[:n_tr],          "split_chrono"] = "train"
        reset_df.loc[bdf.index[n_tr:n_tr+n_va], "split_chrono"] = "val"
    print(f"  split_chrono: {reset_df['split_chrono'].value_counts().to_dict()}")
    return reset_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    print("="*65)
    print("  BUILD RESET DATASET — pipeline/build_reset_dataset.py")
    print(f"  Target  : {TARGET_M2}")
    print(f"  Features: {MODEL2_FEATURES}")
    print(f"  NOTE: All basins clean — held-out selected at bootstrap runtime")
    print("="*65)

    if not EVENT_CSV.exists():
        raise FileNotFoundError(
            f"{EVENT_CSV} not found.\n"
            "Run: python -m pipeline.build_dataset --rebuild"
        )

    df = pd.read_csv(EVENT_CSV, parse_dates=["opening_valve_date"])
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"\n  Event dataset : {len(df):,} rows  "
          f"{df['basin_number'].nunique()} basins")

    print("\n  Building reset rows per basin...")
    all_rows: list[dict] = []
    for bn, bdf in df.groupby("basin_number"):
        rows = _build_basin_resets(int(bn), bdf)
        all_rows.extend(rows)
        if rows:
            print(f"    [{int(bn):>5}]  {len(rows):>4} resets")

    if not all_rows:
        raise ValueError("No reset rows built — check event_dataset.csv.")

    reset_df = pd.DataFrame(all_rows).reset_index(drop=True)
    print(f"\n  Reset dataset : {len(reset_df)} rows  "
          f"{reset_df['basin_number'].nunique()} basins")
    print(f"  Avg per basin : "
          f"{len(reset_df)/reset_df['basin_number'].nunique():.1f} resets")

    print("\n  Adding delta features...")
    reset_df = add_delta_features(reset_df)

    print()
    reset_df = add_chronological_split(reset_df)

    # Target distribution
    tgt = reset_df[TARGET_M2].dropna()
    print(f"\n  Target ({TARGET_M2}):")
    print(f"    mean={tgt.mean():.4f}  std={tgt.std():.4f}  "
          f"|δ|<0.10: {(tgt.abs()<0.10).mean()*100:.1f}%  "
          f"|δ|≥0.10: {(tgt.abs()>=0.10).mean()*100:.1f}%")

    # Save
    reset_df.to_csv(RESET_CSV, index=False)
    print(f"\n  Saved: {RESET_CSV}  ({len(reset_df)} rows)")
    print(f"  Columns ({len(reset_df.columns)}): {list(reset_df.columns)}")

    # Per-basin summary
    basin_summary = reset_df.groupby("basin_number").agg(
        n_resets       = (TARGET_M2,      "count"),
        mean_log_ratio = (TARGET_M2,      "mean"),
        std_log_ratio  = (TARGET_M2,      "std"),
        mean_IRD_reset = ("IRD_at_reset", "mean"),
        field_name     = ("field_name",   "first"),
    ).reset_index()
    basin_summary.to_excel(TABLES_DIR / "reset_dataset_summary.xlsx", index=False)
    print(f"  Saved: reset_dataset_summary.xlsx")

    return reset_df


if __name__ == "__main__":
    main()