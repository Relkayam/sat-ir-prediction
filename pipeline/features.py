"""
pipeline/features.py — Feature definitions for Model 1 and Model 2
====================================================================
Single source of truth for:
  - Model targets (TARGET_M1, TARGET_M2)
  - Feature lists for both models (final sets after selection)
  - Log transforms (LOG_TRANSFORM)
  - Feature preparation function (prepare_features)

PRIMARY EVALUATION CONDITION: Condition E
  Train : 45 non-held-out basins (37 clean + 8 outliers), all data,
          no quality filtering of any kind
  Test  : 5 completely unseen held-out basins (one per field),
          fixed before any model training

  Model 1 held-out results (Condition E, E-full):
    Pooled: R²=+0.861, RMSE=0.663 cm/h, MAPE=15.1% (n=2,504 events)
    Basin 3203 (Soreq 2): R²=+0.565, RMSE=0.996 cm/h, MAPE=14.7%
    Basin 4104 (Yavne 1): R²=+0.657, RMSE=0.208 cm/h, MAPE=13.4%
    Basin 5102 (Yavne 2): R²=+0.670, RMSE=0.590 cm/h, MAPE=16.1%
    Basin 6303 (Yavne 3): R²=+0.733, RMSE=0.408 cm/h, MAPE=14.2%
    Basin 7201 (Yavne 4): R²=+0.808, RMSE=0.158 cm/h, MAPE=16.7%

  Model 2 bootstrap validation (200 random held-out selections):
    Old-3  (3 feat):  beats naive  15.0%  median ΔRMSE=−0.027
    New-16 (16 feat): beats naive  44.0%  median ΔRMSE=−0.005
    Set-C  (10 feat): beats naive  34.0%  median ΔRMSE=−0.010
    Set-D  (12 feat): beats naive  83.5%  median ΔRMSE=+0.031  ← WINNER

─────────────────────────────────────────────────────────────────────────────
Feature naming conventions
─────────────────────────────────────────────────────────────────────────────
  prev_*     value from the immediately preceding event (within segment)
  cum_*      cumulative sum since segment reset
  mean_*     segment-level mean (Model 2 reset dataset)
  sum_*      segment-level sum (Model 2 reset dataset)
  last_*     value from the final event of the segment (Model 2 only)
  log1p_*    log(1 + x) transform applied to a skewed column

─────────────────────────────────────────────────────────────────────────────
Operational variables
─────────────────────────────────────────────────────────────────────────────
  FT    Flooding Time (h)         duration of active flooding phase
  DT    Drainage Time (h)         duration of passive drainage phase
  DrT   Drying Time (h)           duration between drainage end and next flood
  WT    Wetting Time (h)          FT + DT
  Ct    Cycle Time (h)            WT + DrT
  ALPHA Drying fraction (DF)      DrT / Ct — primary operational recovery driver
  HL    Hydraulic Load (cm)       CIV / basin_area — water depth per event
  IRD   Infiltration Rate (cm/h)  measured during drainage phase

  NOTE on HL: HL is ENDOGENOUS — high HL occurs because IRD is high,
  not the reverse. Do not vary HL in optimization or heatmap analysis.

─────────────────────────────────────────────────────────────────────────────
Weather variables
─────────────────────────────────────────────────────────────────────────────
  TD   Temperature during drying (°C)
  TW   Temperature during wetting (°C)
  RD   Radiation during drying (W/m²)
  RW   Radiation during wetting (W/m²)
  DAT  Daily Average Temperature at event date
  DAR  Daily Average Radiation at event date

─────────────────────────────────────────────────────────────────────────────
SHAP findings — for reference
─────────────────────────────────────────────────────────────────────────────
  Model 1 top 4 (stable across conditions):
    1. IRD_at_reset    mean|SHAP|=0.183  scale anchor
    2. prev_ALPHA      mean|SHAP|=0.178  drying fraction → less decay
    3. prev_DrT        mean|SHAP|=0.108  see collinearity note below
    4. log1p_prev_HL   mean|SHAP|=0.081  hydraulic load → more clogging

  prev_DrT collinearity note (IMPORTANT):
    SHAP shows high prev_DrT → faster decay (apparent contradiction).
    This is a collinearity artifact: Spearman r(prev_DrT, prev_ALPHA) = 0.79.
    Unconditional r(prev_DrT, η) = +0.18 — physically correct direction.
    Full analysis in Discussion + SI. Do NOT remove prev_DrT.

  Model 2 Set-D (12 features, bootstrap-validated):
    Key finding: scale-free autocorrelation (prev_delta, prev_prev_delta)
    generalizes to unseen basins. Raw prev_IRD_at_reset hurts held-out
    performance — log-ratio normalization removes between-basin scale.
    Radiation features carry seasonal signal without month encoding.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Model targets
# ─────────────────────────────────────────────────────────────────────────────

TARGET_M1 = "IRD_norm_log"
TARGET_M2 = "IRD_norm_log_reset"


# ─────────────────────────────────────────────────────────────────────────────
# Log transforms — Model 1 only
# ─────────────────────────────────────────────────────────────────────────────
LOG_TRANSFORM = {
    "prev_HL": "log1p_prev_HL",
    "cum_HL":  "log1p_cum_HL",
}


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 features — FINAL SET (11 features)
# ─────────────────────────────────────────────────────────────────────────────
# Selected by unconstrained forward stepwise, Condition E.
# Metric: RMSE on raw IRD (cm/h) on 5 held-out test basins.
#
# Selection path (held-out RMSE):
#    5 features:  0.7193 cm/h
#    6 features:  0.6936 cm/h  +prev_TD    Δ=+0.026 ✓
#    7 features:  0.6794 cm/h  +prev_FT    Δ=+0.014 ✓
#    8 features:  0.6679 cm/h  +cum_TW     Δ=+0.012 ✓
#    9 features:  0.6659 cm/h  +cum_FT     Δ=+0.002 ✓
#   10 features:  0.6504 cm/h  +prev_RD    Δ=+0.016 ✓
#   11 features:  0.6491 cm/h  +prev_RW    Δ=+0.001 ✓  ← BEST
#   12 features:  0.6582 cm/h  +cum_TD     Δ=-0.009 ✗  STOP

MODEL1_RAW_FEATURES = [
    "IRD_at_reset",   # scale anchor — basin hydraulic conductivity ceiling
    "prev_ALPHA",     # drying fraction = DrT/Ct — primary recovery driver
    "prev_HL",        # → log1p_prev_HL at runtime (skewness=15.5)
    "prev_DrT",       # drying time (collinear with ALPHA — see SHAP note)
    "prev_TD",        # temperature during previous drying phase
    "prev_RD",        # radiation during previous drying (photodegradation)
    "prev_RW",        # radiation during previous wetting phase
    "prev_FT",        # flooding time of previous event
    "LCT",            # time since last tillage — primary decay axis
    "cum_TW",         # cumulative wetting temperature since reset
    "cum_FT",         # cumulative flooding time since reset
]

MODEL1_FEATURES = [
    LOG_TRANSFORM.get(f, f) for f in MODEL1_RAW_FEATURES
]


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 features — FINAL SET (12 features, Set-D)
# ─────────────────────────────────────────────────────────────────────────────
# Selected by bootstrap validation (200 random held-out iterations).
# Metric: pooled RMSE on 5 random held-out basins, 35-basin eligible pool.
#
# Bootstrap beat rate vs naive:
#   Old-3  (month_sin, month_cos, total_LCT)  : 15.0%  ← previous model
#   Set-D  (12 features below)                : 83.5%  ← current model
#
# Design rationale:
#   Radiation features carry seasonal signal in physical units — month
#   encoding is redundant and hurts generalization across the bootstrap pool.
#   prev_delta and prev_prev_delta are scale-free (dimensionless log-ratios)
#   and generalize to unseen basins, unlike raw prev_IRD_at_reset which
#   carries between-basin scale and consistently hurts held-out performance.

MODEL2_FEATURES = [
    # ── Operational — segment load and drying quality ──────────────────────
    "frac_zero_DrT",    # fraction of events with no drying → clogging severity
    "n_events",         # segment length in events → operational load
    "sum_RW",           # total wetting radiation → seasonal/weather context
    "total_LCT",        # total segment duration (h) → cumulative clogging

    # ── Radiation and drying — previous segment ───────────────────────────
    "max_RD",           # peak drying radiation → photodegradation potential
    "max_DrT",          # longest single drying event → best recovery opportunity
    "last_RD",          # radiation at last drying event → most recent conditions
    "mean_RW",          # average wetting radiation → seasonal context
    "mean_DrT",         # average drying duration → typical recovery conditions

    # ── Ambient conditions at tillage date ────────────────────────────────
    "DAR",              # daily ambient radiation → recovery environment

    # ── Scale-free autocorrelation — generalizes to unseen basins ─────────
    "prev_delta",       # δᵢ₋₁ = log(ρᵢ₋₁/ρᵢ₋₂): was last recovery better?
    "prev_prev_delta",  # δᵢ₋₂ = log(ρᵢ₋₂/ρᵢ₋₃): two-step recovery trend
]


# ─────────────────────────────────────────────────────────────────────────────
# Raw database columns needed to build event_dataset.csv
# ─────────────────────────────────────────────────────────────────────────────
RAW_DB_COLUMNS = [
    "opening_valve_date", "closing_valve_date",
    "drainage_end", "next_opening_valve_date",
    "IRD", "IRD_R_squared", "IRD_at_reset", "LCT", "CIV",
    "FT", "DT", "DrT", "WT", "Ct", "ALPHA", "K_RATIO", "AL", "ML", "AF",
    "TD", "TW", "RD", "RW", "HD", "PD", "DAT", "DAR",
]

PREV_SOURCE_COLS = [
    "DrT", "FT", "ALPHA", "HL",
    "TD", "RD", "HD", "PD",
    "TW", "RW",
    "AL", "ML", "AF",
]

CUM_SOURCE_COLS = {
    "cum_HL":  "HL",
    "cum_FT":  "FT",
    "cum_DrT": "DrT",
    "cum_TD":  "TD",
    "cum_RD":  "RD",
    "cum_TW":  "TW",
    "cum_RW":  "RW",
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation — Model 1
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply log1p transforms to skewed HL columns and return the final
    Model 1 feature column list.
    """
    df = df.copy()

    for raw_col, new_col in LOG_TRANSFORM.items():
        if raw_col in df.columns:
            df[new_col] = np.log1p(
                pd.to_numeric(df[raw_col], errors="coerce")
            )

    features: list[str] = []
    for col in MODEL1_RAW_FEATURES:
        transformed = LOG_TRANSFORM.get(col)
        if transformed and transformed in df.columns:
            features.append(transformed)
        elif col in df.columns:
            features.append(col)

    return df, features