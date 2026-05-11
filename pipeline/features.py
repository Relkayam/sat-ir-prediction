"""
pipeline/features.py — Feature definitions for Model 1 and Model 2
====================================================================
Single source of truth for:
  - Model targets (TARGET_M1, TARGET_M2)
  - Feature lists for both models (final sets after forward stepwise selection)
  - Log transforms (LOG_TRANSFORM)
  - Feature preparation function (prepare_features)

PRIMARY EVALUATION CONDITION: Condition E
  - Model 1: 45 non-held-out basins (including outliers), all segments,
             tested on 5 completely unseen held-out basins
             Pooled: R²=+0.898, RMSE=0.607 cm/h, MAPE=13.1% (n=4,386)
  - Model 2: 45 non-held-out basins (including outliers), all resets,
             tested on 5 completely unseen held-out basins
             Pooled: R²=+0.884, RMSE=0.983 cm/h, MAPE=14.8% (n=454)

Import this file in:
  build_dataset.py, build_reset_dataset.py,
  model1_decay.py, model2_reset.py,
  model_comparison.py, model2_comparison.py,
  all figure scripts (fig2–fig6)

─────────────────────────────────────────────────────────────────────────────
Feature naming conventions
─────────────────────────────────────────────────────────────────────────────
  prev_*     value from the immediately preceding event (within segment)
  cum_*      cumulative sum since segment reset (up to but not including
             the current event)
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
  ALPHA Drying fraction           DrT / Ct — primary operational recovery driver
  HL    Hydraulic Load (cm)       CIV / basin_area — water depth per event
  IRD   Infiltration Rate (cm/h)  measured during drainage phase

  NOTE on HL: HL is ENDOGENOUS — high HL occurs because IRD is high,
  not the reverse. Do not vary HL in optimization or heatmap analysis.

─────────────────────────────────────────────────────────────────────────────
Weather variables
─────────────────────────────────────────────────────────────────────────────
  TD   Temperature during drying (°C)
  TW   Temperature during wetting (°C)
  RD   Radiation during drying (W/m²)     primary photodegradation signal
  RW   Radiation during wetting (W/m²)
  DAT  Daily Average Temperature at event date
  DAR  Daily Average Radiation at event date — key Model 2 driver

─────────────────────────────────────────────────────────────────────────────
SHAP findings (Condition E) — for reference
─────────────────────────────────────────────────────────────────────────────
  Model 1 top 4 (stable across conditions A, D, E):
    1. IRD_at_reset    mean|SHAP|=0.183  scale anchor
    2. prev_ALPHA      mean|SHAP|=0.178  drying fraction → less decay
    3. prev_DrT        mean|SHAP|=0.108  see collinearity note below
    4. log1p_prev_HL   mean|SHAP|=0.081  hydraulic load → more clogging

  prev_DrT collinearity note (IMPORTANT):
    SHAP shows high prev_DrT → faster decay (apparent contradiction).
    This is a collinearity artifact: Spearman r(prev_DrT, prev_ALPHA) = 0.79.
    After controlling for ALPHA, residual DrT variation correlates with
    long-cycle events that are operationally difficult.
    The unconditional correlation is physically correct:
      Spearman r(prev_DrT, η) = +0.18 — longer drying → less decay.
    Full collinearity analysis in Discussion + SI of paper.
    Do NOT remove prev_DrT from the feature set.

  Model 2 top 2 (stable across all conditions):
    1. prev_IRD_at_reset    autocorrelation anchor
    2. DAR                  daily radiation at tillage → till when sunny
"""

# ─────────────────────────────────────────────────────────────────────────────
# Model targets
# ─────────────────────────────────────────────────────────────────────────────

# Model 1: within-segment IRD decay
# η(t) = log(IRD(t) / IRD_at_reset)
# Starts at 0 at segment reset, goes negative as clogging progresses.
# Back-transform: IRD(t) = IRD_at_reset × exp(η(t))
TARGET_M1 = "IRD_norm_log"

# Model 2: post-tillage recovery
# δᵢ = log(IRD_at_reset[i] / IRD_at_reset[i-1])
# = 0  : recovery matches previous reset exactly
# > 0  : better recovery than last time
# < 0  : worse recovery than last time
# Back-transform: IRD_at_reset[i] = IRD_at_reset[i-1] × exp(δᵢ)
# First reset per basin: no previous reset → NaN → excluded from training.
TARGET_M2 = "IRD_norm_log_reset"

# ─────────────────────────────────────────────────────────────────────────────
# Log transforms
# ─────────────────────────────────────────────────────────────────────────────
# Maps raw column → transformed column name.
# prepare_features() computes transformed columns and substitutes them
# in the feature list. Raw columns are NOT removed from the DataFrame.
#
# Why log1p for HL?
#   prev_HL: skewness=15.5, kurtosis=574 — extreme right tail
#   log1p(x) = log(1+x): safe for near-zero values, compresses tail,
#   dramatically improves gradient boosting split quality.
LOG_TRANSFORM = {
    "prev_HL": "log1p_prev_HL",
    "cum_HL":  "log1p_cum_HL",
}


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 features — FINAL SET (11 features after log transforms)
# ─────────────────────────────────────────────────────────────────────────────
#
# Selected by unconstrained forward stepwise selection on Condition E.
# Metric: RMSE on raw IRD (cm/h) on 5 held-out test basins.
# See: outputs/tables/feature_selection_unconstrained.xlsx
#      analysis/feature_selection_unconstrained.py
#
# Selection path:
#    5 features (base):  0.7193 cm/h  mandatory base
#    6 features:         0.6936 cm/h  +prev_TD      Δ=+0.026 ✓ strong
#    7 features:         0.6794 cm/h  +prev_FT      Δ=+0.014 ✓ strong
#    8 features:         0.6679 cm/h  +cum_TW       Δ=+0.012 ✓ strong
#    9 features:         0.6659 cm/h  +cum_FT       Δ=+0.002 ✓ marginal
#   10 features:         0.6504 cm/h  +prev_RD      Δ=+0.016 ✓ strong
#   11 features:         0.6491 cm/h  +prev_RW      Δ=+0.001 ✓ marginal
#   12 features:         0.6582 cm/h  +cum_TD       Δ=-0.009 ✗ STOP → elbow
#
# Full 21-feature set RMSE = 0.7117 cm/h — worse than 11-feature set.
# Confirms overfitting in larger model; reduced set generalises better.

MODEL1_RAW_FEATURES = [
    # ── Mandatory base — top SHAP across all conditions ────────────────────
    "IRD_at_reset",   # scale anchor — basin hydraulic conductivity ceiling
    "prev_ALPHA",     # drying fraction = DrT/Ct — primary recovery driver
    "prev_HL",        # → log1p_prev_HL at runtime (skewness=15.5)
    "prev_DrT",       # drying time — absolute recovery duration
                      # NOTE: SHAP direction appears counterintuitive due to
                      # collinearity with prev_ALPHA (r=0.79). Unconditional
                      # r(prev_DrT, η)=+0.18 confirms correct physical direction.
                      # See Discussion + SI for full collinearity analysis.
    "LCT",            # time since reset — primary decay axis

    # ── High-value additions (Δ > 0.010 cm/h each) ────────────────────────
    "prev_TD",        # temperature during previous drying phase
    "prev_FT",        # flooding time of previous event
    "cum_TW",         # cumulative wetting temperature since reset (seasonal)

    # ── Marginal but retained ──────────────────────────────────────────────
    "cum_FT",         # cumulative flooding time since reset
    "prev_RD",        # radiation during previous drying (photodegradation)
    "prev_RW",        # radiation during previous wetting phase
]

# Final feature list passed to StandardScaler and LightGBM
# (log1p versions substituted for raw HL columns)
MODEL1_FEATURES = [
    LOG_TRANSFORM.get(f, f) for f in MODEL1_RAW_FEATURES
]


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 features — FINAL SET (11 features)
# ─────────────────────────────────────────────────────────────────────────────
#
# Selected by unconstrained forward stepwise selection on chrono split.
# Metric: RMSE on raw IRD (cm/h) after back-transform on chrono test split.
# See: outputs/tables/feature_selection_m2.xlsx
#      analysis/feature_selection_m2.py
#
# Selection path:
#   Naive baseline:  0.8202 cm/h
#    1 feature:      0.7730  +month_sin               Δ=+0.047 ✓ huge
#    2 features:     0.7557  +prev_IRD_at_reset        Δ=+0.017 ✓ strong
#    3 features:     0.7450  +mean_ALPHA               Δ=+0.011 ✓ strong
#    4 features:     0.7259  +total_LCT                Δ=+0.019 ✓ strongest
#    5 features:     0.7195  +DAR                      Δ=+0.006 ✓ good
#    6 features:     0.7170  +prev_prev_IRD_at_reset   Δ=+0.002 ✓ marginal
#    7 features:     0.7096  +sum_DrT                  Δ=+0.008 ✓ good
#    8 features:     0.7108  +last_RD                  Δ=-0.001 ✗ worsened
#    9 features:     0.7104  +month_cos                Δ=+0.001 ✓ trivial
#   10 features:     0.7076  +sum_FT                   Δ=+0.003 ✓ marginal
#   11 features:     0.7017  +last_DrT                 Δ=+0.006 ✓ BEST → STOP
#   12+ features: all worsen RMSE — clear elbow
#
# Full model RMSE vs 11-feature: 11-feature better by 0.054 cm/h.

MODEL2_FEATURES = [
    # ── Seasonality ────────────────────────────────────────────────────────
    "month_sin",               # sin(2π(month-4)/12) — peak July = +1.0
    "month_cos",               # orthogonal seasonal component

    # ── Cross-segment autocorrelation ──────────────────────────────────────
    "prev_IRD_at_reset",       # previous reset level — primary autocorrelation
    "prev_prev_IRD_at_reset",  # two-step history — medium-term trend

    # ── Segment operational summary ────────────────────────────────────────
    "mean_ALPHA",              # average drying fraction over segment
    "total_LCT",               # total flooding duration of segment (h)
    "sum_DrT",                 # cumulative drying time of segment (h)
    "sum_FT",                  # cumulative flooding time of segment (h)

    # ── Last-event features (most proximal signals before tillage) ─────────
    "last_DrT",                # final drying duration before tillage
    "last_RD",                 # radiation in final drying event

    # ── Ambient conditions at reset ────────────────────────────────────────
    "DAR",                     # daily ambient radiation at tillage date
                               # OPERATIONALLY ACTIONABLE: till when sunny
]

# ─────────────────────────────────────────────────────────────────────────────
# Model 2 feature sub-lists
# ─────────────────────────────────────────────────────────────────────────────
# These sub-lists are used by build_reset_dataset.py for aggregation logic.
# They partition MODEL2_FEATURES by data source.

# Features aggregated from segment i-1 events
MODEL2_SEGMENT_FEATURES = [
    "mean_ALPHA",   # mean drying fraction
    "total_LCT",    # total flooding duration
    "sum_DrT",      # cumulative drying time
    "sum_FT",       # cumulative flooding time
    "last_DrT",     # final drying duration before tillage
    "last_RD",      # radiation in final drying event
]

# Cross-segment history features (from event_dataset.csv, pre-quality-filter)
# prev_IRD_at_reset is both a feature AND the back-transform denominator
MODEL2_HISTORY_FEATURES = [
    "prev_IRD_at_reset",       # previous reset level — back-transform denominator
    "prev_prev_IRD_at_reset",  # two-step history
]

# Seasonality and ambient conditions at reset date
MODEL2_SEASON_FEATURES = [
    "month_sin",   # sin encoding — peak July
    "month_cos",   # cos encoding — orthogonal
    "DAR",         # daily ambient radiation at reset date
]

# Validation: MODEL2_FEATURES must equal the union of all sub-lists
_m2_check = set(MODEL2_SEGMENT_FEATURES + MODEL2_HISTORY_FEATURES + MODEL2_SEASON_FEATURES)
assert _m2_check == set(MODEL2_FEATURES), (
    f"MODEL2 sub-lists do not match MODEL2_FEATURES.\n"
    f"  In sub-lists but not MODEL2_FEATURES: {_m2_check - set(MODEL2_FEATURES)}\n"
    f"  In MODEL2_FEATURES but not sub-lists: {set(MODEL2_FEATURES) - _m2_check}"
)

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

# Columns shifted within each segment to produce prev_* features
PREV_SOURCE_COLS = [
    "DrT", "FT", "ALPHA", "HL",
    "TD", "RD", "HD", "PD",
    "TW", "RW",
    "AL", "ML", "AF",
]

# Columns cumulated within each segment to produce cum_* features
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

    Parameters
    ----------
    df : DataFrame containing at minimum the columns in MODEL1_RAW_FEATURES.
         basin_number is NOT required — this function does not filter by basin.

    Returns
    -------
    df       : copy of input with log1p columns added
    features : list of column names ready to pass to the model
               (log1p versions substituted for raw HL columns)

    Notes
    -----
    - basin_number column is dropped by this function if present in the
      returned feature list (it never is — it is not in MODEL1_RAW_FEATURES).
    - If you need basin_number after calling prepare_features(), preserve it
      before the call:
          basin_col = df["basin_number"].copy()
          df, feats = prepare_features(df)
          df["basin_number"] = basin_col

    Example
    -------
    >>> df_ready, feat_cols = prepare_features(df)
    >>> model.fit(df_ready[feat_cols], df_ready[TARGET_M1])
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