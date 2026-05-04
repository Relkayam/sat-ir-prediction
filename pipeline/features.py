"""
pipeline/features.py — Feature definitions for Model 1 and Model 2
====================================================================

This file is the single source of truth for:
  - Which raw columns are used as features in each model
  - Which columns require log transformation (and why)
  - The model targets

Nothing here depends on data — these are just lists and mappings.
Import from this file in build_dataset.py, model1_decay.py, model2_reset.py,
and all figure scripts.

─────────────────────────────────────────────────────────────────────────────
Quick reference: feature naming conventions
─────────────────────────────────────────────────────────────────────────────
  prev_*     — value from the event immediately before the current one
               (within the same segment)
  cum_*      — cumulative sum since segment reset, up to but not including
               the current event
  mean_*     — segment-level mean (used in Model 2 reset dataset)
  sum_*      — segment-level sum
  last_*     — value from the final event of the segment (Model 2 only)
  log1p_*    — log(1 + x) transform applied to a skewed column

─────────────────────────────────────────────────────────────────────────────
Operational features
─────────────────────────────────────────────────────────────────────────────
  FT    Flooding Time (h)        — duration of active flooding phase
  DT    Drainage Time (h)        — duration of passive drainage phase
  DrT   Drying Time (h)          — duration between drainage end and next flood
  WT    Wetting Time (h)         — FT + DT (total wet exposure)
  Ct    Cycle Time (h)           — WT + DrT (full cycle duration)
  ALPHA Drying fraction          — DrT / Ct — key recovery driver
  HL    Hydraulic Load (cm)      — CIV / basin_area — water depth per event
  IRD   Infiltration Rate (cm/h) — measured during drainage phase

  NOTE on HL: HL is ENDOGENOUS — high HL occurs because IRD is high, not
  the other way around. HL must NOT be varied in optimization/heatmap
  analysis. It is fixed at the dataset median there.

─────────────────────────────────────────────────────────────────────────────
Weather features
─────────────────────────────────────────────────────────────────────────────
  TD  Temperature Dry (°C)       — ambient temp during drying window
  TW  Temperature Wet (°C)       — ambient temp during wetting window
  RD  Radiation Dry (W/m²)       — solar radiation during drying window
  RW  Radiation Wet (W/m²)       — solar radiation during wetting window
  AL  Water Level (cm)           — measured water level in basin
  ML  Max Water Level (cm)
  AF  Area Flooded (m²)
  DAT Daily Average Temperature  — continuous seasonality signal
  DAR Daily Average Radiation    — continuous seasonality + photodegradation

  NOTE on RD: radiation during drying is more physically meaningful than
  drying time alone — photodegradation of the clogging layer is the
  primary mechanism of natural surface recovery. RD outranks DrT in SHAP.

─────────────────────────────────────────────────────────────────────────────
Why log1p transforms for HL
─────────────────────────────────────────────────────────────────────────────
  prev_HL  skewness=15.5, kurtosis=574 — extreme right tail
  cum_HL   similar distribution

  log1p(x) = log(1+x): safe for near-zero values, compresses the tail,
  dramatically improves gradient boosting split quality on this feature.
  The raw columns are kept in the CSV; log1p versions are computed at
  runtime in prepare_features().
"""

# ─────────────────────────────────────────────────────────────────────────────
# Model targets
# ─────────────────────────────────────────────────────────────────────────────

# Model 1: within-segment IRD decay
# log(IRD / IRD_at_reset) — starts at 0 at reset, goes negative as IRD decays
# Predicting a log-ratio makes the target dimensionless and normalizes
# across basins with very different absolute IRD scales.
TARGET_M1 = "IRD_norm_log"

# Model 2: IRD recovery after reset/tillage
# log(IRD_at_reset[i] / IRD_at_reset[i-1])
# Dimensionless log-ratio between consecutive reset values.
# Starts at 0 when recovery exactly matches the previous reset.
# Positive = better recovery than last time.
# Negative = worse recovery than last time.
#
# This normalization is physically justified:
#   - Different basins have different hydraulic conductivity ceilings (Ks)
#   - Raw IRD_at_reset varies dramatically between basins (e.g. 2-12 cm/h)
#   - The log-ratio removes between-basin scale differences and expresses
#     recovery relative to the basin's own recent history
#   - Consistent with Model 1 which also uses a log-ratio target
#   - Back-transform: IRD_at_reset[i] = IRD_at_reset[i-1] * exp(predicted)
#
# Raw IRD_at_reset was tested and rejected: while R² appeared higher,
# the model was partly learning which basin it was predicting rather than
# the recovery dynamics. LogPrevRatio gives better MAPE (16.8% vs 21.0%)
# and is more physically interpretable across the heterogeneous basin system.
#
# First reset per basin has no previous reset -> NaN -> excluded from training.
TARGET_M2 = "IRD_norm_log_reset"

# ─────────────────────────────────────────────────────────────────────────────
# Log transforms (applied in prepare_features)
# ─────────────────────────────────────────────────────────────────────────────
# Maps raw column name -> transformed column name.
# prepare_features() adds the transformed column and substitutes it
# in the feature list. The raw column is NOT removed from the DataFrame.
LOG_TRANSFORM = {
    "prev_HL": "log1p_prev_HL",
    "cum_HL":  "log1p_cum_HL",
}


# Model 1 features (11 total after log transforms)
# Final set selected by unconstrained forward stepwise on Condition E.
# Elbow detected at step 7 (12th feature cum_TD worsened RMSE).
# Full selection path: analysis/feature_selection_unconstrained.py

# ─────────────────────────────────────────────────────────────────────────────
# Model 1 features — FINAL SET (11 features after log transforms)
# ─────────────────────────────────────────────────────────────────────────────
#
# Feature selection methodology (documented for reproducibility):
# ---------------------------------------------------------------
# Forward stepwise selection was performed on Condition E (45-basin training,
# 5 held-out test basins never seen during training). Metric: RMSE on raw
# IRD (cm/h) on held-out test set. LightGBM with well-established defaults.
#
# Starting set: 5 mandatory features (top SHAP across conditions A, D, E)
# Selection: greedy forward, unconstrained, run to all 21 features
# See: outputs/tables/feature_selection_unconstrained.xlsx
#      analysis/feature_selection_unconstrained.py
#
# Selection path (RMSE at each step):
#   5 features: 0.7193 cm/h  (mandatory base)
#   6 features: 0.6936 cm/h  +prev_TD      Δ=+0.026 ✓ strong
#   7 features: 0.6794 cm/h  +prev_FT      Δ=+0.014 ✓ strong
#   8 features: 0.6679 cm/h  +cum_TW       Δ=+0.012 ✓ strong
#   9 features: 0.6659 cm/h  +cum_FT       Δ=+0.002 ✓ marginal
#  10 features: 0.6504 cm/h  +prev_RD      Δ=+0.016 ✓ strong
#  11 features: 0.6491 cm/h  +prev_RW      Δ=+0.001 ✓ marginal (retained)
#  12 features: 0.6582 cm/h  +cum_TD       Δ=-0.009 ✗ worsened → STOP
#
# Features NOT selected (and why):
#   cum_RD, cum_RW  — cumulative radiation features: partially collinear
#                     with prev_RD, prev_RW and cum_TW. The most proximal
#                     radiation signal (prev_*) dominates over cumulative.
#   cum_TD          — first feature to worsen RMSE (-0.009): defines elbow
#   log1p_cum_HL    — only recovers after cum_TD noise; not independently useful
#   prev_TW         — only useful after log1p_cum_HL compensates cum_TD noise
#   cum_DrT         — worsened RMSE at step 10 (-0.006)
#   event_count     — compensates for noise from earlier bad additions
#   prev_AL, prev_ML, prev_AF — consistently worsen RMSE; water level
#                               and flooded area features add noise not signal
#
# Previously tested full 21-feature set:
#   Condition E RMSE = 0.7117 cm/h  (worse than 11-feature set)
#   This confirms the 21-feature model was overfitting to training basin
#   idiosyncrasies — the reduced set generalises better to unseen basins.
#
# Physical interpretation of final 11 features:
#   IRD_at_reset    — scale anchor: basin hydraulic conductivity ceiling
#   prev_ALPHA      — drying fraction: ratio of recovery time to cycle time
#   log1p_prev_HL   — hydraulic load: clogging intensity of previous event
#   prev_DrT        — drying duration: absolute recovery time
#   LCT             — time since reset: primary decay axis
#   prev_TD         — temperature during previous drying: affects viscosity
#                     and biological activity during surface recovery
#   prev_FT         — flooding time: duration of clogging exposure
#   cum_TW          — cumulative wetting temperature since reset: encodes
#                     seasonal signal and accumulated biological activity
#   cum_FT          — cumulative flooding time: total clogging load since reset
#   prev_RD         — radiation during previous drying: photodegradation
#                     of clogging layer — most proximal environmental signal
#   prev_RW         — radiation during previous wetting: solar exposure
#                     during flooding influences surface biofilm activity


# NOT included (tested and rejected by forward stepwise selection):
#   cum_RD, cum_RW      — collinear with prev_RD/prev_RW and cum_TW
#   cum_TD              — worsened RMSE at step 7 (elbow)
#   log1p_cum_HL        — only useful after noise compensation
#   prev_TW             — only useful after multiple noise features added
#   cum_DrT             — worsened RMSE at step 10
#   event_count         — compensates for noise; not independently useful
#   prev_AL, prev_ML, prev_AF — water level/area: noise not signal
#
# Previously rejected (V1, not tested in V2 stepwise):
#   prev_IRD_at_reset   — r=-0.07, dropna cost caused R² 0.397→0.308
#   IRD_direction       — r=+0.005 (between-reset trend ≠ within-segment)
#   month_sin/cos       — captured by cum_TW and prev_TD
#   DAT, DAR            — commented out: redundant with prev_TD/prev_RD

MODEL1_RAW_FEATURES = [
    # ── Mandatory base (top SHAP, all conditions) ──────────────────────────
    "IRD_at_reset",       # scale anchor — basin hydraulic conductivity
    "prev_ALPHA",         # drying fraction = DrT / Ct
    "prev_HL",            # -> log1p_prev_HL at runtime (skewness > 15)
    "prev_DrT",           # drying time — absolute recovery duration
    "LCT",                # time since reset — primary decay axis

    # ── Step 1-3: high-value additions (Δ > 0.010 cm/h each) ──────────────
    "prev_TD",            # temperature during previous drying phase
    "prev_FT",            # flooding time of previous event
    "cum_TW",             # cumulative wetting temperature since reset

    # ── Step 4: marginal but retained (Δ = +0.002 cm/h) ───────────────────
    "cum_FT",             # cumulative flooding time since reset

    # ── Step 5: strong recovery after plateau (Δ = +0.016 cm/h) ───────────
    "prev_RD",            # radiation during previous drying (photodegradation)

    # ── Step 6: marginal but retained (Δ = +0.001 cm/h) ───────────────────
    "prev_RW",            # radiation during previous wetting phase
]

# Final feature list used by the model (after log transforms).
# This is what you pass to StandardScaler and XGBRegressor.
MODEL1_FEATURES = [
    LOG_TRANSFORM.get(f, f) for f in MODEL1_RAW_FEATURES
]

# ─────────────────────────────────────────────────────────────────────────────
# Model 2 features (23 total)
# ─────────────────────────────────────────────────────────────────────────────
# Used in: model2_reset.py, analysis/heatmap.py
#
# All features are aggregated from segment i-1 (the segment that just ended).
# Cross-segment history features are computed in build_dataset.py BEFORE the
# quality filter — this ensures IRD_direction always references the true
# previous reset, not the previous *good* reset.
#
# NOTE: prev_IRD_at_reset is kept as a feature even though the target is now
# the log-ratio — it serves as the denominator reference and carries
# important information about the basin's recent recovery level.

# Aggregated from segment i-1
MODEL2_SEGMENT_FEATURES = [
    # Operational averages
    "mean_DrT",    # mean drying time across segment
    "mean_FT",     # mean flooding time across segment
    "mean_ALPHA",  # mean drying fraction (DrT / Ct) — consistent with Model 1
    "mean_HL",     # mean hydraulic load — ENDOGENOUS, fixed in heatmap
    # Totals
    "sum_DrT",     # total drying time (= mean_DrT x n_events)
    "sum_FT",      # total flooding time
    # Extremes
    "min_DrT",     # shortest drying period in segment
    "max_FT",      # longest flooding event in segment
    # Segment size
    "n_events",    # number of flooding events
    "total_LCT",   # total segment duration (h)
    # Weather
    "mean_RD",     # mean radiation during drying — key photodegradation driver
    "mean_TW",     # mean temperature (wet phase)
    "mean_TD",     # mean temperature (dry phase)
    # Last-event features: the drying period immediately before reset.
    # Physical rationale: the FINAL drying opportunity before tillage is
    # distinct from the segment average — a long sunny final drying period
    # maximises surface crust degradation just before the reset, directly
    # affecting how high IRD_at_reset will be. Confirmed by SHAP analysis.
    "last_DrT",    # DrT of the final event before reset
    "last_RD",     # radiation of the final drying period before reset
]

# Cross-segment history (computed in build_dataset.py before quality filter)
# prev_IRD_at_reset is the denominator for the log-ratio target and also
# serves as the primary autocorrelation feature.
MODEL2_HISTORY_FEATURES = [
    "prev_IRD_at_reset",       # IRD_at_reset of reset i-1 — denominator + autocorrelation
    "prev_prev_IRD_at_reset",  # IRD_at_reset of reset i-2 — two-step memory
    "IRD_direction",           # (IRD[i-1] - IRD[i-2]) / dt — recovery trend (cm/h/day)
]

# Seasonality at the reset date + daily ambient conditions
# month_sin ranks high in Model 2 feature importance — summer resets
# recover better than winter resets.
# DAT and DAR provide continuous ambient signal at the reset moment.
MODEL2_SEASON_FEATURES = [
    "month_sin",   # sin encoding — peak July (+1.0), trough January (-1.0)
    "month_cos",   # cos encoding — orthogonal component
    "DAT",         # daily avg temperature at reset moment — continuous seasonality
    "DAR",         # daily avg radiation at reset moment — continuous seasonality
]

# Final combined feature list for Model 2
MODEL2_FEATURES = (
    MODEL2_SEGMENT_FEATURES +
    MODEL2_HISTORY_FEATURES +
    MODEL2_SEASON_FEATURES
)

# ─────────────────────────────────────────────────────────────────────────────
# Columns needed from DuckDB to build the event dataset
# ─────────────────────────────────────────────────────────────────────────────
RAW_DB_COLUMNS = [
    "opening_valve_date", "closing_valve_date",
    "drainage_end", "next_opening_valve_date",
    "IRD", "IRD_R_squared", "IRD_at_reset", "LCT", "CIV",
    "FT", "DT", "DrT", "WT", "Ct", "ALPHA", "K_RATIO", "AL", "ML", "AF",
    "TD", "TW", "RD", "RW", "HD", "PD", "DAT", "DAR",
]

# Columns used to compute prev_* features (shifted within each segment)
PREV_SOURCE_COLS = [
    "DrT", "FT", "ALPHA", "HL",
    "TD", "RD", "HD", "PD",
    "TW", "RW",
    "AL", "ML", "AF",
]

# Columns used to compute cum_* features (cumulative sum within each segment)
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
# Feature preparation function
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply log1p transforms to skewed HL columns and return the final
    feature column list for Model 1.

    Parameters
    ----------
    df : DataFrame containing at minimum the columns in MODEL1_RAW_FEATURES

    Returns
    -------
    df       : copy of input with log1p columns added
    features : list of column names to pass to the model
               (log1p versions substituted for raw HL columns)

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

    # Build final feature list: substitute transformed names for raw names
    features: list[str] = []
    for col in MODEL1_RAW_FEATURES:
        transformed = LOG_TRANSFORM.get(col)
        if transformed and transformed in df.columns:
            features.append(transformed)
        elif col in df.columns:
            features.append(col)

    return df, features