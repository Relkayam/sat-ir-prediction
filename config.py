"""
config.py — Central configuration for sat-ird-prediction-v2
=============================================================
Primary evaluation condition: Condition E (all-data held-out test)
  - 45 non-held-out basins (including outliers) for training
  - All segments — no good-segment filter
  - 5 held-out basins (one per field) for test — never seen during training

This is the most general and honest evaluation condition.
Condition A (clean basins, good segments only) is reported in SI.

V2 additions over V1:
  - HELD_OUT_BASINS: 5 basins excluded from training, one per field
  - EVAL_CONDITIONS A–E: all conditions defined here
  - BOOSTING_PARAMS_M1 / BOOSTING_PARAMS_M2: separate per-model hyperparameters
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# Data paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

EVENT_CSV       = DATA_DIR / "event_dataset.csv"
RESET_CSV       = DATA_DIR / "reset_dataset.csv"
DAILY_TEMP_CSV  = DATA_DIR / "daily_temperature.csv"
OUTLIER_CSV     = DATA_DIR / "outlier_basins.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────────────────────────────────────
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR  = OUTPUTS_DIR / "tables"

for _dir in [OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

BASIN_METRICS_XLSX    = TABLES_DIR / "basin_metrics.xlsx"
MODEL_COMPARISON_XLSX = TABLES_DIR / "model_comparison_v2.xlsx"
RESET_RESULTS_XLSX    = TABLES_DIR / "model2_results_v2.xlsx"

# ─────────────────────────────────────────────────────────────────────────────
# Data configuration
# ─────────────────────────────────────────────────────────────────────────────
DATA_CUTOFF = "2025-01-01"

QUALITY_FILTER = {
    "IRD_R_squared": 0.94,   # drainage regression quality (Elkayam & Lev, 2024)
    "CIV":           3000,   # m³
    "Ct":            20,     # hours
    "AL":            5,      # cm
}

MIN_EVENTS_PER_SEGMENT    = 3
MIN_SEGMENT_R2            = 0.10
PEARSON_OUTLIER_THRESHOLD = -0.05

TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Held-out basins — fixed before any model training
# ─────────────────────────────────────────────────────────────────────────────
# One basin per field, selected as median performers.
# These basins are NEVER seen during training for conditions D and E.
# Their data is used exclusively for testing.
# Field mapping: 3=Soreq2, 4=Yavne1, 5=Yavne2, 6=Yavne3, 7=Yavne4
HELD_OUT_BASINS = {
    "Soreq 2": 3203,
    "Yavne 1": 4104,
    "Yavne 2": 5102,
    "Yavne 3": 6303,
    "Yavne 4": 7201,
}
HELD_OUT_BASIN_LIST = list(HELD_OUT_BASINS.values())
# [3203, 4104, 5102, 6303, 7201]

# ─────────────────────────────────────────────────────────────────────────────
# Outlier basins — excluded from clean model training
# ─────────────────────────────────────────────────────────────────────────────
# Identified via per-basin R² < 0 on the all-50-basin pass-1 model.
# Two types:
#   Type 1 — Low dynamic range: IRD_at_reset IQR < 0.60 cm/h
#             (below system 20th percentile)
#             Basins: 7102, 4304, 4303, 7103, 7202
#   Type 3 — Non-stationary operations: IRD_at_reset trends monotonically
#             across segments, violating stationarity assumption
#             Basins: 6101, 7303, 4103
#
# Note: Type 2 (extreme high variability) was defined in code but no basins
# met this threshold — the system is relatively uniform at the high end.
#
# Loaded at runtime from data/outlier_basins.csv (produced by basin_analysis.py)
# OUTLIER_BASINS list is NOT hardcoded here — read from CSV to allow reruns.

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation conditions
# ─────────────────────────────────────────────────────────────────────────────
# Model 1 conditions (5 total):
#   A — Clean basins (37), good segments only, random segment split
#       → within-sample, filtered data (SI reference)
#   B — Clean basins (37), ALL segments, random segment split
#       → within-sample, unfiltered (SI)
#   C — All basins (50), ALL segments, random segment split
#       → within-sample, all data (SI)
#   D — Clean basins (37), good segments only, 5 held-out test basins
#       → held-out, filtered (SI reference for generalizability)
#   E — All non-held-out basins (45 incl. outliers), ALL segments, 5 held-out test
#       → held-out, all data → PRIMARY PAPER CONDITION
#
# Model 2 conditions (3 total):
#   Chrono      — Clean basins (37), chrono 70/15/15 split (within-sample)
#   Held-out D  — Clean basins (37), 5 held-out test basins (SI)
#   Held-out E  — 45 basins incl. outliers, 5 held-out → PRIMARY PAPER CONDITION

EVAL_CONDITIONS = {
    "A": {
        "label":      "Clean basins, good segments, within-sample",
        "basins":     "clean",
        "good_only":  True,
        "split_type": "random_segment",
        "si_only":    True,      # SI only — not primary
        "note":       "Reference within-sample condition. Reported in SI Table S2.",
    },
    "B": {
        "label":      "Clean basins, all segments, within-sample",
        "basins":     "clean",
        "good_only":  False,
        "split_type": "random_segment",
        "si_only":    True,
        "note":       "Tests effect of segment filtering within clean basins.",
    },
    "C": {
        "label":      "All basins, all segments, within-sample",
        "basins":     "all",
        "good_only":  False,
        "split_type": "random_segment",
        "si_only":    True,
        "note":       "Tests effect of including outlier basins within-sample.",
    },
    "D": {
        "label":      "Clean basins, good segments, held-out test",
        "basins":     "held_out_train",   # 37 clean basins minus held-out
        "good_only":  True,
        "split_type": "held_out_basin",   # test = 5 held-out basins
        "si_only":    True,
        "note":       "Held-out generalizability with filtered training data.",
    },
    "E": {
        "label":      "All-data held-out test — PRIMARY",
        "basins":     "all_held_out_train",   # 45 non-held-out basins incl. outliers
        "good_only":  False,                  # all segments including no-decay
        "split_type": "held_out_basin",       # test = 5 held-out basins
        "si_only":    False,                  # PRIMARY — in main paper
        "note":       (
            "PRIMARY condition. Most general and honest evaluation. "
            "Training includes outlier basins and all segments. "
            "Test is 5 completely unseen basins. "
            "Result: pooled R²=+0.898, RMSE=0.607 cm/h, MAPE=13.1% (n=4,386)."
        ),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Outlier detection thresholds
# ─────────────────────────────────────────────────────────────────────────────
OUTLIER_R2_THRESHOLD       = 0.0    # pass-1 basin R² below this → candidate outlier
OUTLIER_REL_RMSE_THRESHOLD = 1.0    # rel_RMSE above this → candidate outlier

# ─────────────────────────────────────────────────────────────────────────────
# Model hyperparameters — LightGBM
# ─────────────────────────────────────────────────────────────────────────────
# Published defaults — no grid search performed.
# This is a deliberate choice: using published defaults without tuning
# strengthens the transferability claim (model not overfit to Shafdan system).
#
# NOTE: These constants are the authoritative source.
# model1_decay.py, model2_reset.py, and all figure scripts should import
# these rather than defining inline hyperparameters.

# Model 1 — within-segment decay prediction
# Larger num_leaves (63) for more complex within-segment patterns
BOOSTING_PARAMS_M1 = dict(
    n_estimators      = 1000,
    max_depth         = -1,       # unlimited — controlled by num_leaves
    num_leaves        = 63,
    learning_rate     = 0.05,
    subsample         = 0.8,
    feature_fraction  = 0.8,
    min_child_samples = 20,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = RANDOM_SEED,
    n_jobs            = -1,
    verbose           = -1,
)
EARLY_STOPPING_ROUNDS_M1 = 50

# Model 2 — post-tillage recovery prediction
# Smaller num_leaves (31) — fewer training points (~4,000 resets vs ~46,000 events)
BOOSTING_PARAMS_M2 = dict(
    n_estimators      = 1000,
    max_depth         = -1,
    num_leaves        = 31,
    learning_rate     = 0.05,
    subsample         = 0.8,
    feature_fraction  = 0.8,
    min_child_samples = 10,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = RANDOM_SEED,
    n_jobs            = -1,
    verbose           = -1,
)
EARLY_STOPPING_ROUNDS_M2 = 50

# Legacy aliases — kept for backward compatibility with older scripts
# Do not use in new code — use BOOSTING_PARAMS_M1 / BOOSTING_PARAMS_M2
BOOSTING_PARAMS = BOOSTING_PARAMS_M1
EARLY_STOPPING_ROUNDS = EARLY_STOPPING_ROUNDS_M1

# ─────────────────────────────────────────────────────────────────────────────
# Seasonality encoding
# ─────────────────────────────────────────────────────────────────────────────
# SEASON_PHASE = 4: month_sin = sin(2π(month-4)/12)
# This makes month_sin peak at July (month=7) with value +1.0
# and trough at January (month=1) with value -1.0.
# Consistent with Mediterranean climate: July = hottest, driest month.
SEASON_PHASE = 4

# Season bands for time series plots (plot_style.py)
# Each entry: list of months, hex color, default alpha
SEASONS = {
    "Winter":  ([12, 1, 2],   "#DAE8FC", 0.60),
    "Spring":  ([3, 4, 5],    "#FCE4D6", 0.60),
    "Summer":  ([6, 7, 8],    "#FFF3CD", 0.60),
    "Autumn":  ([9, 10, 11],  "#D5E8D4", 0.60),
}

HEATMAP_SEASONS = {
    "Winter\n(Dec–Feb)":              [12, 1, 2],
    "Transition\n(Mar–May, Sep–Nov)": [3, 4, 5, 9, 10, 11],
    "Summer\n(Jun–Aug)":              [6, 7, 8],
}
HEATMAP_SEASON_MID_MONTH = {
    "Winter\n(Dec–Feb)":              1,
    "Transition\n(Mar–May, Sep–Nov)": 4,
    "Summer\n(Jun–Aug)":              7,
}

# ─────────────────────────────────────────────────────────────────────────────
# Plotting constants
# ─────────────────────────────────────────────────────────────────────────────
# Field colors — consistent across ALL figures (paper + SI)
# Must match plot_style.py FIELD_COLORS
FIELD_COLORS = {
    "Soreq 2": "#065A82",
    "Yavne 1": "#1C7293",
    "Yavne 2": "#E07B39",
    "Yavne 3": "#27AE60",
    "Yavne 4": "#7D3C98",
}

# Split colors for diagnostic plots
SPLIT_COLORS  = {"train": "#555555", "val": "#1C7293", "test": "#E07B39"}
SPLIT_MARKERS = {"train": "o",       "val": "s",       "test": "^"}
SPLIT_ALPHA   = {"train": 0.35,      "val": 0.85,      "test": 0.85}
SPLIT_SIZE    = {"train": 15,        "val": 40,        "test": 40}

FIGURE_DPI = 300   # production quality for paper

# ─────────────────────────────────────────────────────────────────────────────
# Domain metadata
# ─────────────────────────────────────────────────────────────────────────────
FIELD_NAMES = {
    3: "Soreq 2",
    4: "Yavne 1",
    5: "Yavne 2",
    6: "Yavne 3",
    7: "Yavne 4",
}