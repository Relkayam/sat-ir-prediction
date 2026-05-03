"""
config.py — Central configuration for sat-ird-prediction-v2
=============================================================
V2 additions over V1:
  - HELD_OUT_BASINS: basins excluded from training, used for
    out-of-sample generalizability test (one per field)
  - EVALUATION_CONDITIONS: A, B, C, D conditions for SI comparison
  - All other constants identical to V1
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
MODEL_COMPARISON_XLSX = TABLES_DIR / "model_comparison.xlsx"
RESET_RESULTS_XLSX    = TABLES_DIR / "reset_model_results.xlsx"

# ─────────────────────────────────────────────────────────────────────────────
# Data configuration
# ─────────────────────────────────────────────────────────────────────────────
DATA_CUTOFF = "2025-01-01"

QUALITY_FILTER = {
    "IRD_R_squared": 0.94,  # drainage regression quality (Elkayam & Lev, 2024)
    "CIV":           3000,  # m³
    "Ct":            20,    # hours
    "AL":            5,     # cm
}

MIN_EVENTS_PER_SEGMENT    = 3
MIN_SEGMENT_R2            = 0.10
PEARSON_OUTLIER_THRESHOLD = -0.05

TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Outlier basins — excluded from clean model training
# ─────────────────────────────────────────────────────────────────────────────
# Identified via per-basin R² < 0 on the global model.
# Two types:
#   Type 1 — Low dynamic range (IRD IQR below system 20th percentile)
#   Type 3 — Non-stationary operations
# OUTLIER_BASINS = [7202, 7102, 4304, 4303, 7103, 7303, 6101]

# ─────────────────────────────────────────────────────────────────────────────
# Held-out basins — for out-of-sample generalizability test (V2)
# ─────────────────────────────────────────────────────────────────────────────
# One basin per field, selected as median performers.
# These basins are NEVER seen during training in condition D.
# Their data is used exclusively for testing.
# Field mapping: 3=Soreq2, 4=Yavne1, 5=Yavne2, 6=Yavne3, 7=Yavne4
HELD_OUT_BASINS = {
    "Soreq 2": 3203,
    "Yavne 1": 4104,
    "Yavne 2": 5102,
    "Yavne 3": 6303,
    "Yavne 4": 7201,
}
HELD_OUT_BASIN_LIST = list(HELD_OUT_BASINS.values())  # [3203, 4104, 5102, 6303, 7201]

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation conditions (V2)
# ─────────────────────────────────────────────────────────────────────────────
# Used in models/model1_decay.py, models/model2_reset.py,
# and all comparison scripts.
#
# A — Paper model: 43 clean basins, good segments only, random segment split
# B — All segments: 43 clean basins, ALL segments, random segment split
# C — All data: 50 basins, ALL segments, random segment split
# D — Held-out basins: 38 clean basins for training,
#     5 held-out basins for test only (never seen during training)
#
# Conditions B, C, D reported in SI. Condition A is the paper model.

EVAL_CONDITIONS = {
    "A": {
        "label":       "Clean model (paper)",
        "basins":      "clean",        # resolved at runtime
        "good_only":   True,
        "split_type":  "random_segment",
        "si_only":     False,
    },
    "B": {
        "label":       "All segments, clean basins",
        "basins":      "clean",
        "good_only":   False,
        "split_type":  "random_segment",
        "si_only":     True,
    },
    "C": {
        "label":       "All segments, all basins",
        "basins":      "all",
        "good_only":   False,
        "split_type":  "random_segment",
        "si_only":     True,
    },
    "D": {
        "label":       "Held-out basin test",
        "basins":      "held_out_train",  # 38 clean basins minus held-out
        "good_only":   True,
        "split_type":  "held_out_basin",  # test = held-out basins only
        "si_only":     True,
    },

    "E": {
        "label":       "All-data held-out test",
        "basins":      "all_held_out_train",  # all 45 non-held-out basins
        "good_only":   False,                 # all segments including no-decay
        "split_type":  "held_out_basin",
        "si_only":     True,
    },

}

# ─────────────────────────────────────────────────────────────────────────────
# Outlier detection thresholds
# ─────────────────────────────────────────────────────────────────────────────
OUTLIER_R2_THRESHOLD       = 0.0
OUTLIER_REL_RMSE_THRESHOLD = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Seasonality encoding
# ─────────────────────────────────────────────────────────────────────────────
SEASON_PHASE = 4

SEASONS = {
    "Winter":  ([12, 1, 2],   "#aed6f1", 0.18),
    "Spring":  ([3, 4, 5],    "#a9dfbf", 0.18),
    "Summer":  ([6, 7, 8],    "#f9e79f", 0.18),
    "Autumn":  ([9, 10, 11],  "#f0b27a", 0.18),
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
# Model hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
BOOSTING_PARAMS = dict(
    n_estimators     = 500,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
)
EARLY_STOPPING_ROUNDS = 30

BOOSTING_PARAMS_M2 = dict(
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = RANDOM_SEED,
    n_jobs           = -1,
)
EARLY_STOPPING_ROUNDS_M2 = 20

# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
SPLIT_COLORS  = {"train": "#555555", "val": "steelblue",  "test": "tomato"}
SPLIT_MARKERS = {"train": "o",       "val": "s",          "test": "^"}
SPLIT_ALPHA   = {"train": 0.35,      "val": 0.85,         "test": 0.85}
SPLIT_SIZE    = {"train": 15,        "val": 40,           "test": 40}

FIELD_COLORS = {
    "Soreq 2": "steelblue",
    "Yavne 1": "seagreen",
    "Yavne 2": "tomato",
    "Yavne 3": "orange",
    "Yavne 4": "mediumpurple",
}

FIGURE_DPI = 150

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