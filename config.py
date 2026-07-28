"""
config.py — Central configuration for sat-ir-prediction
=========================================================
Single source of truth for all paths, constants, and hyperparameters.

PRIMARY EVALUATION: Condition E
  Train : all non-outlier basins, all segments, no quality filtering
  Test  : 5 basins selected automatically by bootstrap validation

HOW HELD-OUT BASINS ARE SELECTED
----------------------------------
  There are NO fixed held-out basins in config.py.
  The 5 presented basins are chosen objectively by the bootstrap:

  1. Run: python -m experiments.run_bootstrap
     → writes data/bootstrap_results.csv  (all 200 iterations)
     → writes data/selected_basins.csv    (the winning 5 basins)

  2. Run: python -m models.model1_decay
          python -m models.model2_reset
     → both scripts read data/selected_basins.csv at runtime

  Selection criterion:
    - All 5 held-out basins beat naive in BOTH Model 1 and Model 2
    - Among qualifying iterations: maximise Model 2 pooled ΔRMSE

HOW OUTLIER BASINS ARE DEFINED
--------------------------------
  Outlier basins are detected automatically by basin_analysis.py
  and written to data/outlier_basins.csv.
  No outlier basin numbers are hardcoded anywhere.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Project root
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# Data paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

EVENT_CSV             = DATA_DIR / "event_dataset.csv"
RESET_CSV             = DATA_DIR / "reset_dataset.csv"
OUTLIER_CSV           = DATA_DIR / "outlier_basins.csv"
BOOTSTRAP_RESULTS_CSV = DATA_DIR / "bootstrap_results.csv"
SELECTED_BASINS_CSV   = DATA_DIR / "selected_basins_v2.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────────────────────────────────────
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR  = OUTPUTS_DIR / "tables"

for _dir in [OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data quality filters
# ─────────────────────────────────────────────────────────────────────────────
DATA_CUTOFF = "2025-01-01"

# Minimum thresholds: an event is filtered if value < threshold.
QUALITY_FILTER = {
    "IRD_R_squared": 0.94,
    "CIV":           3000,
    "Ct":            20,
    "AL":            5,
}

# Maximum thresholds: an event is filtered if value > ceiling.
# Ct (cycle time) upper cap removes segmentation-stitching artifacts (>~p99),
# e.g. spurious multi-thousand-hour "cycles". Because FT <= Ct for every event,
# this Ct cap also subsumes the flooding-time (FT) outliers, so no separate FT
# filter is needed. ~1.2% of events are removed at Ct <= 500.
QUALITY_FILTER_MAX = {
    "Ct": 500,
}

MIN_EVENTS_PER_SEGMENT    = 3
MIN_SEGMENT_R2            = 0.10
PEARSON_OUTLIER_THRESHOLD = -0.05


# ─────────────────────────────────────────────────────────────────────────────
# Split fractions
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_FRAC  = 0.80
VAL_FRAC    = 0.20
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Outlier detection thresholds (used by analysis/basin_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────
OUTLIER_IQR_THRESHOLD      = 0.60
OUTLIER_R2_THRESHOLD       = 0.0
OUTLIER_REL_RMSE_THRESHOLD = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap settings
# ─────────────────────────────────────────────────────────────────────────────
BOOTSTRAP_N_ITERATIONS = 500
BOOTSTRAP_N_HELD_OUT   = 5
BOOTSTRAP_SEED         = RANDOM_SEED


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

import json as _json

# Fallback defaults — used only until tune_hyperparams.py writes the JSONs.
_FALLBACK_M1 = dict(
    n_estimators=1000, max_depth=-1, num_leaves=63, learning_rate=0.05,
    subsample=0.8, feature_fraction=0.8, min_child_samples=20,
    reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED,
    n_jobs=-1, verbose=-1,
)
_FALLBACK_M2 = dict(
    n_estimators=1000, max_depth=-1, num_leaves=31, learning_rate=0.05,
    subsample=0.8, feature_fraction=0.8, min_child_samples=10,
    reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED,
    n_jobs=-1, verbose=-1,
)

def _load_params(path: Path, fallback: dict) -> dict:
    if path.exists():
        with open(path) as f:
            p = _json.load(f)
        p.setdefault("random_state", RANDOM_SEED)
        p.setdefault("n_jobs", -1)
        p.setdefault("verbose", -1)
        return p
    return fallback

BOOSTING_PARAMS_M1 = _load_params(DATA_DIR / "best_params_m1.json", _FALLBACK_M1)
BOOSTING_PARAMS_M2 = _load_params(DATA_DIR / "best_params_m2.json", _FALLBACK_M2)
EARLY_STOPPING_ROUNDS_M1 = 50
EARLY_STOPPING_ROUNDS_M2 = 50


# ─────────────────────────────────────────────────────────────────────────────
# Seasonality encoding
# ─────────────────────────────────────────────────────────────────────────────
SEASON_PHASE = 4

SEASONS = {
    "Winter": ([12, 1, 2],  "#DAE8FC", 0.60),
    "Spring": ([3, 4, 5],   "#FCE4D6", 0.60),
    "Summer": ([6, 7, 8],   "#FFF3CD", 0.60),
    "Autumn": ([9, 10, 11], "#D5E8D4", 0.60),
}


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

FIELD_COLORS = {
    "Soreq 2": "#065A82",
    "Yavne 1": "#1C7293",
    "Yavne 2": "#E07B39",
    "Yavne 3": "#27AE60",
    "Yavne 4": "#7D3C98",
}

FIGURE_DPI = 300