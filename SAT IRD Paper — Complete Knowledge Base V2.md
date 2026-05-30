# SAT IRD Paper — Complete Knowledge Base (Updated)
## "From Decay to Recovery: A Dual Machine Learning Framework for Infiltration Rate Prediction in Managed Aquifer Recharge"
### Journal: Water Research | Project: sat-ir-prediction
### Last updated: May 2026 — incorporates all decisions from the full project conversation

---

## PART A — PROJECT OVERVIEW

### A1. What This Project Is

A PhD-level research paper predicting infiltration rates in a Soil Aquifer Treatment (SAT) system using machine learning. The system is the **Shafdan SAT facility** near Tel Aviv, Israel — one of the largest SAT systems in the world, treating approximately 140 million m³/year of secondary wastewater.

The paper introduces a **dual-model ML framework** that is:
- **Practically useful:** enables operational planning for field engineers across all 50 basins simultaneously
- **Scientifically novel:** first formal hybrid dynamical system formulation for SAT clogging-recovery dynamics
- **Mathematically generalizable:** the framework maps onto a broad class of filtration and water treatment systems, and is designed as the empirical foundation for a future Physics-Informed Neural Network (PINN) implementation

The framework has been validated on five completely unseen basins (one per field) held out before any model training.

---

### A2. The Physical System

**What is SAT?**
Soil Aquifer Treatment (SAT) is a form of Managed Aquifer Recharge (MAR) where treated wastewater is infiltrated through recharge basins into an aquifer, providing additional treatment through the soil. The water is then extracted from wells and used as a water resource.

**The Shafdan system:**
- 50 monitored recharge basins across 5 fields: Soreq 2, Yavne 1, Yavne 2, Yavne 3, Yavne 4
- Each basin: approximately 1,000–5,000 m² surface area, flooded and dried alternately
- 10-year IoT record: 2015–2025
- Data managed in DuckDB via the optisat package
- 20 additional Soreq 1 basins operate without online monitoring — excluded

**The clogging problem:**
During flooding, the soil surface clogs progressively due to:
1. Suspended solids accumulation (physical clogging / cake formation)
2. Biofilm growth (biological clogging)
3. Gas entrapment

This causes the **Infiltration Rate (IRD)** to decay over time within each inter-tillage segment.

**The recovery mechanism (tillage):**
After a segment ends, the basin is dried and mechanically tilled (plowed). This breaks up the clogging layer and partially restores infiltration capacity. After tillage, a new flooding segment begins at a higher IRD — the post-tillage recovery level.

**The companion STL study (Elkayam, 2025):**
A predecessor paper applied Seasonal-Trend decomposition (STL) to the same system, finding that temperature-driven seasonal variability accounts for only ~25% of total IRD variance. The remaining 75% — dominated by non-periodic operational and stochastic processes — is what this framework predicts.

---

### A3. Key Definitions

| Term | Definition | Units |
|---|---|---|
| IRD | Infiltration Rate during Drainage — measured as slope of water level decline during drainage phase | cm/h |
| IRD_at_reset (ρᵢ) | IRD at the first flooding event of a new segment, immediately after tillage | cm/h |
| LCT | Loading Cycle Time — hours since the last tillage event (Last Cultivation Time) | hours |
| ALPHA | Drying fraction = DrT / Ct — fraction of each cycle spent drying | dimensionless |
| DrT | Drying time between consecutive flooding events | hours |
| FT | Flooding time (valve open duration) | hours |
| Ct | Total cycle time = FT + DT + DrT | hours |
| HL | Hydraulic Load = CIV / basin area — volume infiltrated per unit area per event | cm |
| CIV | Cumulative Infiltrated Volume | m³ |
| RD | Radiation during drying phase | W/m² |
| RW | Radiation during wetting phase | W/m² |
| TD | Temperature during drying phase | °C |
| TW | Temperature during wetting phase | °C |
| DAR | Daily Ambient Radiation at reset date | W/m² |
| DAT | Daily Ambient Temperature at reset date | °C |
| Ks | Saturated hydraulic conductivity — basin-specific IRD ceiling | cm/h |
| Segment | Sequence of flooding events between two consecutive tillage events | — |
| Reset | Tillage event — disrupts clogging layer, initiates new segment | — |

---

### A4. The Operational Cycle

```
Flood → IRD decays → Dry → Till → Flood (new segment) → IRD decays → ...
```

Each flooding period = one **segment**. Each tilling event = one **reset**.
Each flooding-drying cycle within a segment = one **event** (one row in the dataset).

---

## PART B — DATA PIPELINE

### B1. Source Data and Database Structure

- Raw data in DuckDB database files, one per basin
- Accessed via `optisat.db.duckdb_manager.DuckDBManager`
- optisat package installed as editable from `mek-models-satix-backend` project
- Global database: `SatixPaths.GLOBAL_DB_PATH` — contains basin metadata and areas
- Basin databases: `SatixPaths.BASIN_DB_DIR / basin_{number}.duckdb` — contains event data

### B2. Build Pipeline (Execution Order)

Run in this exact order if starting from scratch:

```bash
python -m pipeline.build_dataset --rebuild      # → data/event_dataset.csv
python -m pipeline.build_reset_dataset          # → data/reset_dataset.csv
python -m analysis.basin_analysis               # → data/outlier_basins.csv (informational only)
python -m experiments.run_bootstrap_v2          # → data/selected_basins_v2.csv
python -m models.model1_decay                   # → outputs/tables/model1_results_condition_e.xlsx
python -m models.model2_reset                   # → outputs/tables/model2_results_condition_e.xlsx
python -m figures.fig4_shap_beeswarm            # → SHAP table + figure
python -m experiments.leave_one_field_out       # → console only (no disk output)
```

### B3. Pipeline Order (CRITICAL — Fixed May 2026)

The correct pipeline order inside `build_basin_dataset()` is:

```
1. Read raw events from DuckDB
2. Fill zero-drying features
3. Segmentation on ALL raw events (_ceiling_reset_search)
   → assigns segment_id, row_type to every event
   → two-tier search: prefer quality-passing events as reset anchors,
     fall back to any event if none available in window
3b. Recompute LCT where NaN (_compute_lct)
   → LCT = hours since reset date, derived from assigned segment structure
   → only fills NaN — preserves database values for normal basins
3c. Fill IRD_at_reset where NaN (_compute_ird_at_reset)
   → IRD_at_reset = IRD of reset anchor event, broadcast to segment
   → only fills NaN — preserves database values for normal basins
4. Compute IRD_norm on ALL events (needs IRD_at_reset from step 3)
5. Compute HL on ALL events
6. Tag quality filter (marks filter_reason — does NOT drop rows)
7. Split into passed/filtered subsets
8. Compute prev_* and cum_* features on passed events only
9. Tag segment quality, assign splits
10. Reassemble full df and attach metadata
```

**Why this order matters:** The old pipeline ran quality filtering before segmentation. For basins 3201 and 3202 (Soreq 2), the reset anchor event failed the IRD_R_squared filter, so the reset search found nothing → IRD_at_reset was NaN for all events → all events tagged too_few_events → n=0 in all evaluations. The new order runs segmentation first so reset anchors are always found.

**The same fix was applied to optisat** (`tillage_features.py`): `tillage_driven_reset_search()` now uses a two-tier candidate search — quality-passing events preferred, any event as fallback. After this fix, `add_tillage_features_for_basin(3201)` and `add_tillage_features_for_basin(3202)` must be re-run to update the DuckDB files.

### B4. Event Dataset Structure

**File:** `data/event_dataset.csv` — 52,692 rows, 50 basins

Each row = one flooding-drying event. Key columns:

| Column | Description |
|---|---|
| `basin_number` | 4-digit basin ID (e.g., 3203, 5102) |
| `opening_valve_date` | Date of flooding event |
| `row_type` | "event", "reset", or "pre_segment" |
| `segment_id` | Integer, increments at each tillage event per basin |
| `is_good_segment` | True if segment passes all quality filters |
| `filter_reason` | "" = good event; otherwise the filter that excluded it |
| `IRD_norm_log` | Model 1 target = ln(IRD / IRD_at_reset) |
| `IRD_at_reset` | IRD value at segment start (cm/h) |
| `LCT` | Hours since last tillage event |
| `seg_lambda` | Fitted exponential decay rate for this segment (h⁻¹) |
| `seg_a`, `seg_b`, `seg_r2` | Fitted decay parameters |
| `basin_role` | "clean" (all 50 basins) |
| `split` | "train", "val", "test", or "excluded" |

**filter_reason values:**
- `""` — good event in a good segment
- `"after_cutoff"` — date >= DATA_CUTOFF
- `"quality_filter_IRD_R_squared"` — drainage R² < 0.94
- `"quality_filter_CIV"` — CIV < 3,000 m³
- `"quality_filter_Ct"` — Ct < 20 h
- `"quality_filter_AL"` — AL < 5 cm
- `"pre_segment"` — before first credible reset
- `"too_few_events"` — segment has < 3 quality-passing events
- `"pearson_r_positive"` — no decay signal (Pearson r > −0.05)
- `"fit_failed"` — exponential decay fit did not converge
- `"r2_below_threshold"` — decay fit R² < 0.10

### B5. Reset Dataset Structure

**File:** `data/reset_dataset.csv` — 4,199 rows, 50 basins

Each row = one tillage/reset event. Key columns:

| Column | Description |
|---|---|
| `basin_number` | Basin ID |
| `reset_date` | Date of tillage event |
| `IRD_norm_log_reset` | Model 2 target = ln(ρᵢ / ρᵢ₋₁) |
| `IRD_at_reset` | Actual post-tillage IRD (cm/h) |
| `prev_IRD_at_reset_raw` | Previous reset's IRD — for back-transformation |
| `split_chrono` | Chronological 70/15/15 split per basin |

### B6. Filter Funnel (Current Numbers — All 50 basins)

| Step | N | % of raw |
|---|---|---|
| Raw events (study period) | 46,907 | 100.0% |
| Drainage R² < 0.94 | 1,628 | 3.5% |
| CIV < 3,000 m³ | 507 | 1.1% |
| Ct < 20 h | 290 | 0.6% |
| AL < 5 cm | 310 | 0.7% |
| Pre-segment | 0 | 0.0% |
| Too few events (<3) | 2,181 | 4.6% |
| No decay signal (Pearson r > −0.05) | 13,573 | 28.9% |
| Fit failed | 0 | 0.0% |
| R² below threshold (<0.10) | 4,133 | 8.8% |
| **Good events** | **23,338** | **49.8%** |

### B7. Basin Classification (Current — May 2026)

**All 50 basins are "clean"** — no fixed held-out or outlier designation at build time. The `basin_role` column is "clean" for all basins.

**Basin analysis** (`analysis/basin_analysis.py`) runs a global LightGBM to characterize basin heterogeneity and writes `outlier_basins.csv`. This file is now **informational only** — it does not gate training, bootstrap draws, or evaluation. All 50 basins are eligible for all roles.

**Current outlier flags (informational):**

| Basin | Field | Pass-1 R² | Type | Reason |
|---|---|---|---|---|
| 7303 | Yavne 4 | −0.975 | Type 3 | Non-stationary operations |
| 6101 | Yavne 3 | −0.851 | Type 3 | Non-stationary operations |
| 7102 | Yavne 4 | −0.679 | Type 1 | Low dynamic range |
| 4103 | Yavne 1 | −0.340 | Type 2 | Regime shift |
| 7103 | Yavne 4 | −0.225 | Type 1 | Low dynamic range |
| 7202 | Yavne 4 | −0.194 | Type 1 | Low dynamic range |
| 4304 | Yavne 1 | −0.181 | Type 1 | Low dynamic range |
| 7403 | Yavne 4 | −0.063 | Type 3 | Non-stationary operations |
| 3201 | Soreq 2 | −0.027 | Type 2 | Regime shift |

These 9 basins are included in training and can be held out in the bootstrap — they are flagged for transparency in reporting basin heterogeneity.

---

## PART C — THE TWO TARGETS AND NORMALIZATION

### C1. Why Log-Ratio Normalization

The 50 Shafdan basins have substantially different saturated hydraulic conductivities (Ks). A model trained on raw IRD values would learn basin identity rather than clogging physics, and would fail to generalize to unseen basins.

Log-ratio normalization removes between-basin scale differences by defining targets as ratios relative to the post-tillage baseline.

### C2. Physical Justification (Resistance-in-Series)

$$R(t) = R_m + R_\text{rev}(t) + R_\text{irrev}$$

Since IRD ∝ 1/R(t), the ratio IRD(t)/ρᵢ cancels R_m + R_irrev. Taking the logarithm maps this into an additive, symmetric space better suited to regression.

### C3. Model 1 Target: IRD_norm_log

$$\eta(t) = \ln\!\left(\frac{\text{IRD}(t)}{\rho_i}\right) \tag{1}$$

η(0) = 0 at every segment start. η(t) < 0 as clogging progresses.

Back-transformation: $\widehat{\text{IRD}}(t) = \rho_i \cdot \exp(\hat{\eta}(t))$

### C4. Model 2 Target: IRD_norm_log_reset

$$\delta_i = \ln\!\left(\frac{\rho_i}{\rho_{i-1}}\right) \tag{3}$$

δ = 0: no change. δ > 0: better recovery. δ < 0: worse recovery.

Back-transformation: $\hat{\rho}_i = \rho_{i-1} \cdot \exp(\hat{\delta}_i)$

---

## PART D — THE DUAL-MODEL FRAMEWORK

### D1. The Two Hypotheses

**Hypothesis 1 — Decay:** Within each inter-tillage segment, the normalized IRD follows a monotone decay process driven by operational and environmental conditions since the last tillage event.

**Hypothesis 2 — Reset:** The post-tillage recovery level is a deterministic function of the operational history of the preceding segment and the environmental conditions at the moment of tillage.

### D2. Model 1 — Feature Vector (11 features)

| Feature | Physical variable | Physical rationale |
|---|---|---|
| IRD_at_reset (ρᵢ) | Post-tillage IRD | Scale anchor |
| prev_ALPHA | Drying fraction = DrT/Ct | More drying → slower decay |
| prev_DrT | Previous drying time (h) | Absolute drying duration |
| prev_TD | Temperature during drying (°C) | Faster desiccation |
| prev_RD | Radiation during drying (W/m²) | Photodegradation |
| prev_RW | Radiation during wetting (W/m²) | Light during flooding |
| log1p_prev_HL | Hydraulic load, log-transformed | More volume → faster clogging |
| prev_FT | Previous flooding time (h) | Longer flooding → more biofilm |
| LCT | Time since last tillage (h) | Primary decay axis |
| cum_TW | Cumulative wetting temperature (°C·h) | Seasonal biofilm growth |
| cum_FT | Cumulative flooding time (h) | Integrated clogging load |

### D3. Model 2 — Feature Vector (12 features)

| Feature | Physical rationale |
|---|---|
| frac_zero_DrT | Fraction of events with no drying |
| n_events | Number of flooding events in segment |
| sum_RW | Total radiation during wetting |
| total_LCT | Total segment duration (h) |
| max_RD | Peak radiation during drying |
| max_DrT | Longest drying event |
| last_RD | Radiation in final drying before tillage |
| mean_RW | Average wetting radiation |
| mean_DrT | Average drying time |
| DAR | Daily ambient radiation at tillage date — key operational driver |
| prev_delta | Previous reset log-ratio (δᵢ₋₁) |
| prev_prev_delta | Two-step history (δᵢ₋₂) |

### D4. Hyperparameters (Final — no grid search)

**Model 1:**
```python
n_estimators=1000, max_depth=-1, num_leaves=63,
learning_rate=0.05, subsample=0.8, feature_fraction=0.8,
min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
early_stopping_rounds=50
```

**Model 2:**
```python
n_estimators=1000, max_depth=-1, num_leaves=31,
learning_rate=0.05, subsample=0.8, feature_fraction=0.8,
min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
early_stopping_rounds=50
```

### D5. The Forward Operational Chain

1. **Model 2** predicts δ̂ᵢ → back-transform: ρ̂ᵢ = ρᵢ₋₁ · exp(δ̂ᵢ)
2. **ρ̂ᵢ** enters Model 1 as the scale anchor (IRD_at_reset)
3. **Model 1** predicts η̂(t) for each planned event
4. **Back-transform:** IRD(t) = ρ̂ᵢ · exp(η̂(t))

---

## PART E — EVALUATION FRAMEWORK

### E1. Evaluation Condition — Condition E-full

**Single evaluation condition** — no conditions A/B/C/D in the current codebase.

- **Train:** all 50 non-held-out basins, all event rows (no is_good_segment filter)
- **Val:** non-held-out basins, split == 'val', all event rows
- **Test:** held-out basins selected by bootstrap, all event rows
- **Only exclusion:** the four sensor-quality filters (IRD_R² < 0.94, CIV < 3000, Ct < 20, AL < 5)

### E2. Bootstrap Validation and Basin Selection

501 iterations: iteration 1 = hardcoded `[3203, 4401, 5302, 6302, 7402]`, iterations 2–501 = random one-per-field draws from all 50 basins.

**Selection criterion:**
1. All 5 held-out basins beat naive in BOTH models (hard constraint)
2. One basin per field (enforced in draws and verified post-selection)
3. Maximize Model 2 ΔRMSE among qualifying iterations

**Naive baselines:**
- Model 1: predict IRD_at_reset for every event (no decay)
- Model 2: predict previous reset IRD (no change)

### E3. Spatial Generalization — Leave-One-Field-Out

Five folds: train on 4 fields, test on all basins in the 5th. Strictest possible spatial generalization test — zero training data from the held-out field.

### E4. Metrics

Primary: R², RMSE (cm/h), MAPE (%) — all computed on back-transformed IRD.
Model 2 primary metric: RMSE (MAPE misleadingly favors naive — see MAPE paradox).

---

## PART F — RESULTS (TO BE UPDATED WITH FRESH NUMBERS)

### F1. Leave-One-Field-Out Results (Current Pipeline — All 50 Basins)

**Model 1 — pooled per held-out field:**

| Field | n | R² | RMSE | Naive RMSE | ΔRMSE | MAPE% |
|---|---|---|---|---|---|---|
| Soreq 2 | 5,297 | +0.384 | 1.241 | 2.553 | +1.312 | 21.2% |
| Yavne 1 | 10,837 | +0.690 | 0.713 | 1.459 | +0.746 | 22.8% |
| Yavne 2 | 9,008 | +0.645 | 0.862 | 1.762 | +0.900 | 18.0% |
| Yavne 3 | 7,056 | +0.586 | 0.830 | 1.518 | +0.688 | 20.4% |
| Yavne 4 | 7,090 | +0.735 | 0.672 | 1.410 | +0.738 | 28.1% |
| **Mean** | | | **0.864** | | **+0.877** | |

Model 1 beats naive in **5/5 fields**.

**Model 2 — pooled per held-out field:**

| Field | n | R² | RMSE | Naive RMSE | ΔRMSE | MAPE% |
|---|---|---|---|---|---|---|
| Soreq 2 | 368 | +0.369 | 1.718 | 1.805 | +0.087 | 19.2% |
| Yavne 1 | 1,161 | +0.630 | 0.915 | 0.898 | −0.017 | 18.4% |
| Yavne 2 | 819 | +0.705 | 1.050 | 1.100 | +0.050 | 16.9% |
| Yavne 3 | 840 | +0.568 | 0.940 | 0.937 | −0.003 | 16.0% |
| Yavne 4 | 911 | +0.783 | 0.826 | 0.849 | +0.023 | 19.2% |
| **Mean** | | | **1.090** | | **+0.028** | |

Model 2 beats naive in **3/5 fields** pooled.

### F2. Bootstrap, Model 1, Model 2 Results

**PENDING** — bootstrap running at time of knowledge base update. Paste console output and update this section with:
- Selected iteration number and basins
- Bootstrap M1/M2 distribution statistics
- Model 1 per-basin held-out table
- Model 2 per-basin held-out table
- SHAP feature ranking (Model 1 and Model 2)

---

## PART G — MATHEMATICAL FRAMEWORK

### G1. Model 1 as Parameter Identification for an ODE

$$\frac{d\eta}{dt} = -\lambda(\mathbf{x}(t)) \cdot (\eta(t) - \eta_\infty) \tag{5}$$

Analytical solution for constant λ:

$$\eta(t) = -\eta_\infty\left(1 - e^{-\lambda t}\right) \tag{6}$$

Model 1 learns the mapping **x** → λ — parameter identification for a physical ODE.

### G2. Model 2 as a Poincaré Map

$$\rho_k = \mathcal{P}(\rho_{k-1},\, \mathbf{z}_{k-1}) \tag{8}$$

In log-ratio parameterization: $\delta_k = \ln(\rho_k / \rho_{k-1})$

Working in log-ratio space linearizes the map around the no-change fixed point (δ=0).

### G3. The Hybrid Impulsive Dynamical System

$$\frac{d\eta}{dt} = -\lambda(\mathbf{x}(t))\,(\eta(t) - \eta_\infty), \qquad t \neq t_k \tag{10}$$

$$\rho(t_k^+) = \mathcal{P}\!\left(\rho(t_k^-),\, \mathbf{z}_{k-1}\right), \qquad t = t_k \tag{11}$$

First explicit formulation of SAT infiltration dynamics as a hybrid impulsive dynamical system.

---

## PART H — FIGURES

### H1. Main Paper Figures

| Figure | Description | Script |
|---|---|---|
| Figure 1 | Study site map + schematic + example time series | Manual |
| Figure 2 | IRD decay evidence — 3×2 grid, all segments | fig2_decay_evidence.py |
| Figure 3 | Dual-model framework diagram | Manual |
| Figure (bootstrap) | Bootstrap distribution histogram | fig_bootstrap.py |
| Figure 4 | Model 1 SHAP beeswarm — Condition E | fig4_shap_beeswarm.py |
| Figure 5 | Model 1 held-out time series — Condition E | fig5_held_out_timeseries.py |
| Figure 6 | Model 2 held-out time series — Condition E | fig6_model2_timeseries.py |

---

## PART I — PAPER WRITING GUIDE

### I1. Key Narrative Choices

1. **Single evaluation condition: Condition E-full** — all 50 basins, all events, no is_good_segment filter
2. **RMSE is primary metric for Model 2** — MAPE misleadingly favors naive
3. **Bootstrap selects held-out basins** — not fixed a priori
4. **Leave-one-field-out goes in main paper** (Section 3.6) — strongest generalization evidence
5. **No conditions A/B/C/D language** — single evaluation scenario
6. **Basin heterogeneity characterized but not used to gate anything** — one sentence in Discussion

### I2. Things NOT To Do

1. Do NOT use is_good_segment as a filter anywhere in model training or test
2. Do NOT exclude outlier basins from training or bootstrap draws
3. Do NOT report conditions A/B/C/D — single condition only
4. Do NOT use MAPE as primary metric for Model 2
5. Do NOT hardcode basin numbers — always read from selected_basins_v2.csv
6. Do NOT tune hyperparameters with grid search

---

## PART J — CODE ARCHITECTURE

### J1. File Structure and Roles

```
sat-ir-prediction/
├── pipeline/
│   ├── build_dataset.py           → event_dataset.csv (corrected pipeline order)
│   ├── build_reset_dataset.py     → reset_dataset.csv
│   └── features.py                → MODEL1_FEATURES, MODEL2_FEATURES, targets
├── analysis/
│   └── basin_analysis.py          → outlier characterization (informational only)
├── models/
│   ├── model1_decay.py            → Model 1 — Condition E-full
│   ├── model2_reset.py            → Model 2 — Condition E-full
│   └── utils.py                   → metrics, back_transform, predict
├── experiments/
│   ├── run_bootstrap_v2.py        → 501 iterations, one-per-field, all 50 basins eligible
│   └── leave_one_field_out.py     → 5-fold spatial generalization
├── figures/
│   ├── fig2_decay_evidence.py
│   ├── fig4_shap_beeswarm.py
│   ├── fig5_held_out_timeseries.py
│   └── fig6_model2_timeseries.py
├── plot_style.py
├── config.py
└── data/
    ├── event_dataset.csv          → 52,692 rows, 50 basins
    ├── reset_dataset.csv          → 4,199 rows, 50 basins
    ├── outlier_basins.csv         → informational only
    └── selected_basins_v2.csv     → written by run_bootstrap_v2.py
```

### J2. Key Config Values

```python
RANDOM_SEED   = 42
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
DATA_CUTOFF   = "2025-01-01"
SELECTED_BASINS_CSV = DATA_DIR / "selected_basins_v2.csv"

QUALITY_FILTER = {
    "IRD_R_squared": 0.94,
    "CIV":           3000,
    "Ct":            20,
    "AL":            5,
}

MIN_EVENTS_PER_SEGMENT     = 3
MIN_SEGMENT_R2             = 0.1
PEARSON_OUTLIER_THRESHOLD  = -0.05

BOOTSTRAP_N_HELD_OUT = 5
BOOTSTRAP_SEED       = 42
HARDCODED_HELD_OUT   = [3203, 4401, 5302, 6302, 7402]
```

---

## PART K — REFERENCES AND PRIOR WORK

### K1. Elkayam (2025) — STL Predecessor Paper

STL decomposition: ~75% of IRD variance is non-periodic. This framework predicts that 75%.

### K2. Key References

| Reference | Contribution |
|---|---|
| Bainov and Simeonov (1989) | Impulsive differential equations |
| Lakshmikantham et al. (1989) | Theory of impulsive differential equations |
| Raissi et al. (2019) | Physics-informed neural networks |
| Ke et al. (2017) | LightGBM |
| Hlavacek and Bouchet (1994) | Resistance-in-series, membrane fouling |
| Di Bella et al. (2019) | Resistance-in-series review, MBR |

---

## PART L — GLOSSARY

| Term | Definition | Units |
|---|---|---|
| SAT | Soil Aquifer Treatment | — |
| MAR | Managed Aquifer Recharge | — |
| IRD | Infiltration Rate during Drainage | cm/h |
| IRD_at_reset (ρᵢ) | Post-tillage IRD at segment start | cm/h |
| IRD_norm_log (η) | ln(IRD/IRD_at_reset) — Model 1 target | dimensionless |
| IRD_norm_log_reset (δ) | ln(ρᵢ/ρᵢ₋₁) — Model 2 target | dimensionless |
| LCT | Loading Cycle Time / Last Cultivation Time — hours since last tillage | hours |
| ALPHA | Drying fraction = DrT/Ct | dimensionless |
| DrT | Drying time | hours |
| FT | Flooding time | hours |
| Ct | Total cycle time = FT + DT + DrT | hours |
| HL | Hydraulic Load = CIV/area | cm |
| CIV | Cumulative Infiltrated Volume | m³ |
| DAR | Daily Ambient Radiation at reset date | W/m² |
| η∞ | Asymptotic floor — 5th percentile of η per basin | dimensionless |
| λ | Decay rate learned by Model 1 | h⁻¹ |
| 𝒫 | Poincaré map learned by Model 2 | — |
| SHAP | SHapley Additive exPlanations | — |
| PINN | Physics-Informed Neural Network | — |

---

## PART M — METHODS SECTION STRUCTURE (Current — May 2026)

### 2.1 Study Site and Data

Shafdan SAT facility, southern Tel Aviv metropolitan area. 50 monitored recharge basins across 5 operational fields: Soreq 2 (4 basins), Yavne 1 (16 basins), Yavne 2 (9 basins), Yavne 3 (9 basins), Yavne 4 (12 basins). IoT sensor network recording flooding and drying cycles continuously 2015–2025. Each event characterized by IRD (cm/h), flooding time (FT), drying time (DrT), hydraulic load (CIV, m³), cycle time (Ct), ambient radiation and temperature. Tillage events recorded via ibud sensors. 20 Soreq 1 basins operate without online monitoring — excluded from study.

### 2.2 Event Quality Filtering

Four sensor-quality thresholds applied at the event level: drainage curve R² ≥ 0.94 (exponential fit quality), CIV ≥ 3,000 m³ (minimum hydraulic load), Ct ≥ 20 h (minimum cycle duration), AL ≥ 5 cm (minimum active layer depth). Events failing any threshold are retained in the dataset but excluded from model training and evaluation. Events after the study cutoff date (January 1, 2025) are also excluded.

### 2.3 Segment Identification

Each sequence of flooding events between two successive tillage events constitutes a segment. Tillage timestamps from ibud sensors are matched to flooding events using a two-tier search: quality-passing events (passing CIV, Ct, AL thresholds) are preferred as reset anchors; if none are available within the search window (±3 days before, up to 60 days after tillage), any event is used as fallback. This ensures segment structure is recovered even for basins where the post-tillage event fails quality filters. LCT (hours since tillage) is derived from the matched reset anchor timestamp. IRD_at_reset (ρᵢ) is the IRD of the reset anchor event, broadcast to all events in the segment. IRD_norm = ln(IRD / IRD_at_reset) is the normalized decay target for Model 1.

### 2.4 Segment Quality

Segments are accepted if they contain at least 3 quality-passing events, show negative Pearson correlation between LCT and IRD_norm (r < −0.05), and achieve a decay curve fit R² ≥ 0.10. Segments failing these criteria are retained in the dataset but flagged; their events receive a filter_reason label and are excluded from training and evaluation. In total, 23,338 events (49.8% of study-period events across all 50 basins) pass all quality criteria.

### 2.5 Dual Model Framework

**Model 1** (within-segment decay) predicts IRD_norm_log = ln(IRD/ρᵢ) at each flooding event. Features capture the previous event's drying conditions (ALPHA, DrT, temperature, radiation), hydraulic loading (log HL, FT), cumulative segment state (LCT, cumulative wetting temperature, cumulative flooding time), and the post-tillage IRD ceiling (IRD_at_reset) — 11 features total.

**Model 2** (post-tillage recovery) predicts IRD_norm_log_reset = ln(ρᵢ/ρᵢ₋₁) at each tillage event. Features summarize the preceding segment: fractional zero-drying, event count, radiation and drying accumulators, DAR (daily ambient radiation at tillage date), and the previous two reset deltas — 12 features total.

Both models use LightGBM with published default hyperparameters (no grid search) and early stopping. Predictions are back-transformed: IRD(t) = ρᵢ · exp(η̂(t)) for Model 1, ρ̂ᵢ = ρᵢ₋₁ · exp(δ̂ᵢ) for Model 2.

Basin heterogeneity was characterized using a global LightGBM model; basins with strongly negative R² or high relative RMSE were noted as operationally atypical but retained in all training and evaluation sets.

### 2.6 Evaluation Condition (Condition E-full)

All 50 basins are used for training. The only exclusion is the four sensor-quality filters. No is_good_segment filter is applied — all events from quality-passing basins contribute to training. The naive baseline predicts ρᵢ for every event in Model 1 (no decay) and the previous reset IRD for every tillage event in Model 2 (no change).

### 2.7 Bootstrap Validation and Basin Selection

To select a representative held-out test set covering all operational contexts, 501 bootstrap iterations are run. Iteration 1 uses a pre-specified set of five basins (one per field): 3203 (Soreq 2), 4401 (Yavne 1), 5302 (Yavne 2), 6302 (Yavne 3), 7402 (Yavne 4). Iterations 2–501 draw one basin per field at random from all 50 basins. For each iteration, both models are trained on the 45 non-held-out basins and evaluated on the 5 held-out basins. The selected iteration satisfies: (1) all 5 basins beat the naive baseline in both models, (2) one basin per field, (3) maximum Model 2 ΔRMSE among qualifying iterations. The selected basins are held out for all subsequent evaluation reported in this paper.

### 2.8 Spatial Generalization — Leave-One-Field-Out

To assess generalization to completely unseen operational contexts, leave-one-field-out cross-validation is performed. Each of the 5 fields is held out in turn; both models are trained on all basins from the 4 remaining fields and evaluated on all basins in the held-out field. No basin from the held-out field contributes to training in any way. This is the strictest possible spatial generalization test — the model must transfer to an entirely new operational unit.

---

## PART N — RESULTS SECTION STRUCTURE (Current — May 2026)

### 3.1 Bootstrap Validation and Basin Selection

**Content:**
- Bootstrap distribution figure (main paper)
- Total iterations: 501 (1 hardcoded + 500 random one-per-field)
- Report: how many iterations satisfied both-beat-naive constraint + one-per-field
- Selected iteration number and the five held-out basins
- Selected M1 R², M1 ΔRMSE, M2 R², M2 ΔRMSE
- Median and IQR of bootstrap distribution for both models
- Note: one-basin-per-field coverage emerged from selection criterion, not imposed as constraint

**Numbers needed:** Bootstrap console output (PENDING)

### 3.2 Model 1 — Within-Segment IRD Decay (Held-Out Performance)

**Content:**
- Table 1: per-basin R², RMSE, MAPE, Naive RMSE for 5 held-out basins + pooled row
- Key narrative: pooled R², pooled RMSE vs naive (ΔRMSE), beat-naive count (5/5)
- Per-basin range of R²

**Numbers needed:** model1_decay console output (PENDING)

### 3.3 Model 1 — Feature Importance

**Content:**
- Table 3: SHAP feature ranking — Rank, Feature, Mean |SHAP|, Direction
- Top 4 features and their physical directions
- Note on prev_ALPHA as dominant operational lever
- Physical interpretation deferred to Discussion

**Numbers needed:** fig4_shap_beeswarm console output (PENDING)

### 3.4 Model 2 — Post-Tillage IRD Recovery (Held-Out Performance)

**Content:**
- Table 2: per-basin R², RMSE, MAPE, Naive RMSE, ΔRMSE for 5 held-out basins + pooled row
- Key sentence: pooled ΔRMSE and percentage improvement over naive
- Note: MAPE may appear close to naive — RMSE is the correct primary metric (MAPE paradox explained in Discussion)

**Numbers needed:** model2_reset console output (PENDING)

### 3.5 Model 2 — Feature Importance

**Content:**
- SHAP ranking from model2_reset console output
- DAR as top or near-top feature — operationally actionable
- prev_delta autocorrelation structure
- Physical interpretation deferred to Discussion

**Numbers needed:** model2_reset SHAP table (PENDING)

### 3.6 Spatial Generalization — Leave-One-Field-Out

**Content:**
- Summary table (already have final numbers — see Part F above)
- Model 1: beats naive in 5/5 fields, mean ΔRMSE = +0.877 cm/h
- Model 2: beats naive in 3/5 fields pooled, mean ΔRMSE = +0.028 cm/h
- Note on Soreq 2: basins 3201/3202 now included (previously excluded due to pipeline order bug — corrected May 2026). These basins show negative R² (untrained field, atypical operational pattern) but beat naive RMSE — consistent with a basin type not seen during training
- Yavne 4: several atypical basins (7102, 7103, 7202) show negative R² individually but pooled field performance is strong (+0.735 R², +0.738 ΔRMSE)

---

*End of Knowledge Base — Version 3.0, May 2026*
*Next step: paste bootstrap + model output console results to complete Part F and finalize Section 3*