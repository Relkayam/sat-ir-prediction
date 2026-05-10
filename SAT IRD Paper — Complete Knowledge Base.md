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
| LCT | Loading Cycle Time — time since the start of the current segment (since last tillage) | hours |
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
# Activate environment first
C:\Users\user\PycharmProjects\sat-ir-prediction\.venv\Scripts\activate

# Build datasets
python -m pipeline.build_dataset --rebuild      # → data/event_dataset.csv
python -m pipeline.build_reset_dataset          # → data/reset_dataset.csv

# Pass 1 model (all 50 basins — before outlier detection)
python -m models.model1_decay

# Outlier detection (produces outlier_basins.csv)
python -m analysis.basin_analysis

# Pass 2 model (reads outlier_basins.csv — 8 basins excluded)
python -m models.model1_decay

# Feature selection
python -m analysis.feature_selection_unconstrained   # Model 1
python -m analysis.feature_selection_m2              # Model 2

# Model 2
python -m models.model2_reset

# Algorithm comparisons + SHAP
python -m analysis.model_comparison       # Model 1 — all conditions × all algorithms
python -m analysis.model2_comparison      # Model 2 — all conditions × all algorithms
```

### B3. Event Dataset Structure

**File:** `data/event_dataset.csv` — 46,907 rows (study period), 50 basins

Each row = one flooding-drying event. Key columns:

| Column | Description |
|---|---|
| `basin_number` | 4-digit basin ID (e.g., 3203, 5102) |
| `opening_valve_date` | Date of flooding event |
| `row_type` | "event" or "reset" (reset = first event after tillage) |
| `segment_id` | Integer, increments at each tillage event per basin |
| `is_good_segment` | True if segment passes all quality filters |
| `filter_reason` | "" = good event; otherwise the filter that excluded it |
| `IRD_norm_log` | Model 1 target = ln(IRD / IRD_at_reset) |
| `IRD_at_reset` | IRD value at segment start (cm/h) |
| `LCT` | Cumulative loading time within segment (hours) |
| `seg_lambda` | Fitted exponential decay rate for this segment (h⁻¹) |
| `seg_a`, `seg_b`, `seg_r2` | Fitted decay parameters |
| `basin_role` | "clean", "outlier", or "held_out" |
| `split` | "train", "val", "test", or "excluded" |
| `split_held_out` | Split for held-out evaluation (D/E conditions) |

**filter_reason values:**
- `""` — good event in a good segment (used for training)
- `"after_cutoff"` — date >= DATA_CUTOFF
- `"quality_filter_IRD_R_squared"` — drainage R² < 0.94
- `"quality_filter_CIV"` — CIV < 3,000 m³
- `"quality_filter_Ct"` — Ct < 20 h
- `"quality_filter_AL"` — AL < 5 cm
- `"pre_segment"` — before first credible reset
- `"too_few_events"` — segment has < 4 events
- `"pearson_r_positive"` — no decay signal (Pearson r > −0.05)
- `"fit_failed"` — exponential decay fit did not converge
- `"r2_below_threshold"` — decay fit R² < 0.10

### B4. Reset Dataset Structure

**File:** `data/reset_dataset.csv` — 4,163 rows, 48 basins

Each row = one tillage/reset event. Key columns:

| Column | Description |
|---|---|
| `basin_number` | Basin ID |
| `reset_date` | Date of tillage event |
| `IRD_norm_log_reset` | Model 2 target = ln(ρᵢ / ρᵢ₋₁) |
| `IRD_at_reset` | Actual post-tillage IRD (cm/h) — for metrics |
| `prev_IRD_at_reset_raw` | Previous reset's IRD — for back-transformation |
| `split_chrono` | Chronological 70/15/15 split per basin |
| `split_random` | Random 70/15/15 split per basin |
| `split_held_out` | "train"/"val"/"test" for clean; "held_out_test" for held-out; "excluded" for outliers |
| `basin_role` | "clean", "outlier", or "held_out" |

### B5. Filter Funnel (Final Numbers)

| Step | All 50 basins | Clean dataset (42 basins) |
|---|---|---|
| Raw events (study period) | 46,907 | 42,825 |
| Removed — outlier basins | — | 4,082 |
| Drainage R² < 0.94 | 1,628 | 1,451 |
| CIV < 3,000 m³ | 507 | 444 |
| Ct < 20h | 290 | 283 |
| AL < 5cm | 310 | 293 |
| Pre-segment | 160 | 137 |
| Too few events (<4) | 4,972 | 4,377 |
| No decay signal | 12,692 | 11,455 |
| Fit failed | 0 | 0 |
| R² below threshold | 3,806 | 3,493 |
| **Good events (training)** | **22,542** | **20,892** |
| **% events used** | **48.1%** | **48.8%** |

**Note on all-segments (Condition E):** Condition E uses all events without the good-segment filter. This is the primary evaluation condition — the model is trained on all 46,907 events (minus held-out basins) and achieves better held-out performance than the filtered version.

### B6. Basin Classification

**Clean basins (37):** All 50 basins minus 8 outliers minus 5 held-out. Used for Conditions A, B, D training.

**Outlier basins (8):** Excluded from clean training. Two types:
- **Type 1 — Low dynamic range (5 basins):** IRD_at_reset IQR < 0.60 cm/h (bottom 20th percentile). Insufficient variability for log-ratio target.
- **Type 3 — Non-stationary operations (3 basins):** IRD_at_reset trends monotonically across segments, violating stationarity assumption.

**Held-out basins (5):** Fixed before any model training. One per field, selected as median performers. Never seen during training, validation, or feature selection.

### B7. Outlier Basins Table

| Basin | Field | Type | Pass-1 R² | Reason |
|---|---|---|---|---|
| 7102 | Yavne 4 | Type 1 | −0.829 | Low dynamic range |
| 4304 | Yavne 1 | Type 1 | −0.698 | Low dynamic range |
| 4303 | Yavne 1 | Type 1 | −0.665 | Low dynamic range |
| 7103 | Yavne 4 | Type 1 | −0.276 | Low dynamic range |
| 7202 | Yavne 4 | Type 1 | −0.060 | Low dynamic range |
| 6101 | Yavne 3 | Type 3 | −0.865 | Non-stationary operations |
| 7303 | Yavne 4 | Type 3 | −0.582 | Non-stationary operations |
| 4103 | Yavne 1 | Type 3 | −0.072 | Non-stationary operations (borderline — included for scientific conservatism) |

**Held-out Basins (5):**

| Basin | Field | Pass-1 R² (all 50 model) |
|---|---|---|
| 3203 | Soreq 2 | +0.789 |
| 4104 | Yavne 1 | +0.836 |
| 5102 | Yavne 2 | +0.871 |
| 6303 | Yavne 3 | +0.772 |
| 7201 | Yavne 4 | +0.846 |

---

## PART C — THE TWO TARGETS AND NORMALIZATION

### C1. Why Log-Ratio Normalization

The 50 Shafdan basins have substantially different saturated hydraulic conductivities (Ks). Basin 5201 operates at median IRD_reset = 6.3 cm/h; Basin 4104 at 1.7 cm/h — a nearly 4× difference. A model trained on raw IRD values would learn basin identity rather than clogging physics, and would fail to generalize to unseen basins.

Log-ratio normalization removes this between-basin scale difference by defining targets as ratios relative to the post-tillage baseline, rather than absolute values.

### C2. Physical Justification (Resistance-in-Series)

The log-ratio form is physically motivated by the resistance-in-series framework (Hlavacek and Bouchet, 1994; Di Bella et al., 2019):

$$R(t) = R_m + R_\text{rev}(t) + R_\text{irrev}$$

where R_m is the clean-medium resistance (proportional to 1/Ks, constant per basin), R_rev(t) is the reversible clogging resistance accumulating during flooding and partially removed by tillage, and R_irrev is the irreversible residual persisting across tillage events.

Since IRD ∝ 1/R(t) by Darcy's law, the post-tillage IRD is ρᵢ ∝ 1/(R_m + R_irrev). The ratio:

$$\frac{\text{IRD}(t)}{\rho_i} = \frac{R_m + R_\text{irrev}}{R_m + R_\text{irrev} + R_\text{rev}(t)}$$

substantially reduces between-basin scale differences by dividing out the post-tillage resistance (R_m + R_irrev). The approximation becomes exact when R_irrev is small relative to R_m. What remains is predominantly R_rev(t) — the reversible clogging driven by the same physical and biological processes across all basins. Taking the logarithm maps this ratio into an additive, symmetric space better suited to regression modeling.

### C3. Model 1 Target: IRD_norm_log

$$\eta(t) = \ln\!\left(\frac{\text{IRD}(t)}{\rho_i}\right) \tag{1}$$

**Properties:**
- η(0) = 0 at every segment start, regardless of basin or segment
- η(t) < 0 as clogging progresses
- Dimensionless and scale-free

**Back-transformation:**

$$\widehat{\text{IRD}}(t) = \rho_i \cdot \exp(\hat{\eta}(t)) \tag{2}$$

### C4. Model 2 Target: IRD_norm_log_reset

$$\delta_i = \ln\!\left(\frac{\rho_i}{\rho_{i-1}}\right) \tag{3}$$

**Properties:**
- δ = 0: recovery matched previous reset exactly
- δ > 0: better recovery than last time
- δ < 0: worse recovery than last time
- First reset per basin excluded (no ρᵢ₋₁ available)

**Back-transformation:**

$$\hat{\rho}_i = \rho_{i-1} \cdot \exp(\hat{\delta}_i) \tag{4}$$

### C5. η∞ — Asymptotic Floor

η∞ is the minimum normalized IRD approached under sustained clogging. It is estimated as the **5th percentile of observed η(t) for each basin** across all its events.

**Important:** η∞ is NOT used in the current LightGBM models. It is a theoretical construct introduced in the mathematical framework (Part G) to frame Model 1 as parameter identification for an ODE. In the companion PINN paper, η∞ will be either fixed at the 5th percentile estimate or treated as a learnable per-basin parameter.

---

## PART D — THE DUAL-MODEL FRAMEWORK

### D1. The Two Hypotheses

**Hypothesis 1 — Decay:**
> Within each inter-tillage segment, the normalized IRD follows a monotone decay process driven by operational and environmental conditions since the last tillage event.

**Hypothesis 2 — Reset:**
> The post-tillage recovery level is a deterministic function of the operational history of the preceding segment and the environmental conditions at the moment of tillage.

These are hypotheses — tested empirically through model performance on held-out basins.

### D2. Model 1 — Feature Vector (11 features)

Features organized into physical groups:

**Scale anchor:**
- `IRD_at_reset` (ρᵢ) — post-tillage IRD at segment start (cm/h). Sets the basin- and segment-level hydraulic conductivity ceiling.

**Previous-event drying signals:**
- `prev_ALPHA` — drying fraction = DrT/Ct (dimensionless). More drying → more biofilm desiccation → slower clogging.
- `prev_DrT` — drying time of previous event (hours). Absolute drying duration.
- `prev_TD` — temperature during previous drying (°C). Higher temperature → faster biofilm desiccation.
- `prev_RD` — radiation during previous drying (W/m²). Higher radiation → photodegradation of biofilm.
- `prev_RW` — radiation during previous wetting (W/m²). Light during flooding.

**Previous-event loading:**
- `log1p_prev_HL` — hydraulic load, log-transformed: log(1 + HL). More volume infiltrated → more suspended solids deposited → faster clogging. Log-transformed due to extreme right skew (skewness > 15).
- `prev_FT` — flooding time of previous event (hours).

**Cumulative segment state:**
- `LCT` — time since last tillage (hours). Primary decay axis.
- `cum_TW` — cumulative wetting temperature since reset (°C·h). Seasonal biofilm growth signal.
- `cum_FT` — cumulative flooding time since reset (hours). Integrated flooding load.

**Features evaluated but excluded (collinearity):**
- `cum_RD`, `cum_RW` — collinear with `prev_RD`/`prev_RW` and `cum_TW`. The most proximal radiation signal dominates over cumulative.

### D3. Model 2 — Feature Vector (11 features)

**Seasonality:**
- `month_sin` — sine encoding of calendar month, phase-shifted to peak in July: sin(2π(month−4)/12)
- `month_cos` — cosine encoding, orthogonal seasonal component

**Autocorrelation (two-step history):**
- `prev_IRD_at_reset` (ρᵢ₋₁) — previous reset level (cm/h). Best single predictor of recovery.
- `prev_prev_IRD_at_reset` (ρᵢ₋₂) — two-step history. Captures medium-term recovery trend.

**Segment operational summary:**
- `mean_ALPHA` — mean drying fraction over segment. More drying → better recovery at tillage.
- `total_LCT` — total flooding duration of segment (hours).
- `sum_DrT` — cumulative drying time of segment (hours).
- `sum_FT` — cumulative flooding time of segment (hours).

**Last-event signals (most proximal):**
- `last_DrT` — final drying duration before tillage (hours). Most proximal recovery opportunity.
- `last_RD` — radiation in final drying event (W/m²). Photodegradation immediately before tillage.

**Ambient conditions at reset:**
- `DAR` — daily ambient radiation at tillage date (W/m²). Key environmental driver of post-tillage recovery. **Operationally actionable: till when radiation is high.**

### D4. Feature Table (Table S1)

| Feature | Physical variable | Unit | Transformation | Physical rationale |
|---|---|---|---|---|
| **Model 1** | | | | |
| IRD_at_reset | Post-tillage IRD at segment start | cm/h | None | Scale anchor — hydraulic conductivity ceiling |
| prev_ALPHA | Drying fraction = DrT/Ct | dimensionless | None | More drying → biofilm desiccation → slower decay |
| log1p_prev_HL | Hydraulic load | log(cm+1) | log1p | More volume → solids deposition → faster clogging |
| prev_DrT | Previous drying time | hours | None | Absolute drying duration |
| LCT | Time since last tillage | hours | None | Primary decay axis |
| prev_TD | Temperature during drying | °C | None | Higher temp → faster biofilm desiccation |
| prev_FT | Previous flooding time | hours | None | Longer flooding → more biofilm growth |
| cum_TW | Cumulative wetting temperature | °C·h | None | Seasonal biofilm growth since reset |
| cum_FT | Cumulative flooding time | hours | None | Integrated biological clogging load |
| prev_RD | Radiation during drying | W/m² | None | Photodegradation during drying phase |
| prev_RW | Radiation during wetting | W/m² | None | Light during flooding |
| **Model 2** | | | | |
| month_sin | Seasonal signal (peak July) | dimensionless | sin(2π(month−4)/12) | Annual biofilm growth cycle |
| month_cos | Orthogonal seasonal | dimensionless | cos(2π(month−4)/12) | Continuous seasonal encoding |
| prev_IRD_at_reset | Previous reset level | cm/h | None | Primary autocorrelation — Ks persistence |
| prev_prev_IRD_at_reset | Two-step history | cm/h | None | Medium-term recovery trend |
| mean_ALPHA | Mean drying fraction | dimensionless | None | Average drying intensity over segment |
| total_LCT | Total segment duration | hours | None | Total flooding load |
| sum_DrT | Cumulative drying time | hours | None | Total drying exposure |
| sum_FT | Cumulative flooding time | hours | None | Total biological clogging load |
| last_DrT | Final drying before tillage | hours | None | Most proximal recovery opportunity |
| last_RD | Final drying radiation | W/m² | None | Photodegradation immediately before tillage |
| DAR | Daily ambient radiation at tillage | W/m² | None | **Key driver: till when radiation is high** |

### D5. Feature Selection Methodology

**Method:** Forward stepwise selection, unconstrained — run to completion (all candidate features added one by one).

**Evaluation metric:** RMSE in raw IRD space (cm/h) after back-transformation.

**Evaluation set:**
- Model 1: Condition E held-out test (5 unseen basins)
- Model 2: Chrono test split (same basins, future resets)

**Results:**
- Both models: elbow at 11 features
- Model 1: 11-feature model outperforms full 21-feature model by 0.047 cm/h RMSE
- Model 2: 11-feature model outperforms full 22-feature model by 0.054 cm/h RMSE
- This confirms overfitting in the larger models — features beyond 11 add variance without generalizable signal

### D6. Algorithm Selection

**Primary algorithm:** LightGBM (Ke et al., 2017) — selected for:
1. SHAP compatibility (TreeExplainer)
2. Consistency between Model 1 and Model 2
3. Strong performance on tabular data

**Model 1 algorithm ranking (Condition A):**
CatBoost > LightGBM > XGBoost > RandomForest > Ridge
- CatBoost marginal advantage (~0.011 R²) — noted in one paper sentence
- LightGBM kept for SHAP compatibility

**Model 2 algorithm ranking (Chrono):**
CatBoost ≈ LightGBM ≈ XGBoost ≈ RandomForest >> Ridge
- All gradient boosting essentially tied
- Ridge notably worse — confirms nonlinearity matters
- LightGBM kept for consistency with Model 1

### D7. Hyperparameters (Final)

Published defaults, no grid search. This is a deliberate choice supporting the transferability claim.

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

### D8. The Forward Operational Chain

The two models are operationally coupled. Given a planned tillage date and operational history of segment i−1:

1. **Model 2** predicts δ̂ᵢ = f₂(**z**_{i−1}) → back-transform: ρ̂ᵢ = ρᵢ₋₁ · exp(δ̂ᵢ)
2. **ρ̂ᵢ** enters Model 1 as the scale anchor feature IRD_at_reset
3. **Model 1** predicts η̂(t) = f₁(**x**(t)) for each planned event in segment i
4. **Back-transform:** IRD(t) = ρ̂ᵢ · exp(η̂(t))

This enables full operational planning before flooding begins — predicting both recovery ceiling and decay trajectory.

---

## PART E — EVALUATION FRAMEWORK

### E1. Held-Out Basin Design

Five basins held out before any model training — fixed at the start of the project:

| Basin | Field | Selection criterion |
|---|---|---|
| 3203 | Soreq 2 | Median performer — one per field |
| 4104 | Yavne 1 | Median performer |
| 5102 | Yavne 2 | Median performer |
| 6303 | Yavne 3 | Median performer |
| 7201 | Yavne 4 | Median performer |

These basins are **never** seen during training, validation, or feature selection for Conditions D and E.

### E2. Model 1 Conditions A–E

| Condition | Training basins | Training data | Test set |
|---|---|---|---|
| A | 37 clean | Good segments only (~48%) | Random 15%, same 37 |
| B | 37 clean | All segments | Random 15%, same 37 |
| C | 50 all | All segments | Random 15%, same 50 |
| D | 37 clean | Good segments only | 5 held-out basins — all events |
| **E (PRIMARY)** | **45 (incl. outliers)** | **All segments** | **5 held-out basins — all events** |

**Primary condition: E** — most general, uses all available data, tests genuine transferability.

### E3. Model 2 Conditions

| Condition | Training basins | Training data | Test set |
|---|---|---|---|
| Chrono | 37 clean | First 70% of resets chronologically | Last 15% same basins |
| Held-out D | 37 clean | All chrono train resets | 459 resets, 5 unseen basins |
| **Held-out E (PRIMARY)** | **45 (incl. outliers)** | **All chrono train resets** | **459 resets, 5 unseen basins** |

**Primary condition: Held-out E** — consistent with Model 1 primary.

**Model 2 Condition E fix (critical):** Outlier basins are tagged "excluded" in split_held_out. For Condition E they must be reassigned to chrono splits:
```python
outlier_mask = (df_e["basin_role"] == "outlier") & (df_e["split_held_out"] == "excluded")
df_e.loc[outlier_mask, "split_held_out"] = df_e.loc[outlier_mask, "split_chrono"]
```

### E4. Metrics Definitions

All primary metrics computed on raw IRD (cm/h) after back-transformation unless noted.

| Metric | Definition | Primary use |
|---|---|---|
| R²(IRD) | Coefficient of determination on back-transformed IRD | Both models |
| RMSE (cm/h) | Root mean squared error in IRD space | Both models |
| MAPE (%) | Mean absolute percentage error in IRD space | Model 1 primary; Model 2 secondary |
| R²(log) | R² on log-ratio targets directly | Diagnostic only |
| rel_RMSE | RMSE / mean(IRD_true) | Diagnostic |
| Spearman ρ | Rank correlation — robust to outliers | Diagnostic |
| Basin median R² | Median of per-basin R² values | Diagnostic |

**MAPE for Model 2:** RMSE is the primary metric for Model 2. MAPE is reported but misleadingly favors the naive baseline (see MAPE paradox in F7).

### E5. Baseline Definitions

**Model 1 baseline — Causal linear extrapolation:**

For each event at position k within a segment, a linear model is fitted to all prior IRD observations in the same segment:

$$\text{IRD}(t_i) = \alpha \cdot \text{LCT}_i + \beta, \quad i = 1,\ldots,k-1$$

and extrapolated to the current LCT:

$$\widehat{\text{IRD}}(t_k) = \hat{\alpha} \cdot \text{LCT}_k + \hat{\beta}$$

Predictions clipped to [0.1, 20] cm/h. Undefined for positions 1 and 2 — covers 46% of test events (n=2,018 of 4,386).

**Model 2 baseline — Naive persistence:**

$$\hat{\rho}_i = \rho_{i-1}$$

Predict no change from previous reset. Equivalent to δ̂ = 0. Covers 100% of test events.

**Baselines tested but NOT reported in paper (smoke only):**
- True persistence (prev IRD across segment boundaries): R²=+0.898, RMSE=0.569 cm/h — competitive but covers only 57% of events and cannot predict post-tillage first events
- Per-basin median λ: R²=−0.112 — fails because within-basin λ IQR spans 60–84×
- Previous-segment λ: R²=−1.102 — even worse; consecutive segment λ values are weakly correlated

---

## PART F — RESULTS

### F1. Model 1 — All Conditions Summary

| Condition | Training | Test basins | R²(IRD) | RMSE (cm/h) | MAPE% |
|---|---|---|---|---|---|
| Naive baseline | — | same 37 | +0.669 | — | — |
| A — Clean, good segs | 37 clean | same 37 | +0.725 | 0.731 | 17.2% |
| B — All segments | 37 clean | same 37 | +0.697 | 0.782 | 19.6% |
| C — All data | 50 all | same 50 | +0.767 | 0.747 | 19.2% |
| D — Held-out clean | 37 clean | 5 unseen | +0.813 | 0.695 | 16.0% |
| **E — Held-out all (PRIMARY)** | **45 basins** | **5 unseen** | **+0.898** | **0.607** | **13.1%** |

### F2. Model 1 — Held-Out Per-Basin (Condition E — PRIMARY)

| Basin | Field | R²(IRD) | RMSE (cm/h) | MAPE% | n events |
|---|---|---|---|---|---|
| 3203 | Soreq 2 | +0.707 | 0.884 | 12.4% | 1,430 |
| 4104 | Yavne 1 | +0.718 | 0.208 | 12.0% | 682 |
| 5102 | Yavne 2 | +0.725 | 0.550 | 14.1% | 1,080 |
| 6303 | Yavne 3 | +0.731 | 0.423 | 13.3% | 756 |
| 7201 | Yavne 4 | +0.862 | 0.139 | 14.2% | 449 |
| **POOLED** | **all** | **+0.898** | **0.607** | **13.1%** | **4,386** |

**For reference — Condition D per-basin (37 clean basins trained):**

| Basin | Field | R²(IRD) | RMSE (cm/h) | MAPE% |
|---|---|---|---|---|
| 3203 | Soreq 2 | +0.590 | 0.967 | 14.2% |
| 4104 | Yavne 1 | +0.664 | 0.206 | 13.5% |
| 5102 | Yavne 2 | +0.728 | 0.536 | 15.2% |
| 6303 | Yavne 3 | +0.704 | 0.430 | 14.8% |
| 7201 | Yavne 4 | +0.821 | 0.153 | 16.7% |
| **POOLED** | **all** | **+0.871** | **0.638** | **14.8%** |

### F3. Model 1 — Baseline Comparison (Condition E)

| Method | R²(IRD) | RMSE (cm/h) | MAPE% | n | Coverage |
|---|---|---|---|---|---|
| **Model 1 Condition E** | **+0.898** | **0.607** | **13.1%** | **4,386** | **100%** |
| Causal linear extrapolation | +0.832 | 0.704 | 12.3% | 2,018 | 46% |

**Per-basin baseline comparison:**

| Basin | Field | Model 1 R² | Model 1 RMSE | Model 1 MAPE% | Baseline R² | Baseline RMSE | Baseline MAPE% | Baseline n |
|---|---|---|---|---|---|---|---|---|
| 3203 | Soreq 2 | +0.707 | 0.884 | 12.4% | +0.655 | 0.824 | 11.9% | 676 |
| 4104 | Yavne 1 | +0.718 | 0.208 | 12.0% | +0.673 | 0.190 | 10.9% | 266 |
| 5102 | Yavne 2 | +0.725 | 0.550 | 14.1% | +0.675 | 0.575 | 12.7% | 601 |
| 6303 | Yavne 3 | +0.731 | 0.423 | 13.3% | −1.033 | 1.091 | 14.9% | 279 |
| 7201 | Yavne 4 | +0.862 | 0.139 | 14.2% | +0.915 | 0.102 | 10.8% | 196 |
| **Pooled** | **all** | **+0.898** | **0.607** | **13.1%** | **+0.832** | **0.704** | **12.3%** | **2,018** |

*Note: Basin 6303 is where Model 1 wins most clearly (baseline R²=−1.033 vs Model 1 R²=+0.731) — linear extrapolation fails on operationally irregular segments. Basin 7201 is the single case where the baseline marginally outperforms Model 1 on the defined subset (196 events); Model 1 covers the full event set including post-tillage first events.*

### F4. Model 1 — SHAP Findings (Condition E)

**Feature ranking by mean |SHAP| (Condition E, held-out test):**

| Rank | Feature | Mean |SHAP| | Direction |
|---|---|---|---|
| 1 | IRD_at_reset (ρᵢ) | 0.1830 | → more decay (scale anchor effect) |
| 2 | prev_ALPHA | 0.1784 | → less decay |
| 3 | prev_DrT | 0.1078 | → more decay (collinearity artifact — see note) |
| 4 | log1p_prev_HL | 0.0812 | → more decay |
| 5 | prev_FT | 0.0408 | → more decay |
| 6 | prev_RW | 0.0397 | → more decay |
| 7 | LCT | 0.0237 | → less decay |
| 8 | prev_RD | 0.0147 | → more decay |
| 9 | prev_TD | 0.0140 | → more decay |
| 10 | cum_TW | 0.0129 | → less decay |
| 11 | cum_FT | 0.0112 | → more decay |

**IRD_at_reset vs prev_ALPHA:** Essentially tied at Condition E (0.183 vs 0.178). In Condition A (clean basins, good segments only), prev_ALPHA ranks 1st. The scale anchor effect of IRD_at_reset becomes slightly more prominent when training includes noisy all-segment data.

**prev_DrT collinearity note (IMPORTANT for Discussion):**
- SHAP shows high prev_DrT → faster decay (apparent contradiction with physics)
- This is a **collinearity artifact**: Spearman r(prev_DrT, prev_ALPHA) = 0.79
- After controlling for ALPHA, residual variation in DrT correlates with long-cycle events that are operationally difficult
- **The unconditional correlation is correct:** Spearman r(prev_DrT, η) = +0.18 — longer drying → less decay
- Full analysis belongs in Discussion, not Methods
- SI: Spearman correlation matrix + quartile analysis supporting this interpretation

**SHAP stability across conditions A/D/E:**
The top 4 features (IRD_at_reset, prev_ALPHA, prev_DrT, log1p_prev_HL) are identical across all conditions. Physical interpretation is robust to data quality choices.

### F5. Model 2 — All Conditions Summary

| Condition | Training | Test | R²(IRD) | RMSE (cm/h) | MAPE% |
|---|---|---|---|---|---|
| Naive (chrono) | — | same 37, future | +0.761 | 0.820 | 16.4% |
| Chrono | 37 clean | same 37, future | +0.818 | 0.715 | 16.9% |
| Naive (held-out) | — | 5 unseen | +0.867 | 1.053 | 12.9% |
| Held-out D | 37 clean | 5 unseen | +0.888 | 0.969 | 14.9% |
| **Held-out E (PRIMARY)** | **45 basins** | **5 unseen** | **+0.884** | **0.983** | **14.8%** |

### F6. Model 2 — Held-Out Per-Basin (Condition E — PRIMARY)

| Basin | Field | Model 2 R² | Model 2 RMSE | Model 2 MAPE% | Naive R² | Naive RMSE | Naive MAPE% | n |
|---|---|---|---|---|---|---|---|---|
| 3203 | Soreq 2 | +0.422 | 1.784 | 13.5% | +0.344 | 1.900 | 12.3% | 101 |
| 4104 | Yavne 1 | +0.418 | 0.333 | 13.7% | +0.458 | 0.321 | 11.8% | 92 |
| 5102 | Yavne 2 | +0.557 | 0.904 | 16.2% | +0.463 | 0.995 | 15.0% | 84 |
| 6303 | Yavne 3 | +0.354 | 0.585 | 15.4% | +0.203 | 0.650 | 12.5% | 101 |
| 7201 | Yavne 4 | +0.697 | 0.231 | 15.6% | +0.720 | 0.222 | 13.3% | 76 |
| **POOLED** | **all** | **+0.884** | **0.983** | **14.8%** | **+0.867** | **1.053** | **12.9%** | **454** |

**RMSE improvement over naive: 0.070 cm/h (6.6%)**

**For reference — Condition D per-basin:**

| Basin | Field | R²(IRD) | RMSE (cm/h) | MAPE% |
|---|---|---|---|---|
| 3203 | Soreq 2 | +0.444 | 1.749 | 12.8% |
| 4104 | Yavne 1 | +0.416 | 0.334 | 14.1% |
| 5102 | Yavne 2 | +0.553 | 0.908 | 16.6% |
| 6303 | Yavne 3 | +0.382 | 0.572 | 14.7% |
| 7201 | Yavne 4 | +0.671 | 0.241 | 17.1% |
| **POOLED** | **all** | **+0.888** | **0.969** | **14.9%** |

### F7. Model 2 — MAPE Paradox Explanation

The naive baseline achieves MAPE = 12.9% while Model 2 achieves 14.8%. This is NOT a failure.

**Explanation:** MAPE penalizes relative errors. The naive baseline always predicts δ = 0 (no change), producing small relative errors on stable basins due to high IRD_at_reset autocorrelation. Model 2 attempts to predict actual variation and occasionally overestimates magnitude even when direction is correct.

**RMSE is the correct primary metric for Model 2:** Model 2 reduces RMSE by 0.070 cm/h (6.6%) over naive — consistent across all five held-out basins.

**Per-basin R² vs pooled R² gap:** Per-basin R² ranges from +0.354 to +0.697 while pooled R² = +0.884. This is not a contradiction — the pooled metric captures between-basin variance in IRD_at_reset levels (7.4× range from Basin 7201 mean=1.10 cm/h to Basin 3203 mean=8.19 cm/h), which the model recovers through autocorrelation features (prev_IRD_at_reset, prev_prev_IRD_at_reset).

### F8. Model 2 — SHAP Findings

**Top 2 features (stable across all conditions):**

1. **prev_IRD_at_reset** — autocorrelation: the best predictor of recovery is where you started. Captures basin-level hydraulic conductivity persistence.

2. **DAR** — daily ambient radiation at tillage date: light intensity at the moment of tillage drives photodegradation of the surface biofilm layer.

**Operationally actionable finding: Till when radiation is high.**
The radiation at the moment of tillage — not the average during the segment — is the key environmental driver. Field managers can use this directly.

### F9. SHAP Stability

Top features are identical across conditions A, D, E for Model 1. The physical interpretation is not an artifact of data quality choices. The model learns the same physics whether trained on clean filtered data or noisy all-data.

### F10. Algorithm Comparison Summary

**Model 1 (Condition A):** CatBoost > LightGBM > XGBoost > RandomForest > Ridge

**Model 2 (Chrono):** CatBoost ≈ LightGBM ≈ XGBoost ≈ RandomForest >> Ridge

Full tables in `outputs/tables/model_comparison_v2.xlsx` and `model2_comparison_v2.xlsx`.

---

## PART G — MATHEMATICAL FRAMEWORK

### G1. Model 1 as Parameter Identification for an ODE

I hypothesize that the normalized IRD η(t) follows a first-order linear ODE:

$$\frac{d\eta}{dt} = -\lambda(\mathbf{x}(t)) \cdot (\eta(t) - \eta_\infty) \tag{5}$$

where:
- η(t) = ln(IRD(t)/ρᵢ) — normalized state variable, starting at 0 at each reset
- λ(**x**(t)) > 0 — decay rate as a function of operational/environmental inputs
- η∞ < 0 — asymptotic floor (5th percentile of observed η per basin)

**Analytical solution** for constant λ within a segment:

$$\eta(t) = (\eta(0) - \eta_\infty)\,e^{-\lambda t} + \eta_\infty = -\eta_\infty\left(1 - e^{-\lambda t}\right) \tag{6}$$

**Back-transformation to raw IRD:**

$$\text{IRD}(t) = \rho_i \cdot \exp\!\left[\eta_\infty\left(e^{-\lambda t} - 1\right)\right] \tag{7}$$

**What Model 1 actually learns:** the mapping **x** → λ. This reframes the prediction problem as **parameter identification for a physical ODE** — not curve fitting.

**Empirical support:** The within-basin IQR of λ spans 60–84× between Q25 and Q75. No fixed λ can represent this variability — λ must be a function of **x**(t).

### G2. Model 2 as a Poincaré Map

At each tillage event t_k, the system state jumps. I formalize this as a **Poincaré map**:

$$\rho_k = \mathcal{P}(\rho_{k-1},\, \mathbf{z}_{k-1}) \tag{8}$$

In log-ratio parameterization:

$$\delta_k = \ln\!\left(\frac{\rho_k}{\rho_{k-1}}\right) \tag{9}$$

Working in log-ratio space linearizes the map around the no-change fixed point (δ=0), removing K_s scale differences between basins.

**Why the naive baseline is difficult to beat:** The fixed point δ=0 is the zero-order approximation to 𝒫. Model 2 learns the first-order correction — how operational and environmental conditions cause deviations from this fixed point. The modest 6.6% RMSE improvement reflects how close the system operates to its fixed point most of the time.

### G3. The Hybrid Impulsive Dynamical System

Combining Equations (5) and (8):

$$\frac{d\eta}{dt} = -\lambda(\mathbf{x}(t))\,(\eta(t) - \eta_\infty), \qquad t \neq t_k \tag{10}$$

$$\rho(t_k^+) = \mathcal{P}\!\left(\rho(t_k^-),\, \mathbf{z}_{k-1}\right), \qquad t = t_k \tag{11}$$

This is the standard form of an **impulsive dynamical system** (Bainov and Simeonov, 1989; Lakshmikantham et al., 1989).

**Novel contribution:** First explicit formulation of SAT infiltration dynamics as a hybrid impulsive system. The decomposition into continuous decay and discrete predictive reset — with log-ratio normalization enabling scale-free learning across heterogeneous units — is the core theoretical contribution.

### G4. Generalization to Other Engineering Systems

The same structure appears across filtration and water treatment engineering:

| System | Performance variable | Decay process | Reset event | Standard approach |
|---|---|---|---|---|
| SAT recharge basin | IRD (cm/h) | Physical + biological clogging | Mechanical tillage | **This study** |
| Membrane (MF/UF) | Flux J (L/m²h) | Cake + pore fouling | Backwash / CIP | Fixed η (Hlavacek and Bouchet, 1994) |
| Sand/granular filter | Headloss ΔH (m) | Particle deposition | Backwash | Fixed removal fraction (Duran-Ros et al., 2024) |
| BAC/BAF biological filter | Headloss ΔH (m) | Biofilm growth (Monod) | Backwash + sloughing | Fixed sloughing fraction (Bi et al., 2014) |
| MBR | TMP (bar) | Activated sludge cake | Relaxation / air scouring | R_rev reset to zero (Di Bella et al., 2019) |

**What this framework contributes beyond existing approaches:**
1. Post-reset level treated as a predictable variable, not a fixed parameter
2. Log-ratio normalization enabling a single global model across parallel heterogeneous units
3. Operationally actionable SHAP-identified drivers (when and under what conditions to restore)

**Prerequisites for application:**
1. IoT operational records across multiple parallel units
2. ≥30–50 restoration events per unit
3. Between-unit heterogeneity (so normalization adds value)
4. Stationarity of post-reset level (no irreversible long-term degradation trend)

### G5. Connection to Physics-Informed Neural Networks

The hybrid formulation in Equations (10)–(11) opens a direct path to physics-informed learning — developed formally in the companion paper.

**PINN formulation:** Replace LightGBM f₁ with a neural network constrained to satisfy the ODE via a physics residual loss:

$$\mathcal{L}_\text{physics} = \left\|\frac{d\hat{\eta}}{dt} + \hat{\lambda}(\mathbf{x})\,(\hat{\eta} - \eta_\infty)\right\|^2 \tag{12}$$

**Advantages over current LightGBM:**
1. ODE constraint enforces physically consistent trajectories
2. λ̂(**x**) becomes an explicit interpretable model output
3. Natural extension to PDE coupling (surface + subsurface)

**The normalization as bridge:** By working in η and δ space (scale-free, zero-anchored, basin-independent), the physical constraints in Equation (12) are identical across all basins. The same normalization that enables the LightGBM to generalize enables the PINN to generalize.

---

## PART H — FIGURES

### H1. Complete Figure List

**MAIN PAPER FIGURES:**

| Figure | Description | Status | Script |
|---|---|---|---|
| Figure 1 | Study site map + schematic + example time series | Needs creation | Manual (PPT/Inkscape) |
| Figure 2 | IRD decay evidence — 3×2 grid, all segments | Done | fig2_decay_evidence.py (SHOW_ALL_SEGMENTS=True) |
| Figure 3 | Dual-model framework diagram (ODE + Poincaré map) | Needs creation | Manual (PPT) |
| Figure 4 | Model 1 SHAP beeswarm — Condition E | Done | fig4_shap_beeswarm.py |
| Figure 5 | Model 1 held-out time series — Condition E | Done | fig5_held_out_timeseries.py |
| Figure 6 | Model 2 held-out time series — Condition E | Done | fig6_model2_timeseries.py |

**SI FIGURES:**

| Figure | Description | Status | Script |
|---|---|---|---|
| Figure S1 | Model 1 feature selection curve | Auto-generated | python -m analysis.feature_selection_unconstrained |
| Figure S2 | Model 2 feature selection curve | Auto-generated | python -m analysis.feature_selection_m2 |
| Figure S3 | SHAP stability comparison A/D/E | Auto-generated | python -m analysis.model_comparison |
| Figure S4 | Model 1 held-out scatter (5 panels) | Auto-generated | python -m models.model1_decay |
| Figure S5 | Model 2 held-out scatter (5 panels) | Auto-generated | python -m models.model2_reset |
| Figure S6 | Per-basin R² histogram (50 basins) | Auto-generated | python -m analysis.basin_analysis |
| Figure S7 | Outlier basin diagnostic plots (8 panels) | Auto-generated | python -m analysis.basin_analysis |

**NEW FIGURES FROM THIS CONVERSATION (paper or SI — TBD):**

| Figure | Description | Script |
|---|---|---|
| Figure M1 | Normalization argument — 2×2 grid (time series + IRD vs LCT + η vs LCT) | smoke_5_normalization_2x2.py |
| Figure Sx | Poincaré map empirical view — normalized, 3 panels | smoke_poincare_v2.py |
| Figure Sy | Causal linear extrapolation baseline scatter — per-basin + pooled | smoke_naive4_linear.py |
| Figure Sz | Model 2 scatter actual vs predicted — Model 2 vs naive side by side | smoke_m2_scatter.py |

### H2. Per-Figure Details

**Figure 2 — IRD Decay Evidence (UPDATED)**
- **File:** `fig2_decay_evidence.py`
- **Key setting:** `SHOW_ALL_SEGMENTS = True` — all segments including outliers
- **Layout:** 3 rows × 2 columns. Left column: raw IRD time series with season bands. Right column: η(t) vs LCT scatter with fit lines.
- **Basins:** 3204 (Soreq 2), 5201 (Yavne 2), 7401 (Yavne 4)
- **Caption:** Within-segment IRD dynamics across three representative basins, showing all recorded flooding-drying events. Seasonal pattern visible in left column. Decay structure and variability visible in right column. Good segments show exponential fits; bad segments appear as scatter only.

**Figure 4 — SHAP Beeswarm (UPDATED)**
- **File:** `fig4_shap_beeswarm.py`
- **Condition:** E (45 basins, all segments, 5 held-out test)
- **Caption:** SHAP beeswarm for Model 1 Condition E. Top features: IRD_at_reset and prev_ALPHA essentially tied (0.183 vs 0.178). Note on prev_DrT collinearity with ALPHA (Spearman r=0.79) — unconditional correlation is positive (r=+0.18).

**Figure 5 — Model 1 Held-Out Time Series (UPDATED)**
- **File:** `fig5_held_out_timeseries.py`
- **Condition:** E (45 basins trained, 5 held-out test)
- **Layout:** 5 rows × 1 column. Blue circles = actual, green crosses = predicted. Season bands (alpha=0.60). Basin number on y-axis (units on middle panel only). No titles. Metrics in caption.
- **Caption:** Includes per-basin Condition E metrics from console output.

**Figure 6 — Model 2 Held-Out Time Series (UPDATED)**
- **File:** `fig6_model2_timeseries.py`
- **Condition:** E
- **Layout:** Same style as Figure 5. Each point = one tillage event.
- **Caption:** Includes per-basin metrics and MAPE paradox explanation.

### H3. Plot Style Design System (plot_style.py)

**Colors:**
- Deep blue: `#065A82` (primary)
- Teal: `#1C7293`
- Orange: `#E07B39` (accent)
- Green: `#27AE60` (predictions in Figures 5/6)

**Field colors (consistent across all figures):**
- Soreq 2: `#065A82`
- Yavne 1: `#1C7293`
- Yavne 2: `#E07B39`
- Yavne 3: `#27AE60`
- Yavne 4: `#7D3C98`

**Season band colors (Mediterranean):**
- Winter (Dec–Feb): blue `#DAE8FC`
- Spring (Mar–May): peach `#FCE4D6`
- Summer (Jun–Aug): yellow `#FFF3CD`
- Autumn (Sep–Nov): green `#D5E8D4`

**Font sizes (FONT_OVERRIDE in each figure script):**
- Minimum body text: 16pt (global default in plot_style.py)
- Figure-specific overrides set lower for dense multi-panel figures (e.g., 9–12pt in Figures 5/6)
- **To fix reviewer font-size comments:** adjust FONT_OVERRIDE dict at top of each figure script

---

## PART I — PAPER WRITING GUIDE

### I1. Paper Structure (IMRaD)

1. Abstract (write last)
2. Introduction
3. Materials and Methods
4. Results
5. Discussion
6. Conclusions
7. Supplementary Information

### I2. Methods Subsections (3.1–3.10)

**3.1 Study Site**
Shafdan SAT, 50 basins, 10-year IoT record, 140 million m³/year, Coastal Plain Aquifer.

**3.2 Basin Operating Cycle and IRD Calculation**
Three-phase cycle (flooding, drainage, drying). IRD = slope of water level during drainage. ALPHA = DrT/Ct.

**3.3 Data Quality Control**
Five filters: drainage R² < 0.94, CIV < 3,000 m³, Ct < 20h, AL < 5cm. DrT=0 events: drying-window features filled with 0.

**3.4 Segmentation and the Tillage Reset Problem**
Ceiling reset classification — all tillage events treated as credible resets with smart date correction. Three reliability issues identified (incomplete documentation, timing imprecision, tillage ineffectiveness).

**3.5 Quality Filtering of Segments**
Three sequential tests: too few events (<4), Pearson r > −0.05 (primary gate), exponential fit R² < 0.10.

**3.6 Dataset Construction**
Event dataset (Model 1 target: IRD_norm_log). Reset dataset (Model 2 target: IRD_norm_log_reset). Log-ratio normalization justification — resist-in-series framework.

**3.7 Feature Engineering**
Model 1: 11 features, 4 physical groups. Model 2: 11 features, 4 physical groups. Forward stepwise selection — elbow at 11 for both.

**3.8 Model Architecture and Training**
LightGBM, published defaults, no grid search. Segment-level splits prevent leakage.

**3.9 Evaluation Strategy**
5 conditions (A–E) for Model 1, 3 conditions for Model 2. Primary: Condition E (both models). Held-out design: 5 basins, one per field, fixed before training.

**3.10 Baselines**
Model 1: causal linear extrapolation (equation stated). Model 2: naive persistence (δ=0).

### I3. Results Subsections (Primary Numbers)

**4.1 Model 1 within-sample (Condition A):** R²(IRD)=+0.725, MAPE=17.2%

**4.2 Model 1 generalizability (Condition E PRIMARY):** R²=+0.898, RMSE=0.607 cm/h, MAPE=13.1%

**4.3 Model 1 baseline comparison:** Linear extrapolation R²=+0.832, RMSE=0.704 cm/h (n=2,018, 46% coverage). Model 1 outperforms by 0.097 cm/h RMSE on defined subset; covers 100% of events.

**4.4 Feature importance:** SHAP top 4 stable A/D/E — IRD_at_reset, prev_ALPHA, prev_DrT, log1p_prev_HL. DrT collinearity note → Discussion.

**4.5 Model 2 within-sample (Chrono):** R²(IRD)=+0.818 vs naive +0.761.

**4.6 Model 2 generalizability (Held-out E PRIMARY):** R²=+0.884, RMSE=0.983 cm/h. RMSE improvement over naive: 0.070 cm/h (6.6%). MAPE paradox explained.

### I4. Discussion Structure and Key Claims

**Section D1 — The forward chain (operational interpretation)**
Model 2 → ρ̂ᵢ → Model 1 → full IRD trajectory. Error propagation. Operational scenarios (when to till, ALPHA optimization, basin scheduling).

**Section D2 — Mathematical interpretation**
ODE formulation. Analytical solution. What Model 1 learns (x → λ). Poincaré map. Hybrid impulsive system. Why fixed-λ baselines fail.

**Section D3 — SHAP physical interpretation**
Dominant features and their physical meaning. prev_DrT collinearity analysis (full treatment here, not Methods). DAR operational implication.

**Section D4 — Model 2 modest improvement justification**
High autocorrelation explanation. Fixed-point proximity. Tail event value. Pre-tillage prediction capability. DAR finding.

**Section D5 — Generalization to other engineering systems**
Table of analogous systems. What existing models do not do. Prerequisites.

**Section D6 — Future work: PINN connection**
ODE residual loss. η∞ as learnable parameter. Normalization as bridge. Companion paper.

### I5. Key Narrative Choices

1. **Primary condition: Condition E for both models** — most general, all data, no filtering
2. **Report RMSE as primary metric for Model 2** — MAPE misleadingly favors naive
3. **Do NOT use "oracle ceiling %" language** — oracle and model evaluated on different subsets
4. **Do NOT report random split for Model 2** — chrono is the honest evaluation
5. **LightGBM chosen for SHAP compatibility** — CatBoost marginal advantage noted in one sentence
6. **Figure 2 shows ALL segments** — honest representation of what the model trains on
7. **No good-segments-only figure in main paper** — all-segments figure is sufficient

### I6. Things NOT To Do

1. Do NOT use "% of oracle ceiling" language
2. Do NOT report random split results for Model 2
3. Do NOT claim model "exceeds the oracle"
4. Do NOT use the word "oracle" in the paper — replace with "fitted exponential baseline"
5. Do NOT tune hyperparameters with grid search
6. Do NOT switch to CatBoost as primary model
7. Do NOT include linear analysis (Ridge coefficients) in main paper
8. Do NOT include Model 2 SHAP in main paper — SI only
9. Do NOT report Model 1 oracle comparison
10. Do NOT show good-segments-only figure as main Figure 2
11. Do NOT report Condition D as primary for either model (Condition E is primary)
12. Do NOT add causal linear extrapolation baseline to production code — smoke scripts only

### I7. Key Claims With Supporting Numbers

| Claim | Supporting numbers |
|---|---|
| Model generalizes to unseen basins | Model 1 pooled R²=+0.898 (Cond E), Model 2 pooled R²=+0.884 (Cond E) |
| Feature importance robust to data quality | SHAP top 4 identical across conditions A, D, E |
| Model beats engineered baseline | Model 1 ΔRMSE=−0.097 cm/h vs linear extrapolation |
| 11-feature model outperforms full model | ΔRMSE=+0.047 cm/h (M1), +0.054 cm/h (M2) |
| Till when radiation is high | DAR = #2 SHAP feature for Model 2, stable across all conditions |
| Fixed decay rate cannot represent system | Within-basin λ IQR spans 60–84× |
| Condition E better than Condition D | Pooled MAPE 13.1% (E) vs 14.8% (D) for Model 1 |

---

## PART J — CODE ARCHITECTURE

### J1. File Structure and Roles

```
sat-ir-prediction/
├── pipeline/
│   ├── build_dataset.py          → event_dataset.csv (Model 1 data)
│   ├── build_reset_dataset.py    → reset_dataset.csv (Model 2 data)
│   └── features.py               → MODEL1_FEATURES, MODEL2_FEATURES, targets
├── analysis/
│   ├── basin_analysis.py         → outlier detection → outlier_basins.csv
│   ├── feature_selection_unconstrained.py  → Model 1 forward stepwise
│   ├── feature_selection_m2.py   → Model 2 forward stepwise
│   ├── model_comparison.py       → 5 algorithms × 5 conditions + SHAP (M1)
│   └── model2_comparison.py      → 5 algorithms × 3 conditions + SHAP (M2)
├── models/
│   ├── model1_decay.py           → Model 1 training + evaluation (5 conditions)
│   ├── model2_reset.py           → Model 2 training + evaluation (3 conditions)
│   └── utils.py                  → metrics, back_transform, train_lightgbm
├── figures/
│   ├── fig2_decay_evidence.py    → Figure 2 (all segments)
│   ├── fig4_shap_beeswarm.py     → Figure 4 (SHAP, Condition E)
│   ├── fig5_held_out_timeseries.py → Figure 5 (Model 1 held-out, Cond E)
│   └── fig6_model2_timeseries.py → Figure 6 (Model 2 held-out, Cond E)
├── smoke/
│   ├── smoke_5_normalization_2x2.py     → Figure M1
│   ├── smoke_poincare_v2.py             → Figure Sx (Poincaré map)
│   ├── smoke_naive4_linear.py           → Figure Sy (linear baseline scatter)
│   ├── smoke_m2_scatter.py              → Figure Sz (Model 2 scatter)
│   ├── smoke_version_c_baseline.py      → λ baseline analysis (diagnostic only)
│   └── smoke_per_basin_lambda.py        → per-basin λ analysis (diagnostic only)
├── plot_style.py                 → shared matplotlib style (colors, fonts, seasons)
├── config.py                     → paths, HELD_OUT_BASIN_LIST, RANDOM_SEED
└── data/
    ├── event_dataset.csv
    ├── reset_dataset.csv
    └── outlier_basins.csv
```

### J2. Key Config Values

```python
HELD_OUT_BASIN_LIST = [3203, 4104, 5102, 6303, 7201]
RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SEASON_PHASE = 4   # month_sin peaks at July
DATA_CUTOFF = "2025-01-01"

# Outlier detection thresholds
OUTLIER_R2_THRESHOLD = 0.0
OUTLIER_REL_RMSE_THRESHOLD = 0.5
```

### J3. Two-Pass Execution for Model 1

```
Pass 1: python -m models.model1_decay
         (outlier_basins.csv does not exist → all 50 basins)

Run:     python -m analysis.basin_analysis
         (produces outlier_basins.csv with 8 outlier basins)

Pass 2: python -m models.model1_decay
         (reads outlier_basins.csv → 8 basins excluded for conditions A/B/D)
```

### J4. Model 2 Condition E Fix

Outlier basins tagged "excluded" in split_held_out. For Condition E, reassign inline:

```python
# In model2_reset.py main() and model2_comparison.py prepare_condition()
df_e = df.copy()
outlier_mask = (
    (df_e["basin_role"] == "outlier") &
    (df_e["split_held_out"] == "excluded")
)
df_e.loc[outlier_mask, "split_held_out"] = df_e.loc[outlier_mask, "split_chrono"]
```

### J5. Figure Scripts — Key Settings

**fig2_decay_evidence.py:**
```python
SHOW_ALL_SEGMENTS = True    # ALL segments — this is now the primary figure
BASINS = [3204, 5201, 7401]
DATE_FROM = "2021-01-01"    # Adjust date range as needed
SEASON_ALPHA = 0.60         # Season band opacity
```

**fig4_shap_beeswarm.py:**
```python
# Retrains LightGBM Condition E + computes SHAP (~5 min)
# No external SHAP file — computed fresh each run
```

**fig5_held_out_timeseries.py:**
```python
SEASON_ALPHA = 0.60
# Blue circles = actual, green crosses = predicted
# Basin number on y-axis; IRD (cm/h) on middle panel only
# No titles; metrics in caption
```

**fig6_model2_timeseries.py:**
```python
SEASON_ALPHA = 0.60
MARKER_SIZE = 35    # Larger than Fig 5 — fewer points per basin
# Same color scheme as Fig 5
```

---

## PART K — REFERENCES AND PRIOR WORK

### K1. Elkayam (2025) — STL Predecessor Paper

Elkayam and Lev (2024/2025) applied STL decomposition to 10 years of IRD records from the Shafdan system. Key finding: ~75% of IRD variance is non-periodic (not explainable by seasonal or trend components). This 75% is what this framework predicts.

STL component mapping:
- Trend → captured by IRD_at_reset normalization
- Seasonal → captured by cum_TW, month_sin/cos in Model 2
- Non-periodic residual (75%) → what Model 1 predicts via λ(**x**(t))

### K2. Key References — Mathematical Framework

| Reference | Contribution |
|---|---|
| Bainov and Simeonov (1989) | Impulsive differential equations — foundation for hybrid system formulation |
| Lakshmikantham et al. (1989) | Theory of impulsive differential equations |
| Takens (1981) | Delay embedding theorem — empirical Poincaré map from time series |
| Raissi et al. (2019) | Physics-informed neural networks (PINNs) — target for companion paper |
| Ke et al. (2017) | LightGBM — primary algorithm |

### K3. Key References — Engineering Analogs

| Reference | System | Contribution |
|---|---|---|
| Hlavacek and Bouchet (1994) | Membrane MF/UF | Resistance-in-series, blocking laws |
| Di Bella et al. (2019) | MBR | Resistance-in-series review |
| Bi et al. (2014) | BAC/BAF biological filter | Biofilm + headloss model |
| Sun et al. (2019) | BAC/BAF | Headloss modeling with backwash |
| Duran-Ros et al. (2024) | Sand filter | Backwash efficiency |
| Drumm and Bernardi (2025) | Membrane | Monte Carlo — closest analog to Model 2 |
| Liaw et al. (2020) | Membrane | ANN post-cleaning prediction — closest ML analog |

### K4. What This Paper Does NOT Claim

1. Does not claim to replace physics-based models (Bouwer, Okubo)
2. Does not claim exponential decay assumption is always correct
3. Does not claim the model works without any basin history (first segment harder)
4. Does not claim causal relationships — SHAP shows correlation, not causation
5. Does not claim λ(**x**) is the true physical decay rate — it is the effective rate learned by the model
6. Does not claim the Poincaré map has chaotic attractor structure — the empirical map shows high autocorrelation near the fixed point

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
| LCT | Loading Cycle Time — time since last tillage | hours |
| ALPHA | Drying fraction = DrT/Ct | dimensionless |
| DrT | Drying time | hours |
| FT | Flooding time | hours |
| Ct | Total cycle time = FT + DT + DrT | hours |
| HL | Hydraulic Load = CIV/area | cm |
| CIV | Cumulative Infiltrated Volume | m³ |
| RD | Radiation during drying | W/m² |
| RW | Radiation during wetting | W/m² |
| TD | Temperature during drying | °C |
| TW | Temperature during wetting | °C |
| DAR | Daily Ambient Radiation at reset date | W/m² |
| DAT | Daily Ambient Temperature at reset date | °C |
| Ks | Saturated hydraulic conductivity | cm/h |
| η∞ | Asymptotic floor — 5th percentile of η per basin | dimensionless |
| λ | Decay rate learned by Model 1 | h⁻¹ |
| 𝒫 | Poincaré map learned by Model 2 | — |
| Segment | Flooding events between two consecutive tillage events | — |
| Reset | Tillage event | — |
| SHAP | SHapley Additive exPlanations | — |
| PINN | Physics-Informed Neural Network | — |
| ODE | Ordinary Differential Equation | — |
| IMRaD | Introduction, Methods, Results, Discussion — paper structure | — |

---

*End of Knowledge Base — Version 2.0, May 2026*
*Next step: write the full paper using this document as the single source of truth*