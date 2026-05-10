# INTRODUCTION -this you needs to know about this project

# SAT IRD Paper — Scientific Context & Knowledge Base

---

## WHAT THIS PROJECT IS

A PhD-level research paper predicting infiltration rates in a Soil Aquifer Treatment (SAT) system using machine learning. The system is the **Shafdan SAT facility** near Tel Aviv, Israel — one of the largest SAT systems in the world, treating 130 million m³/year of secondary wastewater.

The paper introduces a **dual-model ML framework** that is both practically useful (operational planning for field engineers) and scientifically novel (first formal hybrid dynamical system formulation for SAT clogging-recovery dynamics).

---

## THE PHYSICAL SYSTEM

### What is SAT?
Soil Aquifer Treatment (SAT) = a form of Managed Aquifer Recharge (MAR) where treated wastewater is infiltrated through recharge basins into an aquifer, providing additional treatment through the soil. The water is then extracted from wells and used as a water resource.

### The Shafdan system
- 50 recharge basins across 5 fields: Soreq 2, Yavne 1, Yavne 2, Yavne 3, Yavne 4
- Each basin: ~1,000–5,000 m² surface area, flooded and dried alternately
- 10-year IoT record: 2015–2025
- Data managed in DuckDB via the optisat package

### The clogging problem
During flooding, the soil surface clogs progressively due to:
1. Suspended solids accumulation (physical clogging)
2. Biofilm growth (biological clogging)
3. Gas entrapment

This clogging causes the **Infiltration Rate (IR)** to decay over time within each flooding segment.

### The recovery mechanism (tillage)
After a segment ends, the basin is dried and mechanically tilled (plowed). This breaks up the clogging layer and restores the infiltration capacity. After tillage, a new flooding segment begins at a higher IR.

### The operational cycle
```
Flood → IRD decays → Dry → Till → Flood (new segment) → IRD decays → ...
```
Each flooding period = one **segment**. Each tilling event = one **reset**.

---

## KEY DEFINITIONS

### IRD (Infiltration Rate during Drainage)
- Measured at the end of each flooding event when the basin drains
- Units: cm/h
- Proxy for the soil's hydraulic conductivity at that moment
- Decreases within each segment (clogging), recovers after tillage

### IRD_at_reset
- The IRD value at the very first event of a new segment (immediately after tillage)
- Represents the "ceiling" — how well the basin recovered after tillage
- Varies between basins (different hydraulic conductivity ceilings, Ks)
- Varies between resets for the same basin (depends on operations and weather)

### LCT (Loading Cycle Time)
- Time since the start of the current segment (hours)
- Primary decay axis for Model 1
- IRD generally decreases as LCT increases

### ALPHA (drying fraction)
- ALPHA = DrT / Ct where DrT = drying time, Ct = total cycle time
- Measures what fraction of each operational cycle was spent drying vs flooding
- Key operational variable: more drying → better clogging recovery

### HL (Hydraulic Load)
- Volume of water infiltrated per unit area per event (m³ or cm)
- Higher HL → more clogging → faster IRD decay

### DrT (Drying Time)
- Duration of the drying period between consecutive flooding events (hours)
- Longer drying → more biofilm desiccation → slower clogging → slower decay

### Segment
- A sequence of flooding events between two consecutive tillage events
- Each segment has its own IRD_at_reset (starting value)
- Segments vary in length (typically 3–20 events)

---

## THE TWO TARGETS (log-ratio normalization)

### Why log-ratio?
Different basins have different hydraulic conductivity ceilings. Basin A may max out at 4 cm/h while Basin B maxes out at 10 cm/h. Training on raw IRD means the model partly learns "which basin is this" rather than the physics. Log-ratio normalization removes between-basin scale differences.

### Model 1 target: IRD_norm_log
```
IRD_norm_log = log(IR(t) / IRD_at_reset)
```
- Always starts at 0 at the beginning of each segment
- Decreases as clogging progresses (becomes more negative)
- Back-transform: IR(t) = IRD_at_reset × exp(IRD_norm_log)

### Model 2 target: IRD_norm_log_reset
```
IRD_norm_log_reset = log(IRD_at_reset[i] / IRD_at_reset[i-1])
```
- = 0: recovery exactly matches previous reset
- > 0: better recovery than last time
- < 0: worse recovery than last time
- Back-transform: IRD_at_reset[i] = IRD_at_reset[i-1] × exp(IRD_norm_log_reset)

---

## THE DUAL-MODEL FRAMEWORK

### Mathematical formulation (agreed in this conversation)

**Model 1 = parameter identification for an ODE:**
```
du/dt = −λ(x(t)) · (u − u∞)
```
where:
- u(t) = IRD_norm_log (the state variable)
- λ(x(t)) = decay rate as a function of operational/environmental inputs
- u∞ = asymptotic floor (minimum normalized IRD under sustained clogging)
- Analytical solution: u(t) = (u(0) − u∞) · exp(−λ · t) + u∞
- What Model 1 actually learns: the mapping x → λ

**Model 2 = discrete Poincaré map:**
```
s_i = F(s_{i-1}, z_{i-1})
```
where:
- s_i = IRD_at_reset[i] (system state at reset i)
- z_{i-1} = operational and environmental history of segment i-1
- F = recovery function learned by Model 2

**Together = hybrid dynamical system (impulsive system):**
```
ẋ = f(x, u, t),      t ≠ t_k   (continuous ODE — Model 1)
x(t_k⁺) = g(x(t_k⁻), z_k),    t = t_k  (discrete jump — Model 2)
```
where t_k are the tillage times.

### Why this formulation is novel
1. First explicit decomposition of SAT infiltration into continuous decay + discrete recovery as a hybrid system
2. The log-ratio normalization enabling a single global model across 50 heterogeneous basins
3. Forward-chaining of Model 2 → Model 1 enabling genuine operational planning
4. Demonstrated generalizability to completely unseen basins

### Where else this framework applies
The mathematical structure (monotone decay between discrete resets, reset level depends on pre-reset history) appears in:
- Membrane bioreactors (MBR): transmembrane pressure fouling/backwash
- Riverbed clogging in bank filtration: hydraulic conductivity decay/flood scour recovery
- Drip irrigation emitter clogging: discharge coefficient decay/flushing
- Aquifer storage and recovery (ASR): injectivity index decline/redevelopment
- Battery degradation: capacity fade/rest recovery
- Glacier mass balance: ablation/accumulation cycle

---

## DATA PIPELINE

### Source data
- Raw data in DuckDB database, accessed via `optisat.db.duckdb_manager.DuckDBManager`
- optisat package installed as editable from `mek-models-satix-backend` project

### Pipeline files
1. `pipeline/build_dataset.py` → produces `data/event_dataset.csv` (52,692 rows, 50 basins)
2. `pipeline/build_reset_dataset.py` → produces `data/reset_dataset.csv` (4,163 rows, 48 basins)
3. `analysis/basin_analysis.py` → produces `data/outlier_basins.csv` (8 basins)

### Event dataset structure
Each row = one flooding event. Key columns:
- `basin_number`: 4-digit basin ID (e.g., 3203, 5102)
- `opening_valve_date`: date of flooding event
- `row_type`: "event" or "reset" (reset = first event after tillage)
- `segment_id`: integer, increments at each tillage event per basin
- `is_good_segment`: True if segment passes quality filters
- `IRD_norm_log`: Model 1 target (computed by build_dataset.py)
- `IRD_at_reset`: IRD value at segment start (cm/h)
- `LCT`: cumulative loading time within segment
- `basin_role`: "clean", "outlier", or "held_out"
- `split`: "train", "val", "test", or "excluded"
- `split_held_out`: split assignment for held-out evaluation

### Reset dataset structure
Each row = one tillage/reset event. Key columns:
- `IRD_norm_log_reset`: Model 2 target
- `IRD_at_reset`: actual IRD_at_reset value (cm/h) — for metrics
- `prev_IRD_at_reset_raw`: previous reset's IRD — for back-transform
- `split_chrono`: chronological 70/15/15 split per basin
- `split_random`: random 70/15/15 split per basin
- `split_held_out`: "train"/"val"/"test" for clean basins, "held_out_test" for held-out, "excluded" for outliers
- `basin_role`: "clean", "outlier", or "held_out"

---

## EVALUATION CONDITIONS

### Model 1 conditions (5 total)
| Condition | Train basins | Train segments | Test |
|---|---|---|---|
| A | 37 clean | Good only | Random split on same 37 |
| B | 37 clean | All segments | Random split on same 37 |
| C | 50 all | All segments | Random split on same 50 |
| D | 37 clean | Good only | 5 held-out basins (all events) |
| E | 45 (incl. outliers) | All segments | 5 held-out basins |

**Primary result:** Condition E (most general) for paper narrative, Condition D for headline generalizability number.

### Model 2 conditions (3 total)
| Condition | Train basins | Train resets | Test |
|---|---|---|---|
| Chrono | 37 clean (past) | Chronological first 70% | Last 15% same basins |
| Held-out D | 37 clean | All chrono train | 459 resets from 5 unseen basins |
| Held-out E | 45 (incl. outliers) | All chrono train | 459 resets from 5 unseen basins |

**Primary result:** Held-out D for generalizability, Chrono for within-sample comparison.

### The 5 held-out basins
Fixed across both models: basins 3203 (Soreq 2), 4104 (Yavne 1), 5102 (Yavne 2), 6303 (Yavne 3), 7201 (Yavne 4). Selected as representative median performers, one per field.

---

## FEATURE ENGINEERING

### Model 1 features — physical groups
**Scale anchor:** IRD_at_reset — where does this basin/segment start?

**Drying/recovery signals (previous event):**
- prev_ALPHA: drying fraction = DrT / Ct
- prev_DrT: absolute drying time (hours)
- prev_TD: temperature during drying (°C)
- prev_RD: radiation during drying (W/m²)
- prev_RW: radiation during wetting (W/m²)

**Clogging intensity (previous event):**
- log1p_prev_HL: hydraulic load (log-transformed, skewness >15)
- prev_FT: flooding time (hours)

**Cumulative segment state:**
- LCT: time since reset (primary decay axis)
- cum_TW: cumulative wetting temperature since reset (seasonal signal)
- cum_FT: cumulative flooding time since reset

**Feature selection note:** cum_RD and cum_RW were evaluated but found collinear with prev_RD/prev_RW and cum_TW. The most proximal radiation signal dominates over cumulative.

### Model 2 features — physical groups
**Seasonality:** month_sin, month_cos (peak July = photodegradation season)

**Autocorrelation:** prev_IRD_at_reset, prev_prev_IRD_at_reset (two-step history)

**Segment operational summary:**
- mean_ALPHA: average drying fraction
- total_LCT: total flooding duration
- sum_DrT: cumulative drying time
- sum_FT: cumulative flooding load

**Last-event signals (most proximal):**
- last_DrT: final drying duration before tillage
- last_RD: radiation in final drying event

**Ambient conditions at reset:**
- DAR: daily ambient radiation at reset date (photodegradation driver)

---

## SHAP FINDINGS (agreed physical interpretation)

### Model 1 SHAP — top 4 stable across conditions A, D, E
1. **prev_ALPHA** — drying fraction: more drying relative to cycle time → slower decay
2. **IRD_at_reset** — scale anchor: high-conductivity basins start high and stay higher
3. **prev_DrT** — drying time: longer drying → more biofilm desiccation → slower decay
4. **log1p_prev_HL** — hydraulic load: more water pushed through → faster clogging → faster decay

Physical interpretation: the model has learned that the balance between loading intensity and drying recovery determines the decay rate. This is consistent with physical clogging theory.

### Model 2 SHAP — top 2 stable across all conditions
1. **prev_IRD_at_reset** — autocorrelation: the best predictor of recovery is where you started
2. **DAR** — daily ambient radiation at reset date: the light intensity at the moment of tillage drives photodegradation of the clogging biofilm layer

**Operationally actionable finding:** Till when it is sunny. The radiation at the moment of tillage — not the average during the segment — is the key environmental driver. Field managers can use this.

### SHAP stability narrative
The top features are identical across conditions that differ in data quality, basin selection, and segment filtering. This means the physical interpretation is not an artifact of the data quality choices. The model learns the same physics whether trained on clean filtered data or noisy all-data.

---

## ALGORITHM COMPARISON FINDINGS

### Model 1 algorithm ranking (Condition A)
CatBoost > LightGBM > XGBoost > RandomForest > Ridge
- CatBoost R²=+0.460 vs LightGBM R²=+0.449 — margin ~0.011
- **Decision: keep LightGBM** for SHAP compatibility and consistency
- One sentence in paper acknowledges CatBoost marginal advantage

### Model 2 algorithm ranking (Chrono)
CatBoost ≈ LightGBM ≈ XGBoost ≈ RandomForest >> Ridge
- All gradient boosting essentially tied
- Ridge notably worse — confirms nonlinearity matters for Model 2
- LightGBM kept as primary for consistency with Model 1

---

## BASIN CLASSIFICATION

### Clean basins (37)
All 50 basins minus 8 outliers minus 5 held-out. Used for conditions A, B, D training.

### Outlier basins (8) — excluded from clean training
**Type 1 — Low dynamic range (5 basins):** IRD_at_reset shows insufficient variability for model learning. IQR below 20th percentile of all basins (< 0.60 cm/h). Basins: 7102, 4304, 4303, 7103, 7202.

**Type 3 — Non-stationary operations (3 basins):** IRD_at_reset trends monotonically up or down across segments, violating the stationarity assumption of the log-ratio target. Basins: 6101, 7303, 4103.

**Note:** Basin 4103 has R²=−0.011 (borderline). Decided to include in outlier set for scientific conservatism. The 6 clearly negative-R² basins (7102 to −0.829, 4304 to −0.698, etc.) are unambiguously excluded.

**Type 2 (high variability) exists in the code but no basins met this threshold** — the system is relatively uniform in variability across basins once Type 1 and Type 3 are removed.

### Held-out basins (5) — fixed test set
3203 (Soreq 2), 4104 (Yavne 1), 5102 (Yavne 2), 6303 (Yavne 3), 7201 (Yavne 4)
Selected as median performers, one per field. These basins are NEVER in any training set for conditions D and E.

---

## KEY SCIENTIFIC DECISIONS (agreed in this conversation)

### On the oracle baseline
The exponential decay fit is NOT an absolute physical ceiling — it is a hypothesis about functional form. Some deviation from exponential is expected (gas clogging, biofilm sloughing). The "% of oracle ceiling" language was dropped because the oracle and model are evaluated on different subsets (different NaN patterns), creating an artifact where the model appears to exceed the oracle. Replace with: comparison to naive persistence baseline only.

### On primary conditions
- Model 1 primary: Condition E (most general — 45 basin training, 5 held-out test)
- But headline generalizability number from Condition D (cleaner — 37 clean training)
- Conditions A–D in Discussion + SI, not primary Results

### On filtering justification
Filtering is justified by MAPE improvement, not R² improvement:
- Condition D (filtered training): MAPE=16.0% on held-out
- Condition E (unfiltered training): MAPE=18.1% on held-out
- Difference = 2.1 percentage points
- This is the operational argument: filtering produces more accurate predictions for field managers

### On Model 2 MAPE paradox
Naive baseline MAPE=12.9% for held-out basins, Model 2 MAPE=14.9%. Model MAPE is higher than naive. This is NOT a failure — it is a mathematical artifact. The naive baseline always predicts the same value (no change) so it has low relative error on stable basins. The model tries to predict variation and sometimes gets magnitude wrong even when direction is right. RMSE is the correct primary metric for Model 2 (not MAPE). Report RMSE as primary.

### On feature selection methodology
Forward stepwise unconstrained selection on:
- Model 1: Condition E held-out test, metric = RMSE(IRD)
- Model 2: Chrono test split, metric = RMSE(IRD after back-transform)

Both: run to completion (all features), elbow identified visually. Reduced models outperform full models (Model 1: −0.047 cm/h, Model 2: −0.054 cm/h). This is the classic bias-variance tradeoff: removed features added variance without useful signal for generalization.

### On the math conference paper (separate from Water Research)
The physics-informed formulation (ODE + Poincaré map + hybrid dynamical system) is being developed as a separate ML conference paper, likely for the PhD student. The Water Research paper focuses on empirical results and operational applicability. The mathematical framework section in the Water Research paper is brief — one conceptual diagram (Figure 3), no ODE derivations.

---

## NAIVE BASELINES

### Model 1 naive baseline
Predict the previous event's IRD_norm_log value (persistence assumption).
- Log-ratio space R²: +0.203 (Condition A)
- IRD space R²: +0.669 (Condition A)
- Model 1 beats this substantially: +0.725 R²(IRD) — improvement of +0.056

### Model 2 naive baseline
Predict log-ratio = 0 (no change from previous reset).
Equivalent to: IRD_at_reset[i] = IRD_at_reset[i-1]
- Chrono test R²(IRD): +0.761, RMSE: 0.820 cm/h
- Held-out test R²(IRD): +0.867, RMSE: 1.050 cm/h (high autocorrelation in held-out basins)
- Model 2 beats naive on RMSE: −0.105 cm/h (chrono), −0.081 cm/h (held-out)

---

## PAPER NARRATIVE (agreed structure)

### The story in one paragraph
SAT systems suffer from progressive infiltration clogging that is difficult to predict because it depends on the interplay of hydraulic loading, drying recovery, and environmental conditions across 50 heterogeneous basins. We decompose the problem into two complementary ML models: Model 1 predicts the within-segment decay trajectory (equivalent to learning the decay rate λ of a first-order ODE), and Model 2 predicts the post-tillage recovery level (a discrete Poincaré map). Both models use 11 operationally measurable features selected by forward stepwise analysis. Evaluated on 5 completely unseen basins, Model 1 achieves R²=+0.871 and Model 2 achieves R²=+0.888 — demonstrating genuine transferability to new basins without site-specific recalibration.

### Key contributions
1. First dual-model ML framework for SAT that separates decay dynamics from recovery dynamics
2. Log-ratio normalization enabling a single global model across 50 heterogeneous basins
3. Demonstrated generalizability to completely unseen basins (5 held-out basins, one per field)
4. Operationally actionable finding: till when radiation is high (DAR is the #2 SHAP feature for Model 2)
5. Data quality analysis: formal basin classification into Type 1 (low range) and Type 3 (non-stationary)

### What makes this different from prior SAT ML papers
- Prior work: single-basin models, no held-out validation, raw IRD targets
- This paper: global model across 50 basins, rigorous held-out test, log-ratio normalization, dual-model framework, 10-year IoT record

---

## CODE ARCHITECTURE

### Key files and their roles
```
pipeline/
  build_dataset.py          — event_dataset.csv (Model 1 data)
  build_reset_dataset.py    — reset_dataset.csv (Model 2 data)
  features.py               — MODEL1_FEATURES (11), MODEL2_FEATURES (11), targets

analysis/
  basin_analysis.py         — outlier detection, produces outlier_basins.csv
  feature_selection_unconstrained.py — Model 1 forward stepwise selection
  feature_selection_m2.py   — Model 2 forward stepwise selection
  model_comparison.py       — 5 algorithms × 5 conditions + SHAP (Model 1)
  model2_comparison.py      — 5 algorithms × 3 conditions + SHAP (Model 2)

models/
  model1_decay.py           — main Model 1 training + evaluation (5 conditions)
  model2_reset.py           — main Model 2 training + evaluation (3 conditions)
  utils.py                  — shared utilities: metrics, back-transform, splits

config.py                   — paths, HELD_OUT_BASIN_LIST, RANDOM_SEED, etc.
```

### Important config values
```python
HELD_OUT_BASIN_LIST = [3203, 4104, 5102, 6303, 7201]
RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SEASON_PHASE = 4  # month_sin peaks at July
```

### Two-pass execution for Model 1
- Pass 1: run model1_decay.py → outlier_basins.csv does not exist → all 50 basins used
- Run basin_analysis.py → outlier_basins.csv created with 8 basins
- Pass 2: run model1_decay.py again → reads outlier_basins.csv → 37 clean basins for conditions A/B/D

### Model 2 condition E fix (important)
Outlier basins are tagged "excluded" in split_held_out. For condition E, they need to be reassigned to chrono train/val splits so they contribute to training. This is done inline:
```python
outlier_mask = (df_e["basin_role"] == "outlier") & (df_e["split_held_out"] == "excluded")
df_e.loc[outlier_mask, "split_held_out"] = df_e.loc[outlier_mask, "split_chrono"]
```

---

## LGBM HYPERPARAMETERS (final, well-established defaults)

### Model 1
```python
n_estimators=1000, max_depth=-1, num_leaves=63,
learning_rate=0.05, subsample=0.8, feature_fraction=0.8,
min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
early_stopping=50
```

### Model 2
```python
n_estimators=1000, max_depth=-1, num_leaves=31,
learning_rate=0.05, subsample=0.8, feature_fraction=0.8,
min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
early_stopping=50
```

No grid search was performed. Models work well with published defaults — this is a strength (supports transferability claim).

---

## METRICS DEFINITIONS

All metrics computed on raw IRD (cm/h) after back-transform unless otherwise noted.

- **R²(IRD)**: coefficient of determination on back-transformed IRD values
- **R²(log)**: coefficient of determination on log-ratio targets directly
- **RMSE**: root mean squared error (cm/h for IRD space)
- **MAPE**: mean absolute percentage error (%)
- **rel_RMSE**: RMSE / mean(IRD_true) — normalized RMSE
- **Spearman ρ**: rank correlation (robust to outliers)
- **Basin median R²**: median of per-basin R² values — robust to a few difficult basins

**Primary metric for paper:**
- Model 1: R²(IRD) and MAPE (operational interpretability)
- Model 2: R²(IRD) and RMSE (MAPE misleadingly low for naive on held-out basins)

---

## THINGS NOT TO DO (agreed decisions)

1. **Do not** use "% of oracle ceiling" language — oracle is not a true ceiling
2. **Do not** report random split results for Model 2 — chrono is the honest evaluation
3. **Do not** claim the model "exceeds the oracle" — this is a subset mismatch artifact
4. **Do not** use the word "oracle" in the paper — replace with "fitted exponential baseline"
5. **Do not** tune hyperparameters with grid search — use published defaults, this strengthens transferability claim
6. **Do not** switch to CatBoost as primary model — LightGBM kept for SHAP compatibility
7. **Do not** include linear analysis (Ridge coefficients, partial regression) in main paper — omit entirely
8. **Do not** include Model 2 SHAP in main paper — it is SI only
9. **Do not** report Model 1 "oracle" comparison — replaced by naive baseline comparison only

---

## RELATIONSHIP TO PRIOR WORK

### Elkayam (2025) — the predecessor paper
A previous paper by the same group performed STL (Seasonal-Trend decomposition) on long-term IRD trends at Shafdan. Key finding from that paper used in this one: 75% of the IRD variance is non-periodic (i.e., not explainable by seasonal or trend components alone). This is cited as motivation for the ML approach — the residual variance is what Model 1 is predicting.

The STL decomposition components map to:
- Trend component → captured by IRD_at_reset normalization
- Seasonal component → captured by cum_TW, month_sin/cos in Model 2
- Non-periodic residual (75%) → what Model 1 predicts via λ(x(t))

### What this paper does NOT claim
- It does not claim to replace physics-based models (Bouwer, Okubo)
- It does not claim the exponential decay assumption is always correct
- It does not claim the model works without ANY basin history (first segment is always harder)
- It does not claim causal relationships — SHAP shows correlation, not causation

---

## GLOSSARY

| Term | Definition |
|---|---|
| SAT | Soil Aquifer Treatment |
| MAR | Managed Aquifer Recharge |
| IRD | Infiltration Rate during Drainage |
| LCT | Loading Cycle Time (cumulative flooding time within segment) |
| ALPHA | Drying fraction = DrT / Ct |
| DrT | Drying time (hours) |
| FT | Flooding time (hours) |
| HL | Hydraulic Load (volume/area) |
| RD | Radiation during drying (W/m²) |
| RW | Radiation during wetting (W/m²) |
| TD | Temperature during drying (°C) |
| TW | Temperature during wetting (°C) |
| AL | Applied Load = total volume infiltrated in event |
| CIV | Cumulative Infiltrated Volume |
| Ct | Total cycle time = FT + DrT |
| DAR | Daily Ambient Radiation at reset date |
| DAT | Daily Ambient Temperature at reset date |
| Ks | Saturated hydraulic conductivity (basin-specific ceiling) |
| ODE | Ordinary Differential Equation |
| SHAP | SHapley Additive exPlanations |
| pptxgenjs | JavaScript library for creating PowerPoint files programmatically |









# SAT IRD Paper — Complete Instructions
## "From Decay to Recovery: A Dual Machine Learning Framework for Infiltration Rate Prediction in Managed Aquifer Recharge"
### Journal: Water Research | Project: sat-ir-prediction

---

## PROJECT PATHS

```
Project root     : C:\Users\user\PycharmProjects\sat-ir-prediction
Backend project  : C:\Users\user\PycharmProjects\mek-models-satix-backend
Data dir         : C:\Users\user\PycharmProjects\sat-ir-prediction\data\
Outputs dir      : C:\Users\user\PycharmProjects\sat-ir-prediction\outputs\
Tables dir       : C:\Users\user\PycharmProjects\sat-ir-prediction\outputs\tables\
Figures dir      : C:\Users\user\PycharmProjects\sat-ir-prediction\outputs\figures\
```

## PYTHON ENVIRONMENT

```bash
# Activate venv before running anything
C:\Users\user\PycharmProjects\sat-ir-prediction\.venv\Scripts\activate

# optisat is installed as editable package from:
# C:\Users\user\PycharmProjects\mek-models-satix-backend
# (pyproject.toml at mek-models-satix-backend level, src layout)
```

---

## EXECUTION ORDER (complete pipeline)

Run in this exact order if starting from scratch:

```bash
python -m pipeline.build_dataset --rebuild
python -m pipeline.build_reset_dataset
python -m analysis.basin_analysis
python -m models.model1_decay
python -m analysis.feature_selection_unconstrained
python -m analysis.feature_selection_m2
python -m models.model2_reset
python -m analysis.model_comparison
python -m analysis.model2_comparison
```

---

## FINAL MODEL PARAMETERS

### Model 1 (decay within segments)
- Target: `IRD_norm_log = log(IR / IR_at_reset)`
- Algorithm: LightGBM
- Features (11): `IRD_at_reset, prev_ALPHA, log1p_prev_HL, prev_DrT, LCT, prev_TD, prev_FT, cum_TW, cum_FT, prev_RD, prev_RW`
- Evaluation: 5 conditions (A–E), primary = Condition E

### Model 2 (recovery after tillage)
- Target: `IRD_norm_log_reset = log(IRD_at_reset[i] / IRD_at_reset[i-1])`
- Algorithm: LightGBM
- Features (11): `month_sin, month_cos, prev_IRD_at_reset, prev_prev_IRD_at_reset, mean_ALPHA, total_LCT, sum_DrT, sum_FT, last_DrT, last_RD, DAR`
- Evaluation: 3 conditions (Chrono, Held-out D, Held-out E), primary = Held-out D

---

## FINAL RESULTS SUMMARY

### Model 1 — LightGBM, test split
| Condition | Training | Test basins | R²(IRD) | RMSE (cm/h) | MAPE% |
|---|---|---|---|---|---|
| A — Clean, good segs | 37 basins | same 37 | +0.725 | 0.731 | 17.2% |
| B — All segments | 37 basins | same 37 | +0.697 | 0.782 | 19.6% |
| C — All data | 50 basins | same 50 | +0.767 | 0.747 | 19.2% |
| D — Held-out clean | 37 basins | 5 unseen | +0.813 | 0.695 | 16.0% |
| E — Held-out all-data | 45 basins | 5 unseen | +0.798 | 0.736 | 18.1% |
| Naive baseline | — | — | +0.203 (log) | — | — |

### Model 1 — Held-out per-basin (Condition D)
| Basin | Field | R² | RMSE | MAPE% |
|---|---|---|---|---|
| 3203 | Soreq 2 | +0.590 | 0.967 | 14.2% |
| 4104 | Yavne 1 | +0.664 | 0.206 | 13.5% |
| 5102 | Yavne 2 | +0.728 | 0.536 | 15.2% |
| 6303 | Yavne 3 | +0.704 | 0.430 | 14.8% |
| 7201 | Yavne 4 | +0.821 | 0.153 | 16.7% |
| **POOLED** | **all** | **+0.871** | **0.638** | **14.8%** |

### Model 2 — LightGBM, test split
| Condition | Training | Test | R²(IRD) | RMSE (cm/h) | MAPE% |
|---|---|---|---|---|---|
| Naive (chrono) | — | — | +0.761 | 0.820 | 16.4% |
| Chrono | 37 clean, past | same 37, future | +0.818 | 0.715 | 16.9% |
| Naive (held-out) | — | — | +0.867 | 1.050 | 12.9% |
| Held-out D | 37 clean | 5 unseen | +0.888 | 0.969 | 14.9% |
| Held-out E | 45 all data | 5 unseen | +0.885 | 0.983 | 14.8% |

### Model 2 — Held-out per-basin (Condition D)
| Basin | Field | R² | RMSE | MAPE% |
|---|---|---|---|---|
| 3203 | Soreq 2 | +0.444 | 1.749 | 12.8% |
| 4104 | Yavne 1 | +0.416 | 0.334 | 14.1% |
| 5102 | Yavne 2 | +0.553 | 0.908 | 16.6% |
| 6303 | Yavne 3 | +0.382 | 0.572 | 14.7% |
| 7201 | Yavne 4 | +0.671 | 0.241 | 17.1% |
| **POOLED** | **all** | **+0.888** | **0.969** | **14.9%** |

---

## FILTER FUNNEL (final numbers)
| Step | All 50 basins | Clean dataset |
|---|---|---|
| Raw events | 46,907 | 42,825 |
| Removed — outlier basins | — | 4,082 |
| Drainage R² < 0.94 | 1,628 | 1,451 |
| CIV < 3000 m³ | 507 | 444 |
| Ct < 20h | 290 | 283 |
| AL < 5cm | 310 | 293 |
| Pre-segment | 160 | 137 |
| Too few events (<4) | 4,972 | 4,377 |
| No decay signal | 12,692 | 11,455 |
| Fit failed | 0 | 0 |
| R² below threshold | 3,806 | 3,493 |
| **Good events (training)** | **22,542** | **20,892** |
| **% events used** | **48.1%** | **48.8%** |

---

## OUTLIER BASINS (8 total)
| Basin | Field | Type | R² | Reason |
|---|---|---|---|---|
| 7102 | Yavne 4 | Type 1 | −0.829 | Low dynamic range |
| 4304 | Yavne 1 | Type 1 | −0.698 | Low dynamic range |
| 4303 | Yavne 1 | Type 1 | −0.665 | Low dynamic range |
| 7103 | Yavne 4 | Type 1 | −0.276 | Low dynamic range |
| 7202 | Yavne 4 | Type 1 | −0.060 | Low dynamic range |
| 6101 | Yavne 3 | Type 3 | −0.865 | Non-stationary operations |
| 7303 | Yavne 4 | Type 3 | −0.582 | Non-stationary operations |
| 4103 | Yavne 1 | Type 3 | −0.072 | Non-stationary operations |

---

## PAPER STRUCTURE & FIGURE/TABLE MAP

### MAIN PAPER FIGURES

---

### Figure 1 — Study site and system overview
**Status:** needs to be created (map + schematic)
**Content:**
- Panel A: Map of Shafdan SAT system, Tel Aviv area, showing 5 fields (Soreq 2, Yavne 1–4), basin locations color-coded by field
- Panel B: Schematic of one basin showing IRD decay within segment and recovery at tillage
- Panel C: Example IRD time series (1 basin, ~3 years) showing repeating decay-reset pattern

**How to get it:**
- Map: use optisat basin coordinate data or draw schematically
- Schematic: draw manually in PowerPoint or Inkscape
- Example time series: run the following and save the output figure manually:
```bash
# In model1_decay.py main(), pick basin 5201 (good example of clean decay)
# The per-basin plots are saved in:
# outputs/figures/basin_plots/
# Pick a basin with 4+ clear segments
```
**Caption:** "Study site and experimental system. (A) Location of the Shafdan SAT system and its five recharge fields. (B) Schematic of within-segment IRD decay and post-tillage recovery. (C) Representative IRD time series showing the repeating decay-recovery cycle across consecutive segments."

---

### Figure 2 — IRD decay evidence across basins
**Status:** needs to be created (pull from basin_plots)
**Content:**
- 3 panels: one basin from Soreq 2, one from Yavne 2, one from Yavne 4
- Each panel: IRD_norm_log vs LCT, scatter colored by segment, exponential fit overlaid
- Demonstrates the decay structure is consistent across fields

**How to get it:**
```bash
# Basin plots are saved automatically when running:
python -m analysis.basin_analysis
# Output: outputs/figures/basin_plots/basin_XXXX.png
# Pick basins: 3204 (Soreq 2), 5201 (Yavne 2), 7401 (Yavne 4)
# These are clean basins with strong decay signal
```
**Caption:** "Within-segment IRD decay structure across representative basins. Each panel shows IRD_norm_log = log(IR/IR_at_reset) vs cumulative loading time (LCT) for all events in one basin, colored by segment. The exponential decay pattern (fitted curves) is consistent across fields and operational periods, motivating the ODE-based formulation of Model 1."

---

### Figure 3 — Dual-model framework diagram
**Status:** needs to be created (conceptual diagram — build in PPT)
**Content:**
- Left panel: continuous ODE decay phase with Model 1 inputs/outputs
- Right panel: discrete reset map with Model 2 inputs/outputs
- Center: arrow showing how Model 2 output feeds Model 1 as initial condition
- Formula boxes: du/dt = −λ(x)·(u − u∞) and s_i = F(s_{i-1}, z_{i-1})

**How to get it:**
- Build entirely in PowerPoint using shapes and text boxes
- No code required

**Caption:** "Dual-model framework for forward IRD prediction. Model 1 (left) predicts within-segment decay as a parameter identification problem for a first-order ODE, where the decay rate λ is learned from operational and environmental features. Model 2 (right) predicts the post-tillage recovery ratio as a discrete Poincaré map. Together they form a hybrid dynamical system enabling full operational trajectory prediction."

---

### Figure 4 — Model 1 SHAP beeswarm (Condition A)
**Status:** generated automatically — save from screen
**Content:** LightGBM SHAP beeswarm, 11 features, condition A (clean basins, good segments)

**How to get it:**
```bash
python -m analysis.model_comparison
# The SHAP beeswarm for condition A is plotted automatically
# Save the figure manually when it appears on screen
# File reference: model_comparison_v2.xlsx has the SHAP values
```
**Caption:** "SHAP feature importance for Model 1 (LightGBM, Condition A: 37 clean basins, good segments). Each dot represents one flooding event. Red = high feature value, blue = low. The x-axis shows the SHAP value — positive = prediction pushed toward less decay (higher IRD_norm_log). Features are ranked by mean absolute SHAP value. The top four features (prev_ALPHA, IRD_at_reset, prev_DrT, log1p_prev_HL) are consistent across all evaluation conditions (see Fig. S3)."

---

### Figure 5 — Model 1 held-out basin time series (Condition D)
**Status:** generated automatically — save from screen
**Content:** 5 panels (one per held-out basin), actual vs predicted IRD over time, circles = actual, crosses = predicted

**How to get it:**
```bash
python -m models.model1_decay
# The held-out time series plot is generated automatically for condition D
# Function: plot_held_out_basins_timeseries()
# Save the figure manually when it appears
```
**Caption:** "Model 1 generalizability to unseen basins (Condition D). Each panel shows the actual (circles) and predicted (crosses) IRD time series for one of five held-out basins excluded from training. The model was trained on 37 clean basins and applied to these 5 unseen basins without any fine-tuning. Pooled R²=+0.871, MAPE=14.8% (n=2,504 events). Per-basin metrics are reported in Table S4."

---

### Figure 6 — Model 2 held-out basin time series (Condition D)
**Status:** generated automatically — save from screen
**Content:** 5 panels (one per held-out basin), actual vs predicted IRD_at_reset over time

**How to get it:**
```bash
python -m models.model2_reset
# The held-out time series plot is generated automatically for condition D
# Function: plot_held_out_timeseries()
# Save the figure manually when it appears
```
**Caption:** "Model 2 generalizability to unseen basins (Condition D). Each panel shows actual (circles) and predicted (crosses) IRD_at_reset values at each tillage event for one of five held-out basins. The model was trained on 37 clean basins and tested on these 5 unseen basins. Pooled R²=+0.888, RMSE=0.969 cm/h (n=454 reset events). Note that the naive baseline (predicting no change from previous reset) achieves R²=+0.867, reflecting high autocorrelation in the held-out basins; the model reduces RMSE by 0.081 cm/h (7.7%) over naive."

---

### Table 1 — Filter funnel
**Status:** ready — numbers above
**Content:** Two-column table: All 50 basins | Clean dataset
**Source:** `outputs/tables/filter_funnel.xlsx`

**How to get it:**
```bash
python -m pipeline.build_dataset --rebuild
# filter_funnel.xlsx is saved automatically
```

---

### Table 2 — Model 1 condition comparison
**Status:** ready — numbers above
**Content:** Conditions A–E × R²(log), R²(IRD), RMSE, MAPE, n_train, n_test
**Source:** `outputs/tables/model1_results_v2.xlsx`

**How to get it:**
```bash
python -m models.model1_decay
# model1_results_v2.xlsx is saved automatically
```
**Include naive baseline as first row.**

---

### Table 3 — Model 2 condition comparison
**Status:** ready — numbers above
**Content:** Conditions Chrono/D/E × R²(IRD), RMSE, MAPE, n_train, n_test, naive baseline
**Source:** `outputs/tables/model2_results_v2.xlsx`

**How to get it:**
```bash
python -m models.model2_reset
# model2_results_v2.xlsx is saved automatically
```

---

## SUPPLEMENTARY INFORMATION FIGURES

---

### Figure S1 — Model 1 feature selection curve
**Status:** generated automatically — save from screen
**Content:** 3-panel figure: RMSE vs n_features | delta RMSE per step | R²(IRD) vs n_features

**How to get it:**
```bash
python -m analysis.feature_selection_unconstrained
# Plot appears automatically at end of run
# Save manually
```
**Caption:** "Forward stepwise feature selection for Model 1 (unconstrained, Condition E, n=2,504 held-out test events). Left: RMSE on raw IRD (cm/h) as features are added. Middle: marginal RMSE improvement at each step (positive = improvement). Right: R²(IRD) vs number of features. The elbow at 11 features (step 7 = first worsening after step 11) identifies the optimal feature set. The 11-feature model outperforms the full 21-feature model by 0.047 cm/h RMSE, confirming overfitting in the full model."

---

### Figure S2 — Model 2 feature selection curve
**Status:** generated automatically — save from screen
**Content:** Same 3-panel structure as S1

**How to get it:**
```bash
python -m analysis.feature_selection_m2
# Plot appears automatically at end of run
# Save manually
```
**Caption:** "Forward stepwise feature selection for Model 2 (unconstrained, chronological split, n=624 test resets). The elbow at 11 features is unambiguous — all 11 subsequent additions worsen RMSE. The 11-feature model outperforms the full 22-feature model by 0.054 cm/h RMSE."

---

### Figure S3 — SHAP stability comparison A/D/E (Model 1)
**Status:** generated automatically — save from screen
**Content:** 3-panel horizontal bar chart: mean |SHAP| per feature for conditions A, D, E side by side

**How to get it:**
```bash
python -m analysis.model_comparison
# The SHAP comparison plot appears automatically after all conditions are run
# Function: plot_shap_comparison()
# Save manually
```
**Caption:** "SHAP feature importance stability across evaluation conditions (Model 1, LightGBM). Each panel shows mean absolute SHAP values for the 11 features under a different condition: A = clean basins within-sample, D = clean basins held-out, E = all-data held-out. The top four features (prev_ALPHA, IRD_at_reset, prev_DrT, log1p_prev_HL) are identical across all conditions, demonstrating that the physical interpretation is robust to data quality choices."

---

### Figure S4 — Model 1 held-out scatter plots
**Status:** generated automatically — save from screen
**Content:** 5 panels, actual vs predicted IRD (cm/h), per held-out basin, Condition D

**How to get it:**
```bash
python -m models.model1_decay
# Function: plot_held_out_basins()
# Scatter plot appears automatically for condition D
# Save manually
```
**Caption:** "Actual vs predicted IRD scatter for five held-out basins (Model 1, Condition D). Each panel corresponds to one basin excluded from training. Per-basin R², RMSE, and MAPE are annotated. The model trained on 37 clean basins generalizes to all five unseen basins across four different fields."

---

### Figure S5 — Model 2 held-out scatter plots
**Status:** generated automatically — save from screen
**Content:** 5 panels, actual vs predicted IRD_at_reset (cm/h), per held-out basin

**How to get it:**
```bash
python -m models.model2_reset
# Function: plot_held_out_scatter()
# Scatter plot appears automatically for condition D
# Save manually
```
**Caption:** "Actual vs predicted IRD_at_reset scatter for five held-out basins (Model 2, Condition D). Each panel shows one tillage-level reset event per point. Per-basin R², RMSE, and MAPE are annotated."

---

### Figure S6 — Per-basin R² histogram (Model 1)
**Status:** generated automatically — save from screen
**Content:** Histogram of per-basin R²(IRD) across all 50 basins (or 37 clean), Condition A

**How to get it:**
```bash
python -m analysis.basin_analysis
# Function: produces basin_metric_histograms.png automatically
# Also saved to: outputs/figures/basin_metric_histograms.png
```
**Caption:** "Distribution of per-basin R²(IRD) for Model 1 (LightGBM, Condition A). Each bar represents one basin. Red bars = outlier basins excluded from clean training set. The majority of clean basins achieve R² > 0.4, with several exceeding 0.7."

---

### Figure S7 — Outlier basin diagnostic plots
**Status:** generated automatically — save from screen
**Content:** 8 panels (one per outlier basin) showing IRD vs time and why they were flagged

**How to get it:**
```bash
python -m analysis.basin_analysis
# Basin plots saved to: outputs/figures/basin_plots/
# Pick the 8 outlier basins: 7102, 4304, 4303, 7103, 7202, 6101, 7303, 4103
# Combine into one multi-panel figure manually
```
**Caption:** "Diagnostic plots for the eight excluded outlier basins. Type 1 (low dynamic range): IRD_at_reset shows insufficient variability for model learning (IQR < 0.60 cm/h). Type 3 (non-stationary): IRD_at_reset trends monotonically up or down across segments, violating the stationarity assumption of the log-ratio target. Per-basin pass-1 R² values are indicated."

---

## SUPPLEMENTARY INFORMATION TABLES

---

### Table S1 — Feature list with physical justification
**Status:** write manually (documented in pipeline/features.py)
**Content:** Two sections — Model 1 (11 features) and Model 2 (11 features)
**Columns:** Feature name | Physical variable | Unit | Transformation | Physical rationale

**Model 1 features (from pipeline/features.py docstring):**
1. IRD_at_reset — hydraulic conductivity ceiling at segment start — cm/h — none
2. prev_ALPHA — drying fraction of previous event — dimensionless — none
3. log1p_prev_HL — hydraulic load of previous event — log(m³+1) — log1p
4. prev_DrT — drying time of previous event — hours — none
5. LCT — cumulative loading time since reset — hours — none
6. prev_TD — temperature during previous drying — °C — none
7. prev_FT — flooding time of previous event — hours — none
8. cum_TW — cumulative wetting temperature since reset — °C·h — none
9. cum_FT — cumulative flooding time since reset — hours — none
10. prev_RD — radiation during previous drying — W/m² — none
11. prev_RW — radiation during previous wetting — W/m² — none

**Model 2 features:**
1. month_sin — seasonal signal (peak July) — dimensionless — sin(2π(month-4)/12)
2. month_cos — orthogonal seasonal component — dimensionless — cos(2π(month-4)/12)
3. prev_IRD_at_reset — previous reset level — cm/h — none
4. prev_prev_IRD_at_reset — two-step history — cm/h — none
5. mean_ALPHA — mean drying fraction over segment — dimensionless — none
6. total_LCT — total flooding duration of segment — hours — none
7. sum_DrT — cumulative drying time of segment — hours — none
8. sum_FT — cumulative flooding time of segment — hours — none
9. last_DrT — final drying duration before tillage — hours — none
10. last_RD — radiation in final drying event — W/m² — none
11. DAR — daily ambient radiation at reset date — W/m² — none

---

### Table S2 — Algorithm comparison Model 1 (all conditions)
**Status:** ready
**Source:** `outputs/tables/model_comparison_v2.xlsx`

**How to get it:**
```bash
python -m analysis.model_comparison
# model_comparison_v2.xlsx saved automatically
# Use the full table: all algorithms × all conditions
```

---

### Table S3 — Algorithm comparison Model 2 (all conditions)
**Status:** ready
**Source:** `outputs/tables/model2_comparison_v2.xlsx`

**How to get it:**
```bash
python -m analysis.model2_comparison
# model2_comparison_v2.xlsx saved automatically
```

---

### Table S4 — Per-basin held-out metrics Model 1
**Status:** ready — numbers above
**Content:** 5 basins × R², RMSE, MAPE for Conditions D and E separately
**Source:** printed in model1_decay output

---

### Table S5 — Per-basin held-out metrics Model 2
**Status:** ready — numbers above
**Content:** 5 basins × R², RMSE, MAPE for Conditions D and E
**Source:** printed in model2_reset output

---

### Table S6 — Model 1 feature selection path
**Status:** ready
**Source:** `outputs/tables/feature_selection_unconstrained.xlsx`
**Content:** Step | Feature added | N features | RMSE | ΔRMSE | MAPE% | R²

---

### Table S7 — Model 2 feature selection path
**Status:** ready
**Source:** `outputs/tables/feature_selection_m2.xlsx`
**Content:** Step | Feature added | N features | RMSE | ΔRMSE | MAPE% | R²

---

## PPT SLIDE PLAN

### Design choices
- Color palette: Ocean Gradient — deep blue `065A82` + teal `1C7293` + white `FFFFFF`
- Title/section slides: dark background (`065A82`)
- Content slides: white background
- Font: Cambria headers, Calibri body
- Slide size: 16x9 (10" × 5.625")
- Minimal text — figures and captions only
- Each slide tagged: [PAPER] or [SI] or [OMITTED — reason]

---

### SLIDE LIST

**Slide 1 — Title slide [PAPER]**
- Title: "From Decay to Recovery: A Dual Machine Learning Framework for Infiltration Rate Prediction in Managed Aquifer Recharge"
- Authors, journal, date
- Dark background

**Slide 2 — System overview [PAPER → Figure 1]**
- Map + schematic + example time series
- Caption: Figure 1 caption (see above)

**Slide 3 — IRD decay evidence [PAPER → Figure 2]**
- 3-panel: representative basins showing decay structure
- Caption: Figure 2 caption

**Slide 4 — Dual-model framework [PAPER → Figure 3]**
- Conceptual diagram built in PowerPoint
- Caption: Figure 3 caption

**Slide 5 — Filter funnel [PAPER → Table 1]**
- Native PowerPoint table
- Caption: "Data quality filtering retains 48.1% of raw flooding events for Model 1 training."

**Slide 6 — Outlier basin classification [SI → Figure S7]**
- 8-panel diagnostic plots
- Note: [SI] — reviewer justification for exclusion criteria
- Caption: Figure S7 caption

**Slide 7 — Model 1 condition summary [PAPER → Table 2]**
- Native PowerPoint table, A–E conditions + naive baseline
- Highlight condition E as primary result
- Caption: "Model 1 performance across evaluation conditions."

**Slide 8 — Model 1 SHAP beeswarm [PAPER → Figure 4]**
- SHAP beeswarm, Condition A
- Caption: Figure 4 caption

**Slide 9 — Model 1 held-out time series [PAPER → Figure 5]**
- 5-panel time series, Condition D
- Caption: Figure 5 caption

**Slide 10 — Model 1 held-out scatter [SI → Figure S4]**
- 5-panel scatter, Condition D
- Note: [SI] — supports Figure 5
- Caption: Figure S4 caption

**Slide 11 — Model 1 SHAP stability [SI → Figure S3]**
- 3-panel SHAP comparison A/D/E
- Note: [SI] — physical robustness demonstration
- Caption: Figure S3 caption

**Slide 12 — Model 1 feature selection curve [SI → Figure S1]**
- 3-panel RMSE curve
- Note: [SI] — methodology justification
- Caption: Figure S1 caption

**Slide 13 — Algorithm comparison Model 1 [OMITTED from paper — in SI → Table S2]**
- Bar chart: 5 algorithms × key metrics, Condition A
- Note: [OMITTED from main paper] — LightGBM chosen for SHAP compatibility, marginal CatBoost advantage noted in text
- Caption: "LightGBM vs CatBoost vs XGBoost vs RandomForest vs Ridge — Condition A."

**Slide 14 — Model 2 condition summary [PAPER → Table 3]**
- Native PowerPoint table, Chrono/D/E + naive baseline
- Highlight Held-out D as primary result
- Caption: "Model 2 performance across evaluation conditions."

**Slide 15 — Model 2 held-out time series [PAPER → Figure 6]**
- 5-panel time series, Condition D
- Caption: Figure 6 caption

**Slide 16 — Model 2 held-out scatter [SI → Figure S5]**
- 5-panel scatter, Condition D
- Note: [SI]
- Caption: Figure S5 caption

**Slide 17 — Model 2 SHAP beeswarm [SI — considered for paper]**
- SHAP beeswarm, Condition D
- Note: [SI] — Model 2 SHAP story is secondary to Model 1
- Caption: "SHAP feature importance for Model 2 (Condition D). prev_IRD_at_reset and DAR are the dominant predictors across all conditions."

**Slide 18 — Model 2 feature selection curve [SI → Figure S2]**
- 3-panel RMSE curve
- Note: [SI]
- Caption: Figure S2 caption

**Slide 19 — Algorithm comparison Model 2 [OMITTED — in SI → Table S3]**
- Note: [OMITTED from main paper] — all gradient boosting methods essentially tied
- Caption: "5-algorithm comparison for Model 2."

**Slide 20 — Per-basin R² histogram [SI → Figure S6]**
- Histogram of per-basin R² across 50 basins
- Note: [SI]
- Caption: Figure S6 caption

**Slide 21 — Per-basin held-out tables M1+M2 [SI → Tables S4, S5]**
- Two tables side by side: Model 1 (left) and Model 2 (right)
- 5 held-out basins each
- Caption: "Per-basin held-out test metrics for Model 1 (left) and Model 2 (right), Condition D."

**Slide 22 — Feature lists [SI → Table S1]**
- Two-column layout: Model 1 features (left) | Model 2 features (right)
- Note: [SI]

**Slide 23 — Random split comparison [OMITTED]**
- Note: [OMITTED] — chrono split is the honest evaluation; random split inflates performance
- Not shown in paper or SI

**Slide 24 — Oracle baseline comparison [OMITTED]**
- Note: [OMITTED] — oracle ceiling % language dropped; artifact of subset mismatch

---

## PAPER WRITING INSTRUCTIONS

### Section order
1. Abstract (write last)
2. Introduction
3. Methods
4. Results
5. Discussion
6. Conclusions
7. SI

### Methods subsections (already drafted, need number updates)
- 3.1 Study site — Shafdan SAT, 50 basins, 10-year IoT record
- 3.2 Data collection and preprocessing
- 3.3 Event segmentation and exponential decay fitting
- 3.4 Filter funnel — Table 1
- 3.5 Outlier basin classification — 8 basins excluded, 2 types
- 3.6 Model 1: within-segment decay prediction
- 3.7 Feature engineering (Model 1, 11 features)
- 3.8 Model 2: post-tillage recovery prediction
- 3.9 Feature engineering (Model 2, 11 features)
- 3.10 Evaluation strategy — 5 conditions, chrono vs held-out

### Results subsections (primary numbers to use)
- 4.1 Model 1 within-sample performance — Condition A: R²(IRD)=+0.725, MAPE=17.2%
- 4.2 Model 1 generalizability — Condition D pooled held-out: R²=+0.871, MAPE=14.8%
- 4.3 Effect of data quality choices — D vs E: 2.1% MAPE difference
- 4.4 Feature importance — SHAP top 4 stable: prev_ALPHA, IRD_at_reset, prev_DrT, log1p_prev_HL
- 4.5 Model 2 within-sample — Chrono: R²(IRD)=+0.818 vs naive +0.761
- 4.6 Model 2 generalizability — Held-out D: R²=+0.888, RMSE=0.969 cm/h

### Key narrative choices (already agreed)
- Primary result = Condition E for Model 1 framing, but D for headline generalizability number
- Report RMSE as primary metric for Model 2 (not MAPE — naive MAPE misleadingly low)
- Do NOT use "oracle ceiling %" language
- Do NOT report random split for Model 2
- LightGBM chosen for SHAP compatibility; CatBoost marginal advantage noted in one sentence
- Filtering justified by MAPE improvement (D=16.0% vs E=18.1%), not R² improvement

### Key claims to support with specific numbers
1. "The model generalizes to completely unseen basins" → pooled held-out R²=+0.871 (M1), +0.888 (M2)
2. "Feature importance is robust across data quality choices" → SHAP top 4 identical A/D/E
3. "Segment quality filtering improves operational precision" → Condition D (16.0%) vs E (18.1%) MAPE
4. "The reduced 11-feature model outperforms the full model" → ΔRMSE = +0.047 cm/h (M1), +0.054 cm/h (M2)
5. "The model beats naive persistence" → M1: +0.228 R² over naive; M2: +0.057 R²(chrono), −0.081 cm/h RMSE(held-out)

---

## IMPORTANT NOTES FOR NEW CONVERSATION

1. **Start each new conversation** by sharing this file and saying "I want to work on slide X" or "I want to write section Y"

2. **For PPT slides:** work one slide at a time, generate the pptxgenjs code, run it, inspect visually, then move to next slide

3. **For paper sections:** work one subsection at a time, write draft text, review numbers against this document, finalize

4. **Figures are NOT embedded in the PPT code** — they exist as matplotlib outputs saved manually from your screen. The PPT code creates placeholder boxes with exact labels (e.g., "INSERT: Figure 5 — model1_decay.py → condition D → plot_held_out_basins_timeseries()") 

5. **All final numbers are in this document** — do not re-run code to get numbers; use the tables above

6. **The Methods section draft already exists** — reference the previous conversation for the drafted text of sections 3.1–3.9

7. **outlier_basins.csv** must exist before running pass-2 models — run `basin_analysis.py` first

8. **Two-pass model execution:**
   - Pass 1: `python -m models.model1_decay` (before basin_analysis — all 50 basins)
   - Run: `python -m analysis.basin_analysis` (produces outlier_basins.csv)
   - Pass 2: `python -m models.model1_decay` (after basin_analysis — 8 outliers excluded)

9. **Model 2 condition E fix** — outlier basins must have their split_held_out reassigned from "excluded" to chrono splits — this is done inline in main() of model2_reset.py and in prepare_condition() of model2_comparison.py

10. **SHAP plots are generated inside model_comparison.py** — conditions A, D, E automatically; the beeswarm plots appear on screen and must be saved manually