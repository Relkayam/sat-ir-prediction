"""
make_feature_table_SI.py — SI Table: Model 1 + Model 2 input features
=====================================================================
Writes the feature-definition table for the SI to Excel and prints the table
caption. Feature LISTS are imported from pipeline.features (single source of
truth) so the table can never silently disagree with the model code — if a
feature is added/removed there, this script must be updated and will flag any
feature it has no definition for.

Definitions/units are curated here (from features.py docstring + Nomenclature)
since they are physical descriptions, not values.

Output: outputs/tables/tableS_features.xlsx  (two sheets: Model_1, Model_2)

Usage
-----
  python -m analysis.make_feature_table_SI
"""
from __future__ import annotations

import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import TABLES_DIR
from pipeline.features import MODEL1_FEATURES, MODEL2_FEATURES

OUT = TABLES_DIR / "tableS_features.xlsx"

# ── Curated definitions, keyed by the FINAL (post-transform) feature name ─────
# symbol, definition, units
M1_DEFS = {
    "IRD_at_reset":   ("ρᵢ",            "Post-tillage recovery ceiling (IRD at the segment's reset anchor)", "cm h⁻¹"),
    "prev_ALPHA":     ("α (prev.)",     "Drying fraction of the previous event, α = t_dry / t_cycle",        "–"),
    "log1p_prev_HL":  ("log HL (prev.)","log(1 + HL) of the previous event; HL = CIV / basin area",          "– (cm)"),
    "prev_DrT":       ("t_dry (prev.)", "Drying time of the previous event",                                 "h"),
    "prev_TD":        ("T_dry (prev.)", "Air temperature during the previous drying phase",                  "°C"),
    "prev_RD":        ("R_dry (prev.)", "Solar radiation during the previous drying phase",                  "W m⁻²"),
    "prev_RW":        ("R_wet (prev.)", "Solar radiation during the previous wetting phase",                 "W m⁻²"),
    "prev_FT":        ("t_flood (prev.)","Flooding time of the previous event",                              "h"),
    "LCT":            ("TST",           "Time since the last tillage event",                                 "h"),
    "cum_TW":         ("cum. T_wet",    "Cumulative wetting-phase air temperature since the segment reset",  "°C"),
    "cum_FT":         ("cum. t_flood",  "Cumulative flooding time since the segment reset",                  "h"),
}

M2_DEFS = {
    "prev_delta":      ("δᵢ₋₁",          "Previous reset log-ratio, ln(ρᵢ₋₁ / ρᵢ₋₂)",                        "–"),
    "prev_prev_delta": ("δᵢ₋₂",          "Reset log-ratio two steps back, ln(ρᵢ₋₂ / ρᵢ₋₃)",                  "–"),
    "DAR":             ("R̄_day",        "Daily-mean solar radiation at the tillage date",                   "W m⁻²"),
    "mean_RW":         ("mean R_wet",    "Mean wetting-phase radiation over segment i−1",                    "W m⁻²"),
    "mean_DrT":        ("mean t_dry",    "Mean drying time over segment i−1",                                "h"),
    "max_DrT":         ("max t_dry",     "Longest drying time in segment i−1",                               "h"),
    "last_RD":         ("last R_dry",    "Drying radiation of the final event of segment i−1",               "W m⁻²"),
    "max_RD":          ("max R_dry",     "Peak drying radiation in segment i−1",                             "W m⁻²"),
    "sum_RW":          ("sum R_wet",     "Summed wetting radiation over segment i−1",                        "W m⁻²"),
    "total_LCT":       ("total seg. dur.","Total time span of segment i−1",                                  "h"),
    "n_events":        ("event count",   "Number of flooding events in segment i−1",                         "–"),
    "frac_zero_DrT":   ("frac. no drying","Fraction of segment i−1 events with t_dry ≈ 0",                   "–"),
}


def _build(features: list[str], defs: dict, model_name: str) -> pd.DataFrame:
    missing = [f for f in features if f not in defs]
    if missing:
        raise KeyError(
            f"{model_name}: no SI definition for feature(s): {missing}\n"
            "Update M1_DEFS / M2_DEFS in make_feature_table_SI.py to match "
            "pipeline.features.")
    rows = []
    for f in features:
        sym, defn, unit = defs[f]
        rows.append({"Feature (code)": f, "Symbol": sym,
                     "Definition": defn, "Units": unit})
    return pd.DataFrame(rows)


CAPTION = (
    "Table S[x]. Input features for Model 1 and Model 2. Model 1 predicts "
    "η = ln(IRD/ρᵢ) from {n1} features (feature vector x(t)) describing the "
    "current event's time since tillage and the preceding event's operational "
    "and weather conditions; \"prev.\" denotes the immediately preceding event "
    "and \"cum.\" a cumulative sum since the segment reset. Model 2 predicts "
    "δ = ln(ρᵢ/ρᵢ₋₁) from {n2} features (feature vector zᵢ₋₁) summarizing the "
    "preceding inter-tillage segment (mean, maximum, sum, or final-event values) "
    "together with the two prior reset log-ratios and the ambient radiation at "
    "the tillage date. Radiation during wetting (R_wet) and during drying "
    "(R_dry) are distinct features. Symbols follow the Nomenclature."
)


def main() -> None:
    print("=" * 64)
    print("  BUILD SI FEATURE TABLE (Model 1 + Model 2)")
    print("=" * 64)

    m1 = _build(MODEL1_FEATURES, M1_DEFS, "Model 1")
    m2 = _build(MODEL2_FEATURES, M2_DEFS, "Model 2")

    print(f"\n  Model 1: {len(m1)} features")
    print(m1.to_string(index=False))
    print(f"\n  Model 2: {len(m2)} features")
    print(m2.to_string(index=False))

    with pd.ExcelWriter(OUT) as xl:
        m1.to_excel(xl, sheet_name="Model_1", index=False)
        m2.to_excel(xl, sheet_name="Model_2", index=False)
    print(f"\n  Saved: {OUT}  (sheets: Model_1, Model_2)")

    print("\n" + "=" * 64)
    print("  TABLE CAPTION:")
    print("=" * 64)
    print(CAPTION.format(n1=len(m1), n2=len(m2)))


if __name__ == "__main__":
    main()