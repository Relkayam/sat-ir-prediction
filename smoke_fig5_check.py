# smoke_fig5_check.py
# Tells me the held-out basin data structure without running full training
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
df = pd.read_csv(PROJECT_ROOT / "data" / "event_dataset.csv",
                 parse_dates=["opening_valve_date"])
df = df.loc[:, ~df.columns.duplicated()]

HELD_OUT = [3203, 4104, 5102, 6303, 7201]

# All events for held-out basins — Condition E uses ALL segments
ho = df[
    (df["basin_number"].isin(HELD_OUT)) &
    (df["row_type"] == "event")
].copy()

for col in ["IRD", "LCT", "IRD_at_reset", "segment_id"]:
    if col in ho.columns:
        ho[col] = pd.to_numeric(ho[col], errors="coerce")

print("Held-out basin summary (ALL segments — Condition E):")
print(f"{'Basin':<8} {'Field':<12} {'n_events':<10} {'n_segs':<8} "
      f"{'date_min':<12} {'date_max':<12} "
      f"{'IRD_min':<8} {'IRD_max':<8} {'IRD_mean':<8}")
print("-" * 90)

from config import FIELD_NAMES
for bn in HELD_OUT:
    bdf = ho[ho["basin_number"] == bn]
    field = FIELD_NAMES.get(int(str(bn)[0]), "")
    print(f"{bn:<8} {field:<12} {len(bdf):<10} "
          f"{bdf['segment_id'].nunique():<8} "
          f"{str(bdf['opening_valve_date'].min().date()):<12} "
          f"{str(bdf['opening_valve_date'].max().date()):<12} "
          f"{bdf['IRD'].min():<8.2f} {bdf['IRD'].max():<8.2f} "
          f"{bdf['IRD'].mean():<8.2f}")

print(f"\nTotal held-out events: {len(ho)}")
print(f"Date range: {ho['opening_valve_date'].min().date()} → "
      f"{ho['opening_valve_date'].max().date()}")