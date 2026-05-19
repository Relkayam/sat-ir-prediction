# smoke_fig6_check.py
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
df = pd.read_csv(PROJECT_ROOT / "data" / "reset_dataset.csv",
                 parse_dates=["reset_date"])
df = df.loc[:, ~df.columns.duplicated()]

HELD_OUT = [3203, 4104, 5102, 6303, 7201]

from config import FIELD_NAMES

ho = df[df["basin_number"].isin(HELD_OUT)].copy()

print("Reset dataset — held-out basins:")
print(f"{'Basin':<8} {'Field':<12} {'n_resets':<10} "
      f"{'date_min':<12} {'date_max':<12} "
      f"{'IRD_min':<8} {'IRD_max':<8} {'IRD_mean':<8}")
print("-" * 80)

for bn in HELD_OUT:
    bdf   = ho[ho["basin_number"] == bn]
    field = FIELD_NAMES.get(int(str(bn)[0]), "")
    ird   = pd.to_numeric(bdf["IRD_at_reset"], errors="coerce")
    print(f"{bn:<8} {field:<12} {len(bdf):<10} "
          f"{str(bdf['reset_date'].min().date()):<12} "
          f"{str(bdf['reset_date'].max().date()):<12} "
          f"{ird.min():<8.2f} {ird.max():<8.2f} {ird.mean():<8.2f}")

print(f"\nTotal held-out resets: {len(ho)}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nsplit_held_out values: {df['split_held_out'].value_counts().to_dict()}")
print(f"basin_role values: {df['basin_role'].value_counts().to_dict()}")