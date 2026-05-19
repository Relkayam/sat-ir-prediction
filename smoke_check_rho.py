# smoke_check_rho.py
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\user\PycharmProjects\sat-ir-prediction")
df = pd.read_csv(PROJECT_ROOT / "data" / "event_dataset.csv")
df = df[df["row_type"] == "event"]

for bn in [4104, 3204, 5201, 7401]:
    bdf = df[df["basin_number"] == bn]
    rho_median = bdf["IRD_at_reset"].median()
    rho_std    = bdf["IRD_at_reset"].std()
    rho_min    = bdf["IRD_at_reset"].min()
    rho_max    = bdf["IRD_at_reset"].max()
    print(f"Basin {bn}: median_rho={rho_median:.2f}  "
          f"std={rho_std:.2f}  "
          f"min={rho_min:.2f}  max={rho_max:.2f}  "
          f"n={len(bdf)}")