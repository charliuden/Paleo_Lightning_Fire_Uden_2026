
"""
Combine TraCE seasonal bias-corrected outputs into a single monthly dataframe
(June-August), then compute precip/tair lags analogous to process_era5_to_csv.py.

Data coverage:
  - tair, precip: corrected for March-August (6 months)
  - sp, wind, swr, rh: corrected for June-August only (3 months)

Lags for June/July/August therefore need March/April/May tair and precip values,
which only exist in the tair/precip files themselves - NOT in the combined
6-variable frame (which is June-August only, since it's inner-joined against the
JJA-only variables). The lag lookup table is built separately from the full
March-August tair/precip files to avoid silently losing March-May data.
"""

import calendar
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Config loading
# ============================================================

def read_properties(file_path: str | Path) -> dict:
    """Parse a key=value properties file into a dict of stripped strings."""
    file_path = Path(file_path)
    props = {}
    with open(file_path, "r") as f:
        for line in f:
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            props[key.strip()] = value.strip()
    return props


# *** paleo_config.properties must be in Paleo_Lightning_Fire_Uden_2026
current_path = Path(__file__).resolve().parent  # use Path.cwd() instead if running interactively
base_path_parts = current_path.parts
try:
    idx = base_path_parts.index("Paleo_Lightning_Fire_Uden_2026")
    base_path = Path(*base_path_parts[: idx + 1])
except ValueError:
    raise RuntimeError(
        "Could not find 'Paleo_Lightning_Fire_Uden_2026' in the current path - "
        "set base_path manually instead."
    )

config = read_properties(base_path / "paleo_config.properties")

drivers_root            = config["drivers_root"]
rds_root                = config["rds_root"]
table_root              = config["table_root"]
predictions_root        = config["predictions_root"]
mu_sigma_root           = config["mu_sigma_root"]
performance_root        = config["performance_root"]
figure_root             = config["figure_root"]
fire_data_root          = config["fire_data_root"]
paleocliamte_data_root  = config["paleocliamte_data_root"]
era5_data_root          = config["era5_data_root"]
proximity_data_root     = config["proximity_data_root"]


# ============================================================
# Configuration
# ============================================================

IN_DIR = Path(paleocliamte_data_root) / "trace21ka_processed"
OUT_DIR = Path(paleocliamte_data_root) / "trace21ka_processed"

VARIABLES = ["tair", "sp", "wind", "swr", "precip", "rh"]

STAGE_PRIORITY = ["_final", "_rescaled", "_corrected"]

ID_COLS = ["lat", "lon", "ce_year", "year", "month"]


# ============================================================
# Helpers
# ============================================================

def pick_final_column(df: pd.DataFrame, var_name: str) -> str:
    """Return the best-available corrected column for this variable, preferring
    _final > _rescaled > _corrected > raw, based on what actually exists."""
    for suffix in STAGE_PRIORITY:
        candidate = f"{var_name}{suffix}"
        if candidate in df.columns:
            return candidate
    if var_name in df.columns:
        return var_name
    raise ValueError(f"No usable column found for '{var_name}' in {list(df.columns)}")


def seconds_in_month(year: int, month: int) -> int:
    return calendar.monthrange(int(year), int(month))[1] * 86400


def shift_year_month(year: np.ndarray, month: np.ndarray, n_back: int):
    """Vectorized: shift (year, month) arrays back by n_back calendar months."""
    total = year.astype(np.int64) * 12 + (month.astype(np.int64) - 1) - n_back
    new_year = total // 12
    new_month = total % 12 + 1
    return new_year, new_month


def add_month_lag(df: pd.DataFrame, lookup: pd.DataFrame, value_col: str,
                   n_back: int, out_col: str) -> pd.DataFrame:
    """Merge in `value_col` from `lookup`, shifted n_back calendar months earlier,
    matched on (lat, lon, year, month). Rows with no match get NaN."""
    ty, tm = shift_year_month(df["year"].values, df["month"].values, n_back)
    key = df[["lat", "lon"]].copy()
    key["year"] = ty
    key["month"] = tm

    merged_lag = key.merge(
        lookup[["lat", "lon", "year", "month", value_col]],
        on=["lat", "lon", "year", "month"],
        how="left",
    )
    df[out_col] = merged_lag[value_col].values
    return df


# ============================================================
# Load + combine all six variables (June-August only, since sp/wind/swr/rh
# only exist for those months)
# ============================================================

print("Loading per-variable seasonal-corrected TraCE parquet files...")

merged = None
for var_name in VARIABLES:
    path = IN_DIR / f"trace_{var_name}_seasonal_corrected.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")

    df = pd.read_parquet(path)
    final_col = pick_final_column(df, var_name)
    print(f"  {var_name}: using '{final_col}' as final value ({len(df):,} rows), "
          f"months present: {sorted(df['month'].unique())}")

    df_slim = df[ID_COLS + [final_col]].rename(columns={final_col: var_name})
    merged = df_slim if merged is None else merged.merge(df_slim, on=ID_COLS, how="inner")

merged = merged.sort_values(["lat", "lon", "year", "month"]).reset_index(drop=True)
print(f"\nCombined 6-variable frame: {merged.shape[0]:,} rows, {merged.shape[1]} columns")
print(f"  Years : {merged.year.min()}\u2013{merged.year.max()}")
print(f"  Months: {sorted(merged.month.unique())}  (expected: [6, 7, 8])")


# ============================================================
# Build a SEPARATE lag lookup from the full March-August tair/precip files.
# This is the key fix: lags for June must reach back into May/April, which
# don't exist in `merged` (JJA-only) but DO exist in the raw tair/precip exports.
# ============================================================

print("\nBuilding lag lookup source from full March-August tair/precip...")

tair_path = IN_DIR / "trace_tair_seasonal_corrected.parquet"
precip_path = IN_DIR / "trace_precip_seasonal_corrected.parquet"

tair_df = pd.read_parquet(tair_path)
precip_df = pd.read_parquet(precip_path)

tair_final_col = pick_final_column(tair_df, "tair")
precip_final_col = pick_final_column(precip_df, "precip")

tair_full = tair_df[ID_COLS + [tair_final_col]].rename(columns={tair_final_col: "tair"})
precip_full = precip_df[ID_COLS + [precip_final_col]].rename(columns={precip_final_col: "precip"})

lag_source = tair_full.merge(precip_full, on=ID_COLS, how="inner")
print(f"  Lag source: {lag_source.shape[0]:,} rows, "
      f"months present: {sorted(lag_source['month'].unique())}  (expected: [3, 4, 5, 6, 7, 8])")

# Precip depth (m) for lag accumulation - same approach as process_era5_to_csv.py
lag_source["precip_depth_m"] = lag_source.apply(
    lambda r: r["precip"] * seconds_in_month(r["year"], r["month"]), axis=1
)

lookup = lag_source[["lat", "lon", "year", "month", "precip_depth_m", "tair"]]


# ============================================================
# Compute 1/2/3-month lags, applied to `merged` (JJA target rows),
# looking up values from `lookup` (full March-August source)
# ============================================================

print("\nComputing 1/2/3-month lags...")

for n in (1, 2, 3):
    merged = add_month_lag(merged, lookup, "precip_depth_m", n, f"_precip_lag{n}")
    merged = add_month_lag(merged, lookup, "tair", n, f"_tair_lag{n}")

merged["precip_1m"] = merged["_precip_lag1"]
merged["precip_2m"] = merged["_precip_lag1"] + merged["_precip_lag2"]
merged["precip_3m"] = merged["_precip_lag1"] + merged["_precip_lag2"] + merged["_precip_lag3"]

merged["tair_1m"] = merged["_tair_lag1"]
merged["tair_2m"] = merged[["_tair_lag1", "_tair_lag2"]].mean(axis=1)
merged["tair_3m"] = merged[["_tair_lag1", "_tair_lag2", "_tair_lag3"]].mean(axis=1)

merged = merged.drop(columns=[c for c in merged.columns
                               if c.startswith("_precip_lag") or c.startswith("_tair_lag")])


# ============================================================
# 5-year JJA mean lags (mean of JJA values over the 5 years PRECEDING
# the current year, per gridcell) - matches process_era5_to_csv.py
# ============================================================

print("Computing 5-year JJA mean lags...")

jja_only = merged[merged["month"].isin([6, 7, 8])]

year_means = (
    jja_only.groupby(["lat", "lon", "year"])
    .agg(precip_yr_mean=("precip", "mean"), tair_yr_mean=("tair", "mean"))
    .reset_index()
    .sort_values(["lat", "lon", "year"])
)

year_means["precip_5y"] = (
    year_means.groupby(["lat", "lon"])["precip_yr_mean"]
    .transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
)
year_means["tair_5y"] = (
    year_means.groupby(["lat", "lon"])["tair_yr_mean"]
    .transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
)

merged = merged.merge(
    year_means[["lat", "lon", "year", "precip_5y", "tair_5y"]],
    on=["lat", "lon", "year"], how="left"
)


# ============================================================
# Assemble + save
# ============================================================

final_cols = ID_COLS + VARIABLES + [
    "precip_1m", "precip_2m", "precip_3m", "precip_5y",
    "tair_1m", "tair_2m", "tair_3m", "tair_5y",
]

trace_monthly_with_lags = merged[final_cols].copy()

out_path = OUT_DIR / "trace_monthly_with_lags_full_paleo.parquet"
trace_monthly_with_lags.to_parquet(out_path, index=False)

print(f"\nSaved: {out_path}")
print(f"  {len(trace_monthly_with_lags):,} rows x {len(trace_monthly_with_lags.columns)} columns")
print(f"  Years: {trace_monthly_with_lags.year.min()}\u2013{trace_monthly_with_lags.year.max()}")

print("\nNaN counts by lag column, by month (all should now be 0 for June/July/August):")
lag_cols = ["precip_1m", "precip_2m", "precip_3m", "tair_1m", "tair_2m", "tair_3m"]
print(trace_monthly_with_lags.groupby("month")[lag_cols].apply(lambda g: g.isna().sum()))
