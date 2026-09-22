"""
Build era_monthly_with_lags.parquet from the single combined ERA5 MAM-JJA file,
computing precip/tair lags analogous to process_trace_monthly_with_lags.py.

Unlike the TraCE version, ERA5's variables are NOT split across six separately
bias-corrected per-variable files - they already live together in one combined
parquet (era5_ignition_burnedarea_model_MAM_JJA_1946_1990.parquet), with:
  - tair, precip           : present for March-August (months 3-8)
  - rh, wind, swr, sp       : present for June-August only (months 6-8),
                              NaN for March-May

This version keeps ERA5's NATIVE grid resolution throughout - no snapping to
the (coarser) TraCE gridcells. Lags are computed per (lat, lon) at whatever
resolution the source ERA5 file already uses. A duplicate-key check (rather
than a forced aggregation) confirms the native grid is already unique per
(lat, lon, year, month) before lags are computed, since duplicate keys would
still cause the same join-explosion problem seen earlier if left unchecked.
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

mu_sigma_root            = config["mu_sigma_root"]
paleoclimate_data_root   = config["paleoclimate_data_root"]
era5_data_root           = config["era5_data_root"]


# ============================================================
# Configuration
# ============================================================

ERA5_MAM_JJA_PATH = (
    Path(era5_data_root)
    / "bias_correction/processed_parquet/era5_ignition_burnedarea_model_MAM_JJA_1946_1990.parquet"
)

OUT_DIR = Path(era5_data_root) / "bias_correction/processed_parquet/"
OUT_PATH = OUT_DIR / "era_calib_period_monthly_with_lags.parquet"

ID_COLS = ["lat", "lon", "year", "month"]

# Variables that exist for June-August only (NaN for March-May in the source file)
JJA_ONLY_VARS = ["rh", "wind", "swr", "sp"]
# Variables that exist for March-August
MAM_JJA_VARS = ["tair", "precip"]


# ============================================================
# Helpers
# ============================================================

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
# Load raw ERA5 MAM-JJA data (already combined, single file)
# ============================================================

print("Loading combined ERA5 MAM-JJA parquet...")
era = pd.read_parquet(ERA5_MAM_JJA_PATH)
print(f"  Raw rows: {len(era):,}, months present: {sorted(era['month'].unique())}")
print(f"  Years: {era.year.min()}\u2013{era.year.max()}")


# ============================================================
# Confirm ERA5's native grid is already unique per (lat, lon, year, month).
# Lags are computed directly on this native resolution - no snapping or
# aggregation to another grid. A duplicate key here would still cause the
# same join-explosion problem diagnosed earlier, so check rather than assume.
# ============================================================

dupe_check = era.duplicated(subset=ID_COLS).sum()
print(f"\nDuplicate (lat, lon, year, month) rows in native ERA5 grid: {dupe_check}")
if dupe_check > 0:
    raise ValueError(
        f"Found {dupe_check} duplicate (lat, lon, year, month) rows in the source "
        "ERA5 file - resolve this (e.g. investigate the source file / aggregate "
        "duplicates) before computing lags, or downstream joins will silently "
        "explode in row count, as happened with the earlier trace-cell-snapped version."
    )

n_gridcells = era[["lat", "lon"]].drop_duplicates().shape[0]
print(f"  Unique ERA5 gridcells: {n_gridcells}")


# ============================================================
# Build lag lookup (tair, precip - available March-August)
# ============================================================

print("\nBuilding lag lookup from March-August tair/precip...")

lookup = era[ID_COLS + MAM_JJA_VARS].dropna(subset=MAM_JJA_VARS).copy()
lookup["precip_depth_m"] = lookup.apply(
    lambda r: r["precip"] * seconds_in_month(r["year"], r["month"]), axis=1
)
lookup = lookup[["lat", "lon", "year", "month", "precip_depth_m", "tair"]]

print(f"  Lag source: {lookup.shape[0]:,} rows, "
      f"months present: {sorted(lookup['month'].unique())}  (expected: [3, 4, 5, 6, 7, 8])")


# ============================================================
# Subset to JJA target rows (since rh/wind/swr/sp only exist there)
# ============================================================

merged = era[era["month"].isin([6, 7, 8])].copy()
merged = merged.sort_values(["lat", "lon", "year", "month"]).reset_index(drop=True)
print(f"\nJJA target rows: {merged.shape[0]:,}")


# ============================================================
# Compute 1/2/3-month lags
# ============================================================

print("Computing 1/2/3-month lags...")

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
# the current year, per gridcell)
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

final_cols = ID_COLS + ["rh", "tair", "precip", "wind", "swr", "sp"] + [
    "precip_1m", "precip_2m", "precip_3m", "precip_5y",
    "tair_1m", "tair_2m", "tair_3m", "tair_5y",
]

era_monthly_with_lags = merged[final_cols].copy()

OUT_DIR.mkdir(parents=True, exist_ok=True)
era_monthly_with_lags.to_parquet(OUT_PATH, index=False)

print(f"\nSaved: {OUT_PATH}")
print(f"  {len(era_monthly_with_lags):,} rows x {len(era_monthly_with_lags.columns)} columns")
print(f"  Years: {era_monthly_with_lags.year.min()}\u2013{era_monthly_with_lags.year.max()}")

print("\nNaN counts by lag column, by month (all should now be 0 for June/July/August):")
lag_cols = ["precip_1m", "precip_2m", "precip_3m", "tair_1m", "tair_2m", "tair_3m"]
print(era_monthly_with_lags.groupby("month")[lag_cols].apply(lambda g: g.isna().sum()))
