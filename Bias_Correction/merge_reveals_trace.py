"""
Merge REVEALS paleovegetation land cover (% conifer/broadleaf/open) with the
combined, lagged TraCE climate dataset.

Steps:
  1. Load each REVEALS time-slice CSV, rename LCC columns to C/B/U.
  2. Restrict to points near the Alaska TraCE grid, then aggregate (mean) to
     each TraCE cell via nearest-neighbor matching (cKDTree), same approach
     used in process_era5_to_csv.py for ERA5 -> TraCE aggregation.
  3. Build a year -> midpoint lookup from the supplied time-window table.
  4. Merge land cover onto trace_monthly_with_lags_full_paleo.parquet by
     (assigned TraCE cell, time-window midpoint).

COVERAGE NOTE: REVEALS only spans year -9750 to 1950 (11500 BP to 1950 CE).
TraCE's full paleo record extends to -14049. Rows outside REVEALS' range are
intentionally assigned NaN land cover, per project decision to exclude
pre-REVEALS years from land-cover-dependent analysis rather than extrapolate.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
from scipy.spatial import cKDTree


# ============================================================
# Config loading (same pattern as other scripts)
# ============================================================

def read_properties(file_path: str | Path) -> dict:
    file_path = Path(file_path)
    props = {}
    with open(file_path, "r") as f:
        for line in f:
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            props[key.strip()] = value.strip()
    return props


current_path = Path(__file__).resolve().parent
base_path_parts = current_path.parts
idx = base_path_parts.index("Paleo_Lightning_Fire_Uden_2026")
base_path = Path(*base_path_parts[: idx + 1])

config = read_properties(base_path / "paleo_config.properties")

rds_root               = config["rds_root"]
paleocliamte_data_root = config["paleocliamte_data_root"]
REVEALS_root            = config["REVEALS_root"]


# ============================================================
# Time-window table (midpoint -> (lower_bound, upper_bound], in years CE,
# BCE expressed as negative). Bins are half-open: (lower, upper].
# ============================================================

TIME_WINDOWS = [
    # (midpoint, lower_bound_exclusive, upper_bound_inclusive)
    (50,    1850, 1950),
    (200,   1600, 1850),
    (500,   1250, 1600),
    (1000,   750, 1250),
    (1500,   250,  750),
    (2000,  -250,  250),
    (2500,  -750, -250),
    (3000, -1250, -750),
    (3500, -1750, -1250),
    (4000, -2250, -1750),
    (4500, -2750, -2250),
    (5000, -3250, -2750),
    (5500, -3750, -3250),
    (6000, -4250, -3750),
    (6500, -4750, -4250),   # <-- fixed: was -4720, now -4750
    (7000, -5250, -4750),
    (7500, -5750, -5250),
    (8000, -6250, -5750),
    (8500, -6750, -6250),
    (9000, -7250, -6750),
    (9500, -7750, -7250),
    (10000, -8250, -7750),
    (10500, -8750, -8250),
    (11000, -9250, -8750),
    (11500, -9750, -9250),
]
MIDPOINTS = [w[0] for w in TIME_WINDOWS]

# bin edges must be strictly ascending for pd.cut
bin_edges = sorted(set([w[1] for w in TIME_WINDOWS] + [w[2] for w in TIME_WINDOWS]))
# labels: midpoint whose (lower, upper] interval matches each consecutive edge pair
edge_to_midpoint = {(w[1], w[2]): w[0] for w in TIME_WINDOWS}
bin_labels = []
for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    bin_labels.append(edge_to_midpoint.get((lo, hi), np.nan))  # NaN for the known gap


# ============================================================
# Load TraCE's 23-cell grid (reference points for nearest-neighbor matching)
# ============================================================

trace_cells_result = pyreadr.read_r(str(Path(rds_root) / "trace_cells.rds"))
trace_cells = trace_cells_result[None]  # pyreadr stores the object under key None for a single-object RDS
trace_cells = trace_cells[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
print(f"Loaded {len(trace_cells)} TraCE grid cells")

trace_tree = cKDTree(trace_cells[["lon", "lat"]].values)

# Distance threshold for nearest-neighbor assignment (degrees). TraCE grid
# spacing is ~3.7-3.75 deg; use roughly half that plus margin so REVEALS
# points far from any cell aren't force-matched to a distant "nearest" one.
MAX_MATCH_DIST = 2.5


# ============================================================
# Load + aggregate each REVEALS time-slice file to the TraCE grid
# ============================================================

print("\nLoading and aggregating REVEALS time slices...")

veg_records = []
for midpoint in MIDPOINTS:
    path = Path(REVEALS_root) / f"Land_Cover_{midpoint}.csv"
    if not path.exists():
        print(f"  WARNING: missing file for midpoint {midpoint}: {path}")
        continue

    df = pd.read_csv(path)

    # Identify and rename the C_/B_/U_ columns regardless of exact suffix
    col_map = {}
    for col in df.columns:
        if col.startswith("C_") or col == "C":
            col_map[col] = "C"
        elif col.startswith("B_") or col == "B":
            col_map[col] = "B"
        elif col.startswith("U_") or col == "U":
            col_map[col] = "U"
    df = df.rename(columns=col_map)

    keep_cols = ["Lon", "Lat", "C", "B", "U"]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Midpoint {midpoint}: missing expected columns {missing} "
                          f"(found: {list(df.columns)})")

    df = df[keep_cols].rename(columns={"Lon": "lon", "Lat": "lat"})

    # Restrict to a generous Alaska bounding box before nearest-neighbor matching,
    # so distant North American points can't be force-matched to an Alaska cell
    df = df[(df["lon"] >= -175) & (df["lon"] <= -125) & (df["lat"] >= 45) & (df["lat"] <= 75)]

    if df.empty:
        print(f"  Midpoint {midpoint}: no points within Alaska bounding box")
        continue

    dist, idx = trace_tree.query(df[["lon", "lat"]].values, k=1,
                                  distance_upper_bound=MAX_MATCH_DIST)
    valid = np.isfinite(dist)
    df = df[valid].copy()
    df["cell_idx"] = idx[valid]
    df["trace_lat"] = trace_cells.loc[df["cell_idx"], "lat"].values
    df["trace_lon"] = trace_cells.loc[df["cell_idx"], "lon"].values

    agg = (
        df.groupby(["trace_lat", "trace_lon"])[["C", "B", "U"]]
        .mean()
        .reset_index()
        .rename(columns={"trace_lat": "lat", "trace_lon": "lon"})
    )
    agg["midpoint"] = midpoint

    n_cells_matched = len(agg)
    print(f"  Midpoint {midpoint}: {len(df)} REVEALS points -> "
          f"{n_cells_matched}/{len(trace_cells)} TraCE cells matched")
    if n_cells_matched < len(trace_cells):
        matched_cells = set(zip(agg["lat"].round(4), agg["lon"].round(4)))
        all_cells = set(zip(trace_cells["lat"].round(4), trace_cells["lon"].round(4)))
        missing_cells = all_cells - matched_cells
        print(f"    Unmatched cells: {missing_cells}")

    veg_records.append(agg)

veg_by_midpoint = pd.concat(veg_records, ignore_index=True)
print(f"\nTotal veg_by_midpoint rows: {len(veg_by_midpoint)}")


# ============================================================
# Assign each TraCE year to a REVEALS midpoint
# ============================================================

trace_path = Path(paleocliamte_data_root) / "trace21ka_processed" / "trace_monthly_with_lags_full_paleo.parquet"
trace = pd.read_parquet(trace_path)
print(f"\nLoaded TraCE climate data: {len(trace):,} rows")
print(f"  Year range: {trace['year'].min()} to {trace['year'].max()}")

trace["midpoint"] = pd.cut(trace["year"], bins=bin_edges, labels=bin_labels, right=True).astype(float)

n_unmatched = trace["midpoint"].isna().sum()
print(f"\nRows with no REVEALS midpoint match: {n_unmatched:,} ({100*n_unmatched/len(trace):.2f}%)")
if n_unmatched > 0:
    unmatched_years = trace.loc[trace["midpoint"].isna(), "year"]
    print(f"  Unmatched year range: {unmatched_years.min()} to {unmatched_years.max()}")
    print(f"  (expected: years < -9750, years > 1950, and years in the -4750/-4720 gap)")


# ============================================================
# Merge land cover onto the climate data
# ============================================================

merged = trace.merge(veg_by_midpoint, on=["lat", "lon", "midpoint"], how="left")

n_missing_veg = merged["C"].isna().sum()
print(f"\nFinal merged rows: {len(merged):,}")
print(f"  Rows missing land cover after merge: {n_missing_veg:,} "
      f"({100*n_missing_veg/len(merged):.2f}%)")

out_path = Path(paleocliamte_data_root) / "trace21ka_processed" / "trace_monthly_with_lags_and_landcover.parquet"
merged.to_parquet(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"  {len(merged):,} rows x {len(merged.columns)} columns")

gap_check = trace[(trace["year"] > -4750) & (trace["year"] <= -4250) & trace["midpoint"].isna()]
len(gap_check)   
