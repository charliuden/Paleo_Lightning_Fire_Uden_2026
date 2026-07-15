"""
Monte Carlo ensemble simulation of the lightning-fire pipeline.

Repeats the stochastic-realization pipeline (r_strike -> n_strikes -> p_ignite
-> fire_count -> burned_area) N times using FIXED posterior-mean parameters,
to quantify PROCESS uncertainty (i.e. how much outcomes vary just from the
randomness inherent in the stochastic draws, holding the fitted model fixed).

Architecture:
    1. Load drivers + parameters ONCE.
    2. Compute all DETERMINISTIC quantities ONCE (alpha_pred, beta_pred,
       theta_pred, p_ignite, gamma_shape, gamma_rate) -- these don't change
       across runs because they only depend on posterior-MEAN coefficients.
    3. Loop over N runs. Each run only re-draws the stochastic variates
       (rgamma, rnbinom, rbinom, rgamma-sum), computes metrics + aggregates,
       and WRITES IMMEDIATELY to disk (append mode). Nothing from a run is
       kept in memory once it's written -- this bounds peak memory
       regardless of how many runs you do.

Outputs (all under <predictions_root>/parameter_uncertainty/):
    metrics_by_iteration.csv       - performance metrics, one row per
                                      (run, submodel, calibration_stage, scale, metric)
    timeseries_by_run_year.csv     - one row per (run, year, submodel, stat_type)
    spatial_by_run_gridcell.csv    - one row per (run, lat, lon, submodel, stat_type)
    distribution_summary_by_run.csv- one row per (run, submodel) with columns
                                      mean/median/min/max/sd/var/q25/q75

NOTE ON .rds FILES: the Bayesian model objects (B12, ignition_bayes_obj,
m6_obj, nb_model) mix simple posterior-mean coefficients with full stanfit
objects. Rather than reading .rds directly from Python, export the pieces
you need from R into flat files (see `export_parameters_from_R.R` companion
script / the `load_model_parameters()` docstring below for the exact fields
expected).
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

def read_properties(file_path: str) -> dict:
    """Python port of the R read_properties() function."""
    props = {}
    with open(file_path) as f:
        for line in f:
            if "=" in line:
                key, val = line.split("=", 1)
                props[key.strip()] = val.strip()
    return props


@dataclass
class PipelineConfig:
    drivers_root: str
    rds_root: str  # holds exported-parameter flat files now, not .rds
    predictions_root: str
    mu_sigma_root: str
    proximity_data_root: str
    n_runs: int = 100
    seed: int = 123
    nbins_perkins: int = 15

    @property
    def output_dir(self) -> str:
        d = os.path.join(self.predictions_root, "parameter_uncertainty")
        os.makedirs(d, exist_ok=True)
        return d

    @classmethod
    def from_properties(cls, path: str, n_runs: int = 100, seed: int = 123):
        p = read_properties(path)
        return cls(
            drivers_root=p["drivers_root"],
            rds_root=p["rds_root"],
            predictions_root=p["predictions_root"],
            mu_sigma_root=p["mu_sigma_root"],
            proximity_data_root=p["proximity_data_root"],
            n_runs=n_runs,
            seed=seed,
        )


# --------------------------------------------------------------------------- #
# METRIC FUNCTIONS
# --------------------------------------------------------------------------- #

def nrmse(obs: np.ndarray, pred: np.ndarray) -> float:
    """RMSE normalized by the observed range. NaNs dropped pairwise."""
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[ok], pred[ok]
    if len(obs) == 0 or np.ptp(obs) == 0:
        return np.nan
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    return rmse / np.ptp(obs)


def pearson_r(obs: np.ndarray, pred: np.ndarray) -> float:
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[ok], pred[ok]
    if len(obs) < 2 or np.std(obs) == 0 or np.std(pred) == 0:
        return np.nan
    return float(np.corrcoef(obs, pred)[0, 1])


def spearman_r(obs: np.ndarray, pred: np.ndarray) -> float:
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[ok], pred[ok]
    if len(obs) < 2:
        return np.nan
    # rank-based Pearson correlation == Spearman
    from scipy.stats import spearmanr
    r, _ = spearmanr(obs, pred)
    return float(r)


def perkins_skill_score(obs: np.ndarray, pred: np.ndarray, nbins: int = 15) -> float:
    """
    Perkins Skill Score (PSS): overlap between the obs and pred probability
    density histograms, on a SHARED set of bin edges. 1 = identical
    distributions, 0 = no overlap. Does not require obs/pred to be paired --
    valid for the "pooled, unpaired" scale.
    """
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[ok], pred[ok]
    if len(obs) == 0 or len(pred) == 0:
        return np.nan
    lo, hi = min(obs.min(), pred.min()), max(obs.max(), pred.max())
    if lo == hi:
        return np.nan
    breaks = np.linspace(lo, hi, nbins + 1)
    hist_obs, _ = np.histogram(obs, bins=breaks)
    hist_pred, _ = np.histogram(pred, bins=breaks)
    prob_obs = hist_obs / hist_obs.sum()
    prob_pred = hist_pred / hist_pred.sum()
    return float(np.minimum(prob_obs, prob_pred).sum())


def zero_match(obs: np.ndarray, pred: np.ndarray) -> float:
    """|P(obs==0) - P(pred==0)| -- how well the model captures zero-inflation."""
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[ok], pred[ok]
    if len(obs) == 0:
        return np.nan
    return float(abs(np.mean(obs == 0) - np.mean(pred == 0)))


def dispersion_ratio(obs: np.ndarray, pred: np.ndarray) -> float:
    """var(pred)/var(obs). <1 means the model underdisperses (too smooth)."""
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[ok], pred[ok]
    vo = np.var(obs)
    if vo == 0 or len(obs) == 0:
        return np.nan
    return float(np.var(pred) / vo)


def count_ratio(obs: np.ndarray, pred: np.ndarray) -> float:
    """sum(pred)/sum(obs). The headline aggregate-bias number."""
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[ok], pred[ok]
    so = obs.sum()
    if so == 0:
        return np.nan
    return float(pred.sum() / so)


def weighted_corr(obs: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Pearson correlation -- used for p_ignite, weighted by strike count."""
    ok = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(weights) & (weights > 0)
    obs, pred, w = obs[ok], pred[ok], weights[ok]
    if len(obs) < 2:
        return np.nan
    mo = np.average(obs, weights=w)
    mp = np.average(pred, weights=w)
    cov = np.average((obs - mo) * (pred - mp), weights=w)
    vo = np.average((obs - mo) ** 2, weights=w)
    vp = np.average((pred - mp) ** 2, weights=w)
    if vo == 0 or vp == 0:
        return np.nan
    return float(cov / np.sqrt(vo * vp))


def expected_calibration_error(
    obs_rate: np.ndarray, pred_prob: np.ndarray, weights: np.ndarray, n_bins: int = 10
) -> float:
    """
    ECE for a probability model validated against sparse binary-ish outcomes.
    Bins gridcell-months by predicted probability decile, pools
    sum(events)/sum(trials) WITHIN each bin (this is what fixes the
    low-n-per-cell noise problem -- a single low-strike-count gridcell-month
    has an unreliable empirical rate on its own, but pooling within a bin
    gives a stable estimate), then reports the weighted mean absolute gap
    between predicted and pooled-observed rate.

    obs_rate:  empirical rate per row (e.g. fire_count / strike_count)
    pred_prob: predicted probability per row (e.g. p_ignite)
    weights:   trials per row (e.g. strike_count) -- used both for binning
               weight and for pooling the observed rate within each bin
    """
    ok = np.isfinite(obs_rate) & np.isfinite(pred_prob) & np.isfinite(weights) & (weights > 0)
    obs_rate, pred_prob, weights = obs_rate[ok], pred_prob[ok], weights[ok]
    if len(pred_prob) == 0:
        return np.nan
    bins = pd.qcut(pred_prob, q=min(n_bins, len(np.unique(pred_prob))), duplicates="drop")
    df = pd.DataFrame({"bin": bins, "obs_rate": obs_rate, "pred": pred_prob, "w": weights})
    ece, total_w = 0.0, weights.sum()
    for _, g in df.groupby("bin", observed=True):
        pooled_obs = np.average(g["obs_rate"], weights=g["w"])
        pooled_pred = np.average(g["pred"], weights=g["w"])
        ece += (g["w"].sum() / total_w) * abs(pooled_obs - pooled_pred)
    return float(ece)


def pod_far_csi(obs_binary: np.ndarray, pred_binary: np.ndarray) -> dict:
    """Standard categorical verification stats for rare-event presence/absence."""
    ok = np.isfinite(obs_binary) & np.isfinite(pred_binary)
    obs_binary, pred_binary = obs_binary[ok].astype(bool), pred_binary[ok].astype(bool)
    hits = np.sum(obs_binary & pred_binary)
    misses = np.sum(obs_binary & ~pred_binary)
    false_alarms = np.sum(~obs_binary & pred_binary)
    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan
    return {"pod": float(pod), "far": float(far), "csi": float(csi)}


# --------------------------------------------------------------------------- #
# MULTI-SCALE METRIC COMPUTATION
# --------------------------------------------------------------------------- #

def compute_metrics_for_submodel(
    df: pd.DataFrame,
    obs_col: str,
    pred_col: str,
    submodel: str,
    run_id: int,
    calibration_stage: str = "final",
    log_transform: bool = False,
    weight_col: Optional[str] = None,
    include_zero_match: bool = True,
    include_dispersion: bool = False,
    nbins: int = 15,
) -> list[dict]:
    """
    Computes the standard metric battery at all four scales:
      gridcell_month_pooled (unpaired, PSS only -- doesn't need pairing)
      gridcell_month_paired (raw point-by-point -- noisiest, report w/ caveat)
      spatial (per-gridcell totals/means over all 10 years, then correlated)
      temporal (per-year totals/means over all gridcells, then correlated)
      overall (single 10-year aggregate ratio)

    Returns a list of tidy-row dicts ready to append to metrics_by_iteration.csv.
    """
    rows = []
    obs, pred = df[obs_col].to_numpy(float), df[pred_col].to_numpy(float)

    if log_transform:
        obs_t, pred_t = np.log1p(obs), np.log1p(pred)
    else:
        obs_t, pred_t = obs, pred

    def add(scale, metric, value):
        rows.append(dict(run_id=run_id, submodel=submodel,
                          calibration_stage=calibration_stage,
                          scale=scale, metric=metric, value=value))

    # --- gridcell_month_pooled (unpaired) ---
    add("gridcell_month_pooled", "perkins_skill", perkins_skill_score(obs_t, pred_t, nbins))

    # --- gridcell_month_paired ---
    add("gridcell_month_paired", "nrmse", nrmse(obs_t, pred_t))
    add("gridcell_month_paired", "spearman", spearman_r(obs, pred))
    if weight_col is not None:
        add("gridcell_month_paired", "weighted_corr",
            weighted_corr(obs, pred, df[weight_col].to_numpy(float)))
    if include_zero_match:
        add("gridcell_month_paired", "zero_match", zero_match(obs, pred))
    if include_dispersion:
        add("gridcell_month_paired", "dispersion_ratio", dispersion_ratio(obs, pred))

    # --- spatial (aggregate over time, per gridcell, then correlate) ---
    spatial = df.groupby(["lat", "lon"], as_index=False)[[obs_col, pred_col]].sum()
    add("spatial", "correlation", pearson_r(spatial[obs_col].to_numpy(float),
                                             spatial[pred_col].to_numpy(float)))

    # --- temporal (aggregate over space, per year, then correlate) ---
    temporal = df.groupby("year", as_index=False)[[obs_col, pred_col]].sum()
    add("temporal", "correlation", pearson_r(temporal[obs_col].to_numpy(float),
                                              temporal[pred_col].to_numpy(float)))

    # --- overall (single ratio) ---
    add("overall", "bias_ratio", count_ratio(obs, pred))

    return rows


# --------------------------------------------------------------------------- #
# AGGREGATION FOR PLOTTING OUTPUTS (timeseries / spatial / distribution)
# --------------------------------------------------------------------------- #

# submodel -> which stats to compute, and which column(s) supply them
AGGREGATION_SPEC = {
    "r_strike":     {"value_col": "r_strike",     "stats": ["mean"]},
    "n_strikes":    {"value_col": "n_strikes",    "stats": ["sum", "mean"]},
    "p_ignite":     {"value_col": "p_ignite",     "stats": ["mean"], "weight_col": "n_strikes"},
    "n_fires":      {"value_col": "fire_count_pred", "stats": ["sum", "mean"]},
    "burned_area":  {"value_col": "total_burned_area", "stats": ["sum", "mean"]},
}


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.average(values[ok], weights=weights[ok]))


def build_timeseries_rows(df: pd.DataFrame, run_id: int) -> list[dict]:
    """One row per (run, year, submodel, stat_type)."""
    rows = []
    for submodel, spec in AGGREGATION_SPEC.items():
        col = spec["value_col"]
        for year, g in df.groupby("year"):
            for stat in spec["stats"]:
                val = g[col].sum() if stat == "sum" else g[col].mean()
                rows.append(dict(run_id=run_id, year=year, submodel=submodel,
                                  stat_type=stat, value=float(val)))
            if "weight_col" in spec:
                wval = _weighted_mean(g[col].to_numpy(float), g[spec["weight_col"]].to_numpy(float))
                rows.append(dict(run_id=run_id, year=year, submodel=submodel,
                                  stat_type="weighted_mean", value=wval))
    return rows


def build_spatial_rows(df: pd.DataFrame, run_id: int) -> list[dict]:
    """One row per (run, lat, lon, submodel, stat_type), aggregated over all 10 years."""
    rows = []
    for submodel, spec in AGGREGATION_SPEC.items():
        col = spec["value_col"]
        grouped = df.groupby(["lat", "lon"])
        for (lat, lon), g in grouped:
            for stat in spec["stats"]:
                val = g[col].sum() if stat == "sum" else g[col].mean()
                rows.append(dict(run_id=run_id, lat=lat, lon=lon, submodel=submodel,
                                  stat_type=stat, value=float(val)))
            if "weight_col" in spec:
                wval = _weighted_mean(g[col].to_numpy(float), g[spec["weight_col"]].to_numpy(float))
                rows.append(dict(run_id=run_id, lat=lat, lon=lon, submodel=submodel,
                                  stat_type="weighted_mean", value=wval))
    return rows


def build_distribution_rows(df: pd.DataFrame, run_id: int) -> list[dict]:
    """
    One row per (run, submodel): distribution of the RAW gridcell-month values
    across the full 10-year record (i.e. not pre-aggregated by year or gridcell).
    """
    rows = []
    for submodel, spec in AGGREGATION_SPEC.items():
        vals = df[spec["value_col"]].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        rows.append(dict(
            run_id=run_id, submodel=submodel,
            mean=float(np.mean(vals)), median=float(np.median(vals)),
            min=float(np.min(vals)), max=float(np.max(vals)),
            sd=float(np.std(vals, ddof=1)), var=float(np.var(vals, ddof=1)),
            q25=float(np.percentile(vals, 25)), q75=float(np.percentile(vals, 75)),
        ))
    return rows


# --------------------------------------------------------------------------- #
# INCREMENTAL CSV WRITER (bounds memory: one run's rows in memory at a time)
# --------------------------------------------------------------------------- #

class IncrementalCSVWriter:
    """Writes header on first call, appends (no header) on subsequent calls."""

    def __init__(self, path: str):
        self.path = path
        self._header_written = os.path.exists(path)

    def write(self, rows: list[dict]):
        if not rows:
            return
        df = pd.DataFrame(rows)
        df.to_csv(self.path, mode="a", index=False, header=not self._header_written)
        self._header_written = True
