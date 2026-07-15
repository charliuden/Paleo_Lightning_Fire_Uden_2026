"""
Main entry point. Loads drivers + parameters once, precomputes deterministic
quantities once, then loops N times doing only the stochastic draws + metrics
+ aggregation + immediate write.

BEFORE RUNNING THIS: run export_parameters_from_R.R first. It reads your
.rds model objects and writes flat parameter files into
<rds_root>/exported_params/ that load_model_parameters() below expects:

    exported_params/
        lightning_coefs.json        {a_alpha, b_alpha, ..., f_beta}  (B12 posterior means)
        strike_count_nb.json        {gamma_0, gamma: {swr, tair, rh, precip, sp, wind}}
        ignition_coefs.json         {alpha, beta: {...}, cal_shift_bayes, rate_cal_shift}
        burned_area_coefs.json      {alpha, beta: {...}, phi_mean}
        mu_sigma_lightning.csv, mu_sigma_lightning_count.csv,
        mu_sigma_ignition.csv, mu_sigma_burned_area.csv   (same format as your existing files)

This mirrors exactly the posterior-MEAN coefficients your R script already
extracts with `mean(posterior_samples$...)` / `coef_means$...` -- nothing new
to fit, just re-saved as JSON so Python can read them without rpy2/pystan.
"""

import json
import os
import numpy as np
import pandas as pd

from monte_carlo_fire_pipeline import (
    PipelineConfig, IncrementalCSVWriter,
    compute_metrics_for_submodel, build_timeseries_rows,
    build_spatial_rows, build_distribution_rows,
    expected_calibration_error, pod_far_csi,
)

config_path = "/raid/cuden/config_files/paleo_config.properties"


# --------------------------------------------------------------------------- #
# STEP 1: LOAD DRIVERS + PARAMETERS (once)
# --------------------------------------------------------------------------- #

def standardize(df: pd.DataFrame, mu_sigma: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    mu = dict(zip(mu_sigma["var"], mu_sigma["mu"]))
    sigma = dict(zip(mu_sigma["var"], mu_sigma["sigma"]))
    for c in cols:
        df[c] = (df[c] - mu[c]) / sigma[c]
    return df


def load_drivers(cfg: PipelineConfig) -> dict:
    """Loads the same driver CSVs your R script reads, mirroring each block."""
    d = {}
    d["lightning"] = pd.read_csv(os.path.join(cfg.drivers_root,
                        "lightning_strike_rate_climate_jja_2002_2011.csv"))
    d["strike_count"] = pd.read_csv(os.path.join(cfg.drivers_root,
                        "lightning_strike_count_climate_jja_2002_2011.csv"))
    d["fire"] = pd.read_csv(os.path.join(cfg.drivers_root,
                        "lightning_fires_burned_area_jja_2002_2011.csv"))
    d["fire"]["lat"] = d["fire"]["lat"].round(2)
    d["fire"]["lon"] = d["fire"]["lon"].round(2)
    d["fire"] = d["fire"].drop_duplicates(subset=["lat", "lon", "year", "month"])
    d["area"] = pd.read_csv("~/raid/cuden/data/paleo_fire/era5_grid_area.csv")
    d["proximity"] = pd.read_csv(os.path.join(cfg.proximity_data_root,
                        "era5_proximity_road_river_urban.csv"))
    d["proximity"]["lat"] = d["proximity"]["lat"].round(2)
    d["proximity"]["lon"] = d["proximity"]["lon"].round(2)
    return d


def load_model_parameters(cfg: PipelineConfig) -> dict:
    """See module docstring for expected file layout."""
    p = os.path.join(cfg.rds_root, "exported_params")
    params = {}
    with open(os.path.join(p, "lightning_coefs.json")) as f:
        params["lightning"] = json.load(f)
    with open(os.path.join(p, "strike_count_nb.json")) as f:
        params["strike_count"] = json.load(f)
    with open(os.path.join(p, "ignition_coefs.json")) as f:
        params["ignition"] = json.load(f)
    with open(os.path.join(p, "burned_area_coefs.json")) as f:
        params["burned_area"] = json.load(f)
    params["mu_sigma_lightning"] = pd.read_csv(
        os.path.join(cfg.mu_sigma_root, "mu_sigma_lightning_predictors.csv"))
    params["mu_sigma_strike_count"] = pd.read_csv(
        os.path.join(cfg.mu_sigma_root, "mu_sigma_lightning_count_predictors.csv"))
    params["mu_sigma_ignition"] = pd.read_csv(
        os.path.join(cfg.mu_sigma_root, "mu_sigma_ignition_predictors.csv"))
    params["mu_sigma_burned_area"] = pd.read_csv(
        os.path.join(cfg.mu_sigma_root, "mu_sigma_burned_area_predictors.csv"))

    _validate_named(params["strike_count"]["gamma"], "strike_count_nb.json -> gamma",
                     ["swr", "tair", "rh", "precip", "sp", "wind"])
    _validate_named(params["ignition"]["beta"], "ignition_coefs.json -> beta",
                     ["rh", "tair", "precip", "precip_1m", "precip_2m", "precip_3m",
                      "precip_5y", "tair_1m", "tair_2m", "tair_3m", "tair_5y", "B"])
    _validate_named(params["burned_area"]["beta"], "burned_area_coefs.json -> beta",
                     ["rh", "tair", "precip", "wind", "precip_1m", "precip_3m", "precip_5y",
                      "tair_2m", "tair_5y", "road_proximity_binary", "urban_proximity_binary"])
    return params


def _validate_named(obj, label: str, expected_keys: list[str]):
    """
    Fails loudly and specifically if a coefficient vector exported from R came
    through as an unnamed JSON array instead of a named object (this happens
    when the R-side vector lost its names() before write_json()) -- rather
    than letting it surface later as a confusing 'list indices must be
    integers' TypeError deep in precompute_deterministic().
    """
    if not isinstance(obj, dict):
        raise TypeError(
            f"{label} loaded as a {type(obj).__name__}, not a dict -- the R export "
            f"script must attach names() to this coefficient vector before "
            f"write_json(). Expected keys: {expected_keys}"
        )
    missing = set(expected_keys) - set(obj.keys())
    if missing:
        raise KeyError(f"{label} is missing expected coefficient names: {missing}")


# --------------------------------------------------------------------------- #
# STEP 2: DETERMINISTIC PRECOMPUTE (once) -- mirrors each R chunk exactly,
# stopping BEFORE the rgamma/rnbinom/rbinom calls (those move into the loop)
# --------------------------------------------------------------------------- #

def precompute_deterministic(drivers: dict, params: dict) -> dict:
    out = {}

    # ---- lightning (r_strike): alpha_pred, beta_pred ----
    lp = params["lightning"]
    lc = standardize(drivers["lightning"], params["mu_sigma_lightning"],
                      ["precip", "tair", "wind", "swr", "sp", "rh"])
    lp_alpha = (lp["a_alpha"] + lp["b_alpha"] * lc["swr"] + lp["c_alpha"] * lc["tair"]
                + lp["d_alpha"] * lc["rh"] + lp["e_alpha"] * lc["precip"] + lp["f_alpha"] * lc["sp"])
    lp_beta = (lp["a_beta"] + lp["b_beta"] * lc["swr"] + lp["c_beta"] * lc["tair"]
               + lp["d_beta"] * lc["rh"] + lp["e_beta"] * lc["precip"] + lp["f_beta"] * lc["sp"])
    lc["alpha_pred"] = np.exp(lp_alpha)
    lc["beta_pred"] = np.exp(lp_beta)
    out["lightning_scaled"] = lc[["lat", "lon", "year", "strikes", "alpha_pred", "beta_pred"]]

    # ---- strike count (n_strikes): mu_r_strike_gridcell, nb_theta ----
    # NOTE: r_strike itself is stochastic (drawn from alpha_pred/beta_pred), so
    # mu_r_strike_gridcell must be recomputed INSIDE the loop, after r_strike is
    # drawn. Here we precompute only the deterministic piece: nb_theta.
    df = drivers["strike_count"].merge(
        out["lightning_scaled"][["lat", "lon", "year", "strikes"]],
        on=["lat", "lon", "year"], how="left"
    ).dropna()
    df = df.rename(columns={"strikes": "r_strike_obs", "strike_count": "n_strikes_obs"})
    df = df.merge(drivers["area"][["lat", "lon", "area"]], on=["lat", "lon"], how="left")
    df = standardize(df, params["mu_sigma_strike_count"],
                      ["precip", "tair", "wind", "rh", "swr", "sp"])
    sc = params["strike_count"]
    theta_predictors = ["swr", "tair", "rh", "precip", "sp", "wind"]
    log_theta = sc["gamma_0"] + sum(sc["gamma"][v] * df[v] for v in theta_predictors)
    df["nb_theta"] = np.exp(log_theta)
    out["strike_count_df"] = df

    # ---- ignition efficiency (p_ignite): deterministic point estimate ----
    fire = drivers["fire"]
    ic = standardize(fire, params["mu_sigma_ignition"],
                      ["precip", "tair", "wind", "rh", "precip_1m", "precip_2m", "precip_3m",
                       "precip_5y", "tair_1m", "tair_2m", "tair_3m", "tair_5y", "B"])
    ip = params["ignition"]
    predictors_m2 = ["rh", "tair", "precip", "precip_1m", "precip_2m", "precip_3m",
                      "precip_5y", "tair_1m", "tair_2m", "tair_3m", "tair_5y", "B"]
    linpred = ip["alpha"] + sum(ip["beta"][v] * ic[v] for v in predictors_m2)
    eps = 1e-8
    p_ignite_raw = 1 / (1 + np.exp(-linpred))
    linpred_cal = np.clip(linpred + ip["cal_shift_bayes"],
                           np.log(eps / (1 - eps)), np.log((1 - eps) / eps))
    p_ignite_cal = 1 / (1 + np.exp(-linpred_cal))
    linpred_rate_cal = np.clip(
        np.log(np.clip(p_ignite_cal, eps, 1 - eps) / (1 - np.clip(p_ignite_cal, eps, 1 - eps)))
        + ip["rate_cal_shift"],
        np.log(eps / (1 - eps)), np.log((1 - eps) / eps))
    fire = fire.copy()
    fire["p_ignite_raw"] = p_ignite_raw
    fire["p_ignite_rate_cal"] = 1 / (1 + np.exp(-linpred_rate_cal))
    out["fire_with_pignite"] = fire

    # ---- burned area: gamma_shape, gamma_rate (deterministic) ----
    bp = params["burned_area"]
    ba = standardize(fire, params["mu_sigma_burned_area"],
                      ["rh", "tair", "precip", "wind", "precip_1m", "precip_3m",
                       "precip_5y", "tair_2m", "tair_5y"])
    ba = ba.merge(drivers["proximity"][["lat", "lon", "road_proximity_binary",
                                          "urban_proximity_binary"]],
                   on=["lat", "lon"], how="left")
    preds_final = ["rh", "tair", "precip", "wind", "precip_1m", "precip_3m", "precip_5y",
                    "tair_2m", "tair_5y", "road_proximity_binary", "urban_proximity_binary"]
    linpred_ba = bp["alpha"] + sum(bp["beta"][v] * ba[v] for v in preds_final)
    ba["gamma_mean"] = np.exp(linpred_ba)
    ba["gamma_shape"] = bp["phi_mean"]  # constant across rows
    ba["gamma_rate"] = ba["gamma_shape"] / ba["gamma_mean"]
    out["burned_area_df"] = ba[["lat", "lon", "year", "month", "gamma_shape", "gamma_rate"]]

    return out


# --------------------------------------------------------------------------- #
# STEP 3: SINGLE STOCHASTIC RUN
# --------------------------------------------------------------------------- #

def simulate_one_run(pre: dict, rng: np.random.Generator) -> pd.DataFrame:
    """
    Draws ONE stochastic realization of the full pipeline, using the
    deterministic parameters computed once in `pre`. Returns a single
    gridcell-month-level dataframe with everything needed for metrics.
    """
    # --- r_strike ~ Gamma(alpha_pred, beta_pred) ---
    lc = pre["lightning_scaled"].copy()
    lc["r_strike"] = rng.gamma(lc["alpha_pred"], 1 / lc["beta_pred"])

    # --- n_strikes ~ NegBinomial(mu = r_strike * area, theta) ---
    sc = pre["strike_count_df"].merge(
        lc[["lat", "lon", "year", "r_strike"]], on=["lat", "lon", "year"], how="left"
    )
    mu = np.clip(sc["r_strike"] * sc["area"], 1e-8, None)
    # numpy's negative_binomial parameterizes by (n=size, p); convert from (mu, theta)
    theta = sc["nb_theta"].to_numpy()
    p = theta / (theta + mu.to_numpy())
    sc["n_strikes"] = rng.negative_binomial(theta, p)

    # --- fire_count ~ Binomial(n_strikes, p_ignite_rate_cal) ---
    fire = pre["fire_with_pignite"].merge(
        sc[["lat", "lon", "year", "month", "r_strike_obs", "n_strikes_obs",
            "r_strike", "n_strikes"]],
        on=["lat", "lon", "year", "month"], how="left"
    )
    n_strikes_int = np.nan_to_num(fire["n_strikes"], nan=0).astype(int)
    fire["fire_count_pred"] = rng.binomial(n_strikes_int, fire["p_ignite_rate_cal"].fillna(0))

    # --- total_burned_area = sum of Gamma draws, one per predicted fire ---
    fire = fire.merge(pre["burned_area_df"], on=["lat", "lon", "year", "month"], how="left")

    def draw_burned_area(n, shape, rate):
        if n == 0 or pd.isna(n) or pd.isna(shape) or pd.isna(rate):
            return 0.0
        return float(rng.gamma(shape, 1 / rate, int(n)).sum())

    fire["total_burned_area"] = [
        draw_burned_area(n, s, r) for n, s, r in
        zip(fire["fire_count_pred"], fire["gamma_shape"], fire["gamma_rate"])
    ]
    fire["p_ignite"] = fire["p_ignite_rate_cal"]  # convenience alias for aggregation spec

    return fire


# --------------------------------------------------------------------------- #
# STEP 4: METRICS FOR ONE RUN (all submodels, all scales)
# --------------------------------------------------------------------------- #

def compute_all_metrics(fire: pd.DataFrame, run_id: int, nbins: int) -> list[dict]:
    """
    NOTE: r_strike metrics use `fire` (gridcell-MONTH resolution, via the merge
    in simulate_one_run), not the original gridcell-YEAR lightning_scaled frame
    -- `fire` already carries both r_strike_obs and the drawn r_strike at the
    resolution everything else is computed at, so no separate lookup is needed.
    """
    rows = []

    # r_strike: predicted vs observed rate
    rows += compute_metrics_for_submodel(
        fire, obs_col="r_strike_obs", pred_col="r_strike", submodel="r_strike",
        run_id=run_id, include_zero_match=False, nbins=nbins)

    # n_strikes: predicted vs observed count
    rows += compute_metrics_for_submodel(
        fire, obs_col="n_strikes_obs", pred_col="n_strikes", submodel="n_strikes",
        run_id=run_id, include_dispersion=True, nbins=nbins)

    # p_ignite: raw stage -- weighted corr + ECE against empirical rate
    eps = 1e-8
    fire = fire.copy()
    fire["empirical_rate"] = np.where(
        fire["n_strikes_obs"] > 0, fire["fire_count"] / fire["n_strikes_obs"].replace(0, np.nan), np.nan
    ) if "fire_count" in fire.columns else np.nan
    for stage, col in [("raw", "p_ignite_raw"), ("rate_calibrated", "p_ignite_rate_cal")]:
        if "empirical_rate" in fire.columns and fire["empirical_rate"].notna().any():
            ece = expected_calibration_error(
                fire["empirical_rate"].to_numpy(float), fire[col].to_numpy(float),
                fire["n_strikes_obs"].to_numpy(float))
            wcorr = weighted_corr_safe(fire, "empirical_rate", col, "n_strikes_obs")
            rows.append(dict(run_id=run_id, submodel="p_ignite", calibration_stage=stage,
                              scale="gridcell_month_paired", metric="ece", value=ece))
            rows.append(dict(run_id=run_id, submodel="p_ignite", calibration_stage=stage,
                              scale="gridcell_month_paired", metric="weighted_corr", value=wcorr))
        rows.append(dict(run_id=run_id, submodel="p_ignite", calibration_stage=stage,
                          scale="overall", metric="bias_ratio",
                          value=safe_ratio(fire["n_strikes"] * fire[col], fire.get("fire_count"))))

    # n_fires: predicted vs observed fire count (counts + categorical POD/FAR/CSI)
    rows += compute_metrics_for_submodel(
        fire, obs_col="fire_count", pred_col="fire_count_pred", submodel="n_fires",
        run_id=run_id, nbins=nbins)
    if "fire_count" in fire.columns:
        cats = pod_far_csi((fire["fire_count"] > 0).to_numpy(),
                            (fire["fire_count_pred"] > 0).to_numpy())
        for k, v in cats.items():
            rows.append(dict(run_id=run_id, submodel="n_fires", calibration_stage="final",
                              scale="gridcell_month_paired", metric=k, value=v))

    # total_burned_area: log-scale metrics
    if "burned_area_km2" in fire.columns:
        rows += compute_metrics_for_submodel(
            fire, obs_col="burned_area_km2", pred_col="total_burned_area",
            submodel="burned_area", run_id=run_id, log_transform=True,
            include_zero_match=False, nbins=nbins)

    return rows


def weighted_corr_safe(df, obs_col, pred_col, weight_col):
    from monte_carlo_fire_pipeline import weighted_corr
    ok = df[obs_col].notna()
    if ok.sum() < 2:
        return np.nan
    return weighted_corr(df.loc[ok, obs_col].to_numpy(float),
                          df.loc[ok, pred_col].to_numpy(float),
                          df.loc[ok, weight_col].to_numpy(float))


def safe_ratio(numer, denom):
    if denom is None:
        return np.nan
    n, d = np.nansum(numer), np.nansum(denom)
    return float(n / d) if d else np.nan


# --------------------------------------------------------------------------- #
# STEP 5: MAIN LOOP
# --------------------------------------------------------------------------- #

def run_pipeline(config_path: str, n_runs: int = 100, seed: int = 123):
    cfg = PipelineConfig.from_properties(config_path, n_runs=n_runs, seed=seed)

    print("Loading drivers and parameters (once)...")
    drivers = load_drivers(cfg)
    params = load_model_parameters(cfg)

    print("Precomputing deterministic quantities (once)...")
    pre = precompute_deterministic(drivers, params)

    writers = {
        "metrics": IncrementalCSVWriter(os.path.join(cfg.output_dir, "metrics_by_iteration.csv")),
        "timeseries": IncrementalCSVWriter(os.path.join(cfg.output_dir, "timeseries_by_run_year.csv")),
        "spatial": IncrementalCSVWriter(os.path.join(cfg.output_dir, "spatial_by_run_gridcell.csv")),
        "distribution": IncrementalCSVWriter(os.path.join(cfg.output_dir, "distribution_summary_by_run.csv")),
    }

    rng = np.random.default_rng(cfg.seed)
    for run_id in range(cfg.n_runs):
        fire = simulate_one_run(pre, rng)

        writers["metrics"].write(
            compute_all_metrics(fire, run_id, cfg.nbins_perkins))
        writers["timeseries"].write(build_timeseries_rows(fire, run_id))
        writers["spatial"].write(build_spatial_rows(fire, run_id))
        writers["distribution"].write(build_distribution_rows(fire, run_id))

        del fire  # explicit: nothing from this run persists past this point
        if (run_id + 1) % 10 == 0:
            print(f"  completed run {run_id + 1}/{cfg.n_runs}")

    print(f"Done. Outputs written to {cfg.output_dir}")


if __name__ == "__main__":
    run_pipeline(config_path, n_runs=100, seed=123)
