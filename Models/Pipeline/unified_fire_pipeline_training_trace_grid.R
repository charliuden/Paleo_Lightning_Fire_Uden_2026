#!/usr/bin/env Rscript
# =============================================================================
# UNIFIED FIRE MODEL PIPELINE — posterior-sampling version
#
# Consolidates Model_pipeline_predictions.Rmd (training, ERA5 2002-2011),
# TRACE_ERA5_calibration_period_predictions_mixture.Rmd (calibration,
# 1949-1989, ERA5 or TraCE), and CopyOfpaleo_predictions.Rmd (paleo, TraCE,
# full Holocene) into one script, selected by the PERIOD switch below.
#
# KEY CHANGE FROM THE ORIGINAL SCRIPTS: every sub-model (lightning Gamma,
# strike-count NegBinomial, ignition logistic, burned-area mixture) now
# draws a full set of coefficients from its POSTERIOR at the start of each
# replicate, rather than using the posterior mean for every replicate. Each
# replicate = one complete pass of coefficients -> predictions -> sampling
# -> gridcell-area cap, for the WHOLE dataset. N_DRAWS replicates are run
# and stacked into one long-format output with a `draw_id` column, so the
# output captures both parameter uncertainty AND sampling uncertainty.
#
# RUN FROM THE TERMINAL (recommended, especially for PERIOD = "paleo"):
#   Rscript unified_fire_pipeline.R > pipeline_log_$(date +%Y%m%d_%H%M).txt 2>&1 &
#   disown
# then check progress any time with:
#   tail -f pipeline_log_*.txt
#
# OUTPUT: one CSV, path/name set in OUTPUT_FILE below. No plots — this
# script only produces predictions; make plots in a separate script that
# reads OUTPUT_FILE.
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(rstan)
  library(arrow)   # read_parquet — needed for calibration (TraCE) and paleo
  library(FNN)     # get.knnx — needed for ERA5 calibration grid-snapping
  library(evd)     # rgpd — needed for the strike-count extreme-tail correction
})

# =============================================================================
# 1. CONFIG — edit this block for each run
# =============================================================================

## ---- Which period to run -----------------------------------------------
## One of: "training", "calibration", "paleo"
PERIOD <- "training"

## Only read when PERIOD == "calibration": "era5" or "trace"
CLIMATE_SOURCE <- "era5"

## ---- Output ---------------------------------------------------------------
OUTPUT_FILE <- "predictions_training_posterior_era5_trace_grid_DIAGNOSTIC_BAinteractions_noTruncation.csv"

## ---- Posterior sampling ----------------------------------------------------
N_DRAWS <- 500      # number of full-pipeline posterior replicates
SEED    <- 123

## ---- Gridcell-area cap: years before a burned-out cell can burn again -----
## Inf = never recovers within the run (matches the training/calibration
## scripts as written). Paleo previously hardcoded 20 -- now adjustable here.
CAP_RECOVERY_YEARS <- switch(PERIOD,
                             training    = Inf,
                             calibration = Inf,
                             paleo       = 12,
                             stop("PERIOD must be 'training', 'calibration', or 'paleo'")
)

## ---- Extreme-fire threshold during Gamma sampling -------------------------
## TRUE (default): the extreme component's Gamma draws are truncated below
## at burned_area_threshold (m20's 95th-percentile classification cutoff),
## via rejection sampling -- see rgamma_truncated() below. This keeps a
## fire classified "extreme" from ever sampling an implausibly small
## magnitude, correcting the mismatch between the extreme Gamma's training
## data (fires already >= threshold) and its otherwise-unbounded support.
## FALSE: the extreme component samples an ordinary, UNTRUNCATED rgamma()
## -- set this to see how the pipeline's output responds without the
## threshold constraint (e.g. to check how much the truncation correction
## itself is driving total predicted burned area).
APPLY_EXTREME_TRUNCATION <- FALSE


## ---- Strike-count extreme-tail correction (GPD replacement) --------------
## Rather than hard-capping n_strikes at a fixed ceiling, draws exceeding
## STRIKE_GPD_THRESH are replaced with fresh samples from a Generalized
## Pareto Distribution fit to observed nonzero strike counts (peaks-over-
## threshold, ismev::gpd.fit(strike_vals, threshold = STRIKE_GPD_THRESH)).
## This preserves realistic large-but-plausible values (unlike a hard cap,
## which would pile everything above the ceiling onto one fixed number),
## while the NegBinomial model's overdispersion-driven runaway blowups
## (e.g. single draws in the thousands) are replaced by draws consistent
## with the real tail's shape. With xi < 0, the GPD has a finite implied
## upper bound at (STRIKE_GPD_THRESH - STRIKE_GPD_SIGMA / STRIKE_GPD_XI).
## Set STRIKE_GPD_THRESH <- Inf to disable the correction entirely.
STRIKE_GPD_THRESH <- 100
STRIKE_GPD_SIGMA  <- 25.5900251
STRIKE_GPD_XI     <- -0.1429466

## ---- Config file (paths shared by all periods: model RDS, mu/sigma, etc) --
read_properties <- function(file_path) {
  lines <- readLines(file_path)
  lines <- lines[grepl("=", lines)]
  key_vals <- strsplit(lines, "=")
  setNames(trimws(sapply(key_vals, `[`, 2)), trimws(sapply(key_vals, `[`, 1)))
}

# *** paleo_config.properties must be in Paleo_Lightning_Fire_Uden_2026 ***
# Use rstudioapi when running interactively inside RStudio (handles the
# script being nested arbitrarily deep inside the project). Fall back to a
# manually-set path when running via Rscript from the terminal, since there
# is no "active document" for rstudioapi to query in that case.
base_path <- "/users/c/u/cuden/raid/cuden/Paleo_Lightning_Fire_Uden_2026"

config_file_path <- file.path(base_path, "paleo_config.properties")
config <- read_properties(path.expand(config_file_path))

drivers_root           <- config[["drivers_root"]]
rds_root                <- config[["rds_root"]]
predictions_root        <- config[["predictions_root"]]
mu_sigma_root           <- config[["mu_sigma_root"]]
paleoclimate_data_root  <- config[["paleoclimate_data_root"]]
era5_data_root          <- config[["era5_data_root"]]
proximity_data_root     <- config[["proximity_data_root"]]

## ---- Period-specific input file paths --------------------------------------
## Uncomment/edit ONLY the block matching PERIOD above -- the others are
## never read, since they sit inside an `if (PERIOD == ...)` guard, but
## keeping the paths visible here makes it obvious what to change per run.

if (PERIOD == "training") {
  # --- TRAINING: ERA5, 2002-2011 ---
  lightning_rate_path  <- file.path(drivers_root, "lightning_strike_rate_climate_jja_2002_2011.csv")
  lightning_count_path <- file.path(drivers_root, "lightning_strike_count_climate_jja_2002_2011.csv")
  fire_path            <- file.path(drivers_root, "lightning_fires_burned_area_jja_2002_2011.csv")
  ignition_new_path     <- file.path(drivers_root, "lightning_ignition_jja_2002_2011_new.csv")
  ba_new_path           <- file.path(drivers_root, "burned_area_jja_2002_2011_new.csv")
  proximity_path        <- file.path(proximity_data_root, "era5_proximity_road_river_urban.csv")
  area_path             <- file.path(paleoclimate_data_root, "era5_grid_area.csv")
  area_col              <- "area"
  # Needed only for the coarse-grid diagnostic snapping step below --
  # unused otherwise, matches the path defined under PERIOD == "calibration"
  trace_cells_path      <- file.path(rds_root, "trace_cells.rds")
}

if (PERIOD == "calibration") {
  # --- CALIBRATION: ERA5 or TraCE, 1949-1989 ---
  trace_cells_path <- file.path(rds_root, "trace_cells.rds")
  
  if (CLIMATE_SOURCE == "era5") {
    climate_summer_path <- file.path(era5_data_root, "bias_correction/processed_parquet/era5_lightning_model_1946_1990.parquet")
    climate_month_path  <- file.path(era5_data_root, "bias_correction/processed_parquet/era_calib_period_monthly_with_lags.parquet")
    needs_grid_snap      <- TRUE
  } else if (CLIMATE_SOURCE == "trace") {
    climate_summer_path <- file.path(paleoclimate_data_root, "trace21ka_processed/trace_calibration_lightning_seasonal_corrected.parquet")
    climate_month_path  <- file.path(paleoclimate_data_root, "trace21ka_processed/trace_monthly_with_lags_and_landcover.parquet")
    needs_grid_snap      <- FALSE
  } else stop("CLIMATE_SOURCE must be 'era5' or 'trace'")
  
  # Modern (2002-2011) proximity + landcover, snapped to the TraCE grid --
  # used for BOTH climate sources during calibration (per original scripts)
  proximity_path       <- file.path(proximity_data_root, "era5_proximity_road_river_urban.csv")
  landcover_source_path <- file.path(drivers_root, "lightning_fires_burned_area_jja_2002_2011.csv")
  
  area_path <- file.path(paleoclimate_data_root, "TraCE_21ka_gridcell_area.csv")
  area_col  <- "area_clipped_km2"
}

if (PERIOD == "paleo") {
  # --- PALEO: TraCE, full Holocene ---
  paleo_summer_path <- file.path(paleoclimate_data_root, "trace21ka_processed/trace_lightning_seasonal_corrected.parquet")
  paleo_month_path  <- file.path(paleoclimate_data_root, "trace21ka_processed/trace_monthly_with_lags_and_landcover.parquet")
  min_year               <- -9750  # Holocene cutoff (matches `year > -9750` in original)
  landcover_freeze_year  <- 1950   # landcover reconstruction unavailable after this year
  
  area_path <- file.path(paleoclimate_data_root, "TraCE_21ka_gridcell_area.csv")
  area_col  <- "area_clipped_km2"
}

set.seed(SEED)

# =============================================================================
# 2. LOAD SHARED MODEL OBJECTS (same models regardless of period)
# =============================================================================

cat("Loading model objects...\n")

lightning_mu_sigma   <- read.csv(file.path(mu_sigma_root, "mu_sigma_lightning_predictors.csv"))
n_strike_mu_sigma    <- read.csv(file.path(mu_sigma_root, "mu_sigma_lightning_count_predictors.csv"))
ignition_mu_sigma    <- read.csv(file.path(mu_sigma_root, "mu_sigma_ignition_predictors_redo.csv"))
burned_area_mu_sigma <- read.csv(file.path(mu_sigma_root, "mu_sigma_burned_area_predictors.csv"))

get_mu    <- function(tbl, var) tbl$mu[tbl$var == var]
get_sigma <- function(tbl, var) tbl$sigma[tbl$var == var]

# --- Lightning: B12 Gamma model, full posterior available -------------------
B12 <- readRDS(file.path(rds_root, "Lightning/B12_gamma_bayes.rds"))
lightning_posterior <- as.data.frame(B12)   # one row per MCMC draw
n_lightning_draws <- nrow(lightning_posterior)

# --- Strike count: NegBinomial model -----------------------------------------
# Uses the FULL version (with the stanfit object), not neg_bin_strike_count_small.rds
# -- the "small" file deliberately omits `fit` to save space and only has
# posterior means/summaries, so it can't be sampled from.
nb_model_obj <- readRDS(file.path(rds_root, "Lightning/neg_bin_strike_count_no_outliers.rds"))
theta_predictors <- c("swr", "tair", "rh", "precip", "sp", "wind")

nb_posterior <- rstan::extract(nb_model_obj$fit, pars = c("gamma_0", "gamma"))
n_nb_draws   <- length(nb_posterior$gamma_0)

sample_nb_coefs <- function() {
  i <- sample(n_nb_draws, 1)
  list(gamma_0 = nb_posterior$gamma_0[i],
       gamma   = nb_posterior$gamma[i, ])
}

# --- Ignition efficiency: Bayesian Beta-Binomial model, full posterior via stanfit --
# Replaces the earlier plain-binomial m9 model + two-stage (binary + rate)
# calibration. The Beta-Binomial's own dispersion parameter (phi) absorbs
# extra-binomial variance directly, so only a SINGLE rate-calibration shift
# is needed downstream -- no separate binary cal_shift_bayes stage.
ignition_bayes_obj  <- readRDS(file.path(rds_root, "Ignition_Efficiency/redo/beta_binomial_ignition_bayes.rds"))
ignition_fit        <- ignition_bayes_obj$fit          # stanfit
ignition_posterior  <- rstan::extract(ignition_fit, pars = c("alpha", "beta"))
n_ignition_draws    <- length(ignition_posterior$alpha)
predictors_ignition <- ignition_bayes_obj$predictors   # c("rh", "tair", "precip", "precip_1m", "precip_3m", "precip_5y", "tair_3m", "tair_5y", "U")

rate_cal_shift <- readRDS(file.path(rds_root, "Ignition_Efficiency/redo/beta_binomial_calibration_shift.rds"))$rate_cal_shift


# --- Burned area: mixture model (classifier + extreme Gamma + typical Gamma) -
# Using m20 (formula-based, with interaction terms + VPD), fit via
# fit_mixture_gamma_formula() -- NOT m3. Predictor sets are stored as
# FORMULA OBJECTS (to preserve interaction terms), not plain predictor-name
# vectors, so downstream prediction code uses model.matrix() against these
# formulas rather than simple column subsetting.
mixture_obj  <- readRDS(file.path(rds_root, "Burned_Area/redo/m20_mixture_interactions.rds"))
fit_logit    <- mixture_obj$fit_logit
fit_extreme  <- mixture_obj$fit_extreme
fit_normal   <- mixture_obj$fit_normal

logit_posterior   <- rstan::extract(fit_logit,   pars = c("alpha", "beta"))
extreme_posterior <- rstan::extract(fit_extreme, pars = c("alpha", "beta", "phi"))
normal_posterior  <- rstan::extract(fit_normal,  pars = c("alpha", "beta", "phi"))

n_mixture_draws <- length(logit_posterior$alpha)

# m20's predictor formulas (with interactions), suppressing the intercept
# column since alpha is estimated separately in Stan.
formula_classifier <- update(mixture_obj$formula_classifier, ~ . - 1)
formula_extreme     <- update(mixture_obj$formula_extreme,     ~ . - 1)
formula_typical      <- update(mixture_obj$formula_typical,      ~ . - 1)

# m20's extreme-fire classification threshold, needed for truncated sampling
burned_area_threshold <- mixture_obj$threshold

# NOTE: the p_extreme calibration shift is deliberately NOT applied here.
# Diagnostic work comparing several shift-derivation methods produced
# inconsistent estimates that did not converge on a single trustworthy
# value. m20's raw, uncalibrated p_extreme is used directly instead.
eps <- 1e-8

cat("Model objects loaded.\n")

# =============================================================================
# 3. PREPROCESSING — the three checks, implemented as functions
# =============================================================================

## ---- Check 1: ERA5 -> TraCE grid aggregation (calibration/ERA5 only) ------
aggregate_to_trace_grid <- function(data, trace_cells, group_cols) {
  nn <- get.knnx(data = trace_cells[, c("lon", "lat")],
                 query = data[, c("lon", "lat")], k = 1)
  data$trace_lon <- trace_cells$lon[nn$nn.index]
  data$trace_lat <- trace_cells$lat[nn$nn.index]
  
  data <- data %>%
    dplyr::select(-lat, -lon) %>%
    rename(lat = trace_lat, lon = trace_lon)
  
  data %>%
    group_by(across(all_of(group_cols))) %>%
    summarise(across(where(is.numeric), \(x) mean(x, na.rm = TRUE)), .groups = "drop")
}

## ---- Check 2: proximity + landcover assignment, by period -----------------
assign_proximity_and_landcover <- function(period, climate_source, month_data,
                                           trace_cells = NULL) {
  if (period == "training") {
    prox <- read.csv(proximity_path) %>%
      dplyr::select(lon, lat, road_proximity_binary, urban_proximity_binary) %>%
      mutate(lat = round(lat, 2), lon = round(lon, 2))
    month_data <- month_data %>%
      mutate(lat = round(lat, 2), lon = round(lon, 2)) %>%
      left_join(prox, by = c("lat", "lon"))
    # landcover (B/C/U) already present in the training driver file --
    # nothing to join here
    return(month_data)
  }
  
  if (period == "calibration") {
    prox_raw <- read.csv(proximity_path)
    nn <- get.knnx(data = trace_cells[, c("lon", "lat")],
                   query = prox_raw[, c("lon", "lat")], k = 1)
    prox_raw$trace_lon <- trace_cells$lon[nn$nn.index]
    prox_raw$trace_lat <- trace_cells$lat[nn$nn.index]
    
    prox <- prox_raw %>%
      dplyr::select(-lat, -lon) %>%
      rename(lat = trace_lat, lon = trace_lon) %>%
      dplyr::select(lon, lat, dist_road_km, dist_river_km, dist_urban_km) %>%
      group_by(lat, lon) %>%
      summarise(across(where(is.numeric), \(x) mean(x, na.rm = TRUE)), .groups = "drop") %>%
      # Calef et al. (2008): suppression effects drop off past 40 km (road) / 30 km (urban)
      mutate(road_proximity_binary  = ifelse(dist_road_km  > 40, 0, 1),
             urban_proximity_binary = ifelse(dist_urban_km > 30, 0, 1),
             lat = round(lat, 4), lon = round(lon, 4))
    
    lc_raw <- read.csv(landcover_source_path)
    nn2 <- get.knnx(data = trace_cells[, c("lon", "lat")],
                    query = lc_raw[, c("lon", "lat")], k = 1)
    lc_raw$trace_lon <- trace_cells$lon[nn2$nn.index]
    lc_raw$trace_lat <- trace_cells$lat[nn2$nn.index]
    
    lc <- lc_raw %>%
      dplyr::select(-lat, -lon) %>%
      rename(lat = trace_lat, lon = trace_lon) %>%
      group_by(lat, lon) %>%
      summarise(across(where(is.numeric), \(x) mean(x, na.rm = TRUE)), .groups = "drop") %>%
      mutate(lat = round(lat, 4), lon = round(lon, 4))
    
    month_data <- month_data %>%
      mutate(lat = round(lat, 4), lon = round(lon, 4)) %>%
      left_join(prox %>% dplyr::select(lat, lon, road_proximity_binary, urban_proximity_binary),
                by = c("lat", "lon")) %>%
      left_join(lc %>% dplyr::select(lat, lon, B, C, U), by = c("lat", "lon"))
    return(month_data)
  }
  
  if (period == "paleo") {
    month_data$road_proximity_binary  <- 0L
    month_data$urban_proximity_binary <- 0L
    # B/C/U already present from trace_monthly_with_lags_and_landcover.parquet;
    # freeze at landcover_freeze_year for any year beyond reconstruction coverage
    lc_freeze <- month_data %>%
      filter(year == landcover_freeze_year) %>%
      dplyr::select(lat, lon, C_freeze = C, B_freeze = B, U_freeze = U) %>%
      distinct(lat, lon, .keep_all = TRUE)
    
    month_data <- month_data %>%
      left_join(lc_freeze, by = c("lat", "lon")) %>%
      mutate(
        C = if_else(year > landcover_freeze_year, C_freeze, C),
        B = if_else(year > landcover_freeze_year, B_freeze, B),
        U = if_else(year > landcover_freeze_year, U_freeze, U)
      ) %>%
      dplyr::select(-C_freeze, -B_freeze, -U_freeze) %>%
      filter(year > min_year)
    return(month_data)
  }
  
  stop("Unknown period in assign_proximity_and_landcover()")
}

## ---- Check 3: cap area lookup + recovery lag, by period --------------------
area_lookup <- read.csv(area_path)
names(area_lookup)[names(area_lookup) == area_col] <- "area"
area_lookup <- area_lookup %>% dplyr::select(lat, lon, area)

cat(sprintf("PERIOD = %s | CLIMATE_SOURCE = %s | CAP_RECOVERY_YEARS = %s | APPLY_EXTREME_TRUNCATION = %s\n",
            PERIOD, if (PERIOD == "calibration") CLIMATE_SOURCE else "n/a",
            as.character(CAP_RECOVERY_YEARS), as.character(APPLY_EXTREME_TRUNCATION)))

# =============================================================================
# 4. LOAD + PREPROCESS DRIVER DATA FOR THE SELECTED PERIOD
#    (produces `summer_df` for lightning rate, `month_df` for everything else)
# =============================================================================

if (PERIOD == "training") {
  summer_df <- read.csv(lightning_rate_path)
  
  month_df <- read.csv(lightning_count_path)
  month_df <- na.omit(month_df)
  names(month_df)[names(month_df) == "strikes"]      <- "r_strike_obs"
  names(month_df)[names(month_df) == "strike_count"] <- "n_strikes_obs"
  
  fire_df <- read.csv(fire_path)
  
  # NIFC-corrected observed fire count / burned area replace the driver
  # file's originals (training-period-only step -- calibration/paleo have
  # no observed fire record to substitute)
  ignition_new <- read.csv(ignition_new_path)[, 2:26]
  ignition_new <- ignition_new[, c("lat", "lon", "year", "month", "fire_count", "strike_count")]
  ignition_new <- ignition_new %>% filter(!(strike_count > 0 & fire_count > strike_count))
  
  fire_df <- fire_df %>%
    dplyr::select(-fire_count) %>%
    left_join(ignition_new %>% dplyr::select(lat, lon, year, month, fire_count),
              by = c("lat", "lon", "year", "month")) %>%
    mutate(fire_count = ifelse(is.na(fire_count), 0, fire_count))
  
  ba_new <- read.csv(ba_new_path)
  fire_df <- fire_df %>%
    dplyr::select(-burned_area_km2, -lightning_ignition, -grid_id,
                  -single_lightning_ignition, -total_ignition_count, -burned_area_km2_total) %>%
    left_join(ba_new %>% dplyr::select(lat, lon, year, month, burned_area_km2_total),
              by = c("lat", "lon", "year", "month")) %>%
    mutate(burned_area_km2_total = ifelse(is.na(burned_area_km2_total), 0, burned_area_km2_total))
  
  fire_df <- fire_df %>%
    mutate(lat = round(lat, 2), lon = round(lon, 2)) %>%
    distinct(lat, lon, year, month, .keep_all = TRUE)
  
  month_df <- assign_proximity_and_landcover("training", NULL, month_df)
  fire_df  <- assign_proximity_and_landcover("training", NULL, fire_df)
  
  ## ---- DIAGNOSTIC: snap training climate predictors to the coarse TraCE
  ## grid, matching how calibration-period ERA5 was processed, to test
  ## whether the predicted training-vs-calibration mismatch found earlier
  ## is a spatial-resolution artifact rather than a genuine period effect.
  ##
  ## Only the climate PREDICTOR columns are snapped via mean-aggregation
  ## (correct for covariates like temperature/precipitation). Observed
  ## outcome totals (fire_count, burned_area_km2_total, r_strike_obs,
  ## n_strikes_obs) are dropped before snapping -- mean-aggregating counts
  ## across ~192 collapsed ERA5 cells per TraCE cell would silently
  ## corrupt their totals, and they are not needed here: this diagnostic
  ## only compares PREDICTED totals against the unchanged, native-
  ## resolution observed totals already established separately.
  trace_cells <- readRDS(trace_cells_path)
  
  summer_df <- summer_df %>% dplyr::select(-any_of("r_strike_obs"))
  summer_df <- aggregate_to_trace_grid(summer_df, trace_cells, c("lat", "lon", "year"))
  summer_df <- summer_df %>% mutate(lat = round(lat, 4), lon = round(lon, 4))
  
  month_df <- month_df %>% dplyr::select(-any_of(c("r_strike_obs", "n_strikes_obs")))
  month_df <- aggregate_to_trace_grid(month_df, trace_cells, c("lat", "lon", "year", "month"))
  month_df <- month_df %>% mutate(lat = round(lat, 4), lon = round(lon, 4))
  
  fire_df <- fire_df %>% dplyr::select(-any_of(c("fire_count", "burned_area_km2_total")))
  fire_df <- aggregate_to_trace_grid(fire_df, trace_cells, c("lat", "lon", "year", "month"))
  fire_df <- fire_df %>% mutate(lat = round(lat, 4), lon = round(lon, 4))
  
  # Re-apply proximity/landcover against the NEW coarse-grid coordinates
  # (must be re-run: the earlier call above joined against the original
  # fine-grid coordinates, which no longer match after snapping)
  # landcover_source_path is normally only defined under PERIOD == "calibration"
  # in this script's config block, which never runs here (PERIOD == "training")
  # -- define it directly, since assign_proximity_and_landcover()'s
  # "calibration" branch (used below to re-snap proximity/landcover onto the
  # new coarse-grid coordinates) requires it.
  landcover_source_path <- file.path(drivers_root, "lightning_fires_burned_area_jja_2002_2011.csv")
  
  # NOTE: unlike the ORIGINAL calibration branch (where fire_df <- month_df
  # is correct, since both come from the same source file), month_df and
  # fire_df here come from DIFFERENT training-period source files with
  # different predictor columns -- fire_df alone carries precip_1m,
  # precip_3m, tair_3m, etc. needed by predict_ignition() and the
  # burned-area sub-models. Each must be re-snapped for proximity/landcover
  # INDEPENDENTLY; overwriting fire_df with month_df would silently drop
  # fire_df's own required columns.
  # fire_df (and possibly month_df) may already carry B/C/U and/or
  # road_proximity_binary/urban_proximity_binary natively from the training
  # driver files (preserved through the coarse-grid mean-aggregation above).
  # The "calibration" branch below does its own left_join() bringing in a
  # SECOND copy of these columns from landcover_source_path/proximity_path --
  # left_join() silently renames colliding columns to .x/.y suffixes rather
  # than erroring, so without this drop, "U" would become "U.x"/"U.y" and
  # silently vanish under its expected name. Drop any pre-existing copies
  # first so the join cleanly (re)adds them fresh, using the coarse-grid
  # coordinates, with no naming collision.
  drop_existing_lc_prox <- c("B", "C", "U", "road_proximity_binary", "urban_proximity_binary")
  month_df <- month_df %>% dplyr::select(-any_of(drop_existing_lc_prox))
  fire_df  <- fire_df  %>% dplyr::select(-any_of(drop_existing_lc_prox))
  
  month_df <- assign_proximity_and_landcover("calibration", "era5", month_df, trace_cells)
  fire_df  <- assign_proximity_and_landcover("calibration", "era5", fire_df, trace_cells)
  
  # Rebuild area_lookup using the TraCE-resolution area file, matching
  # what calibration-period ERA5 predictions used -- overwrites the
  # native-ERA5-grid area_lookup already constructed earlier in this
  # script (built from area_path/area_col as originally set above).
  area_path <- file.path(paleoclimate_data_root, "TraCE_21ka_gridcell_area.csv")
  area_col  <- "area_clipped_km2"
  area_lookup <- read.csv(area_path)
  names(area_lookup)[names(area_lookup) == area_col] <- "area"
  area_lookup <- area_lookup %>%
    dplyr::select(lat, lon, area) %>%
    mutate(lat = round(lat, 4), lon = round(lon, 4))
}

if (PERIOD == "calibration") {
  trace_cells <- readRDS(trace_cells_path)
  
  if (CLIMATE_SOURCE == "era5") {
    summer_df <- read_parquet(climate_summer_path) %>% filter(year >= 1949)
    summer_df <- aggregate_to_trace_grid(summer_df, trace_cells, c("lat", "lon", "year"))
    
    month_df <- read_parquet(climate_month_path) %>% filter(month >= 6 & year >= 1949)
    month_df <- aggregate_to_trace_grid(month_df, trace_cells, c("lat", "lon", "year", "month"))
  } else {
    summer_df <- read_parquet(climate_summer_path) %>% filter(year >= 1949)
    month_df  <- read_parquet(climate_month_path) %>%
      dplyr::select(-B, -C, -U, -midpoint) %>%   # these contain NAs on the raw file; re-joined below
      filter(month >= 6 & year >= 1949)
  }
  
  fire_df <- month_df   # calibration has no separate "fire" driver file -- same monthly df carries through
  
  month_df <- assign_proximity_and_landcover("calibration", CLIMATE_SOURCE, month_df, trace_cells)
  fire_df  <- month_df
}

if (PERIOD == "paleo") {
  summer_df <- read_parquet(paleo_summer_path) %>% filter(year > min_year)
  
  month_df <- read_parquet(paleo_month_path) %>% filter(year > min_year)
  month_df <- assign_proximity_and_landcover("paleo", NULL, month_df)
  fire_df  <- month_df
}

cat(sprintf("Driver data loaded: %d summer rows, %d monthly rows\n", nrow(summer_df), nrow(month_df)))

# =============================================================================
# 5. PER-DRAW PREDICTION FUNCTIONS
#    Each takes one posterior draw's coefficients and returns predictions
#    for the FULL dataset under that draw.
# =============================================================================

## ---- 5a. Lightning strike rate (Gamma) -------------------------------------
predict_lightning_rate <- function(summer_df, draw_row) {
  scaled <- summer_df %>%
    mutate(
      precip = (precip - get_mu(lightning_mu_sigma, "precip")) / get_sigma(lightning_mu_sigma, "precip"),
      tair   = (tair   - get_mu(lightning_mu_sigma, "tair"))   / get_sigma(lightning_mu_sigma, "tair"),
      swr    = (swr    - get_mu(lightning_mu_sigma, "swr"))    / get_sigma(lightning_mu_sigma, "swr"),
      sp     = (sp     - get_mu(lightning_mu_sigma, "sp"))     / get_sigma(lightning_mu_sigma, "sp"),
      rh     = (rh     - get_mu(lightning_mu_sigma, "rh"))     / get_sigma(lightning_mu_sigma, "rh")
    )
  
  lp_alpha <- draw_row$a_alpha + draw_row$b_alpha * scaled$swr + draw_row$c_alpha * scaled$tair +
    draw_row$d_alpha * scaled$rh + draw_row$e_alpha * scaled$precip + draw_row$f_alpha * scaled$sp
  lp_beta  <- draw_row$a_beta  + draw_row$b_beta  * scaled$swr + draw_row$c_beta  * scaled$tair +
    draw_row$d_beta  * scaled$rh + draw_row$e_beta  * scaled$precip + draw_row$f_beta  * scaled$sp
  
  alpha_pred <- exp(lp_alpha)
  beta_pred  <- exp(lp_beta)
  
  scaled %>%
    mutate(r_strike = rgamma(n(), shape = alpha_pred, rate = beta_pred)) %>%
    dplyr::select(lat, lon, year, r_strike)
}

## ---- Helper: replace GPD-threshold exceedances with fresh GPD draws -------
## Used by predict_strike_count() below. Requires the `evd` package (rgpd).
apply_gpd_tail_correction <- function(raw_draws, u, sigma, xi) {
  if (is.infinite(u)) return(raw_draws)   # correction disabled
  corrected <- raw_draws
  exceed_idx <- which(raw_draws > u)
  if (length(exceed_idx) > 0) {
    corrected[exceed_idx] <- as.integer(round(
      u + rgpd(length(exceed_idx), loc = 0, scale = sigma, shape = xi)
    ))
  }
  corrected
}

## ---- 5b. Strike count (NegBinomial) ----------------------------------------
predict_strike_count <- function(month_df, r_strike_df, area_lookup, nb_coefs) {
  data <- month_df %>%
    left_join(r_strike_df, by = c("lat", "lon", "year")) %>%
    left_join(area_lookup, by = c("lat", "lon"))
  
  scaled <- data %>%
    mutate(
      precip = (precip - get_mu(n_strike_mu_sigma, "precip")) / get_sigma(n_strike_mu_sigma, "precip"),
      tair   = (tair   - get_mu(n_strike_mu_sigma, "tair"))   / get_sigma(n_strike_mu_sigma, "tair"),
      wind   = (wind   - get_mu(n_strike_mu_sigma, "wind"))   / get_sigma(n_strike_mu_sigma, "wind"),
      rh     = (rh     - get_mu(n_strike_mu_sigma, "rh"))     / get_sigma(n_strike_mu_sigma, "rh"),
      swr    = (swr    - get_mu(n_strike_mu_sigma, "swr"))    / get_sigma(n_strike_mu_sigma, "swr"),
      sp     = (sp     - get_mu(n_strike_mu_sigma, "sp"))     / get_sigma(n_strike_mu_sigma, "sp"),
      mu_r_strike_gridcell = r_strike * area
    )
  
  Z <- as.matrix(scaled[, theta_predictors])
  theta_pred <- exp(as.numeric(nb_coefs$gamma_0 + Z %*% nb_coefs$gamma))
  
  scaled %>%
    mutate(
      # RAW draw, kept uncorrected -- useful downstream for showing how the
      # tail correction changes the distribution, but NOT used for fire count
      n_strikes_raw = rnbinom(n(), mu = pmax(mu_r_strike_gridcell, 1e-8), size = theta_pred),
      # Corrected draw -- this is what feeds fire count / everything downstream
      n_strikes = apply_gpd_tail_correction(
        n_strikes_raw, u = STRIKE_GPD_THRESH, sigma = STRIKE_GPD_SIGMA, xi = STRIKE_GPD_XI
      )
    ) %>%
    dplyr::select(lat, lon, year, month, r_strike, n_strikes_raw, n_strikes, area)
}

## ---- 5c. Ignition efficiency (Bayesian Beta-Binomial) ----------------------
predict_ignition <- function(fire_df, draw_idx) {
  scaled <- fire_df %>%
    mutate(
      rh        = (rh        - get_mu(ignition_mu_sigma, "rh"))        / get_sigma(ignition_mu_sigma, "rh"),
      tair      = (tair      - get_mu(ignition_mu_sigma, "tair"))      / get_sigma(ignition_mu_sigma, "tair"),
      precip    = (precip    - get_mu(ignition_mu_sigma, "precip"))    / get_sigma(ignition_mu_sigma, "precip"),
      precip_1m = (precip_1m - get_mu(ignition_mu_sigma, "precip_1m")) / get_sigma(ignition_mu_sigma, "precip_1m"),
      precip_3m = (precip_3m - get_mu(ignition_mu_sigma, "precip_3m")) / get_sigma(ignition_mu_sigma, "precip_3m"),
      precip_5y = (precip_5y - get_mu(ignition_mu_sigma, "precip_5y")) / get_sigma(ignition_mu_sigma, "precip_5y"),
      tair_3m   = (tair_3m   - get_mu(ignition_mu_sigma, "tair_3m"))   / get_sigma(ignition_mu_sigma, "tair_3m"),
      tair_5y   = (tair_5y   - get_mu(ignition_mu_sigma, "tair_5y"))   / get_sigma(ignition_mu_sigma, "tair_5y"),
      U         = (U         - get_mu(ignition_mu_sigma, "U"))        / get_sigma(ignition_mu_sigma, "U")
    )
  
  X <- as.matrix(scaled[, predictors_ignition])
  alpha_draw <- ignition_posterior$alpha[draw_idx]
  beta_draw  <- ignition_posterior$beta[draw_idx, ]
  
  linpred <- as.numeric(alpha_draw + X %*% beta_draw)
  # Single rate-calibration shift only -- no separate binary cal_shift_bayes
  # stage (the Beta-Binomial's own dispersion parameter already absorbs the
  # extra-binomial variance that previously required a two-stage correction)
  p_ignite_rate_cal <- plogis(pmin(pmax(linpred + rate_cal_shift,
                                        log(eps / (1 - eps))), log((1 - eps) / eps)))
  
  scaled %>%
    mutate(p_ignite_rate_cal = p_ignite_rate_cal) %>%
    dplyr::select(lat, lon, year, month, p_ignite_rate_cal)
}

## ---- 5d. Burned area (mixture: classifier + extreme Gamma + typical Gamma) -
# Vapor pressure deficit (VPD), computed from RAW (unstandardized) tair
# (Kelvin) and rh (%), using the Tetens saturation-vapor-pressure formula --
# needed for m20; era5/bias-corrected TraCE drivers never carry VPD natively.
compute_vpd <- function(tair_kelvin, rh_percent) {
  tair_C <- tair_kelvin - 273.15
  es <- 0.6108 * exp((17.27 * tair_C) / (tair_C + 237.3))
  ea <- es * (rh_percent / 100)
  es - ea
}

predict_burned_area <- function(fire_df, draw_idx) {
  scaled <- fire_df %>%
    mutate(vpd = compute_vpd(tair, rh)) %>%   # compute VPD from RAW tair/rh, BEFORE standardizing either
    mutate(
      rh        = (rh        - get_mu(burned_area_mu_sigma, "rh"))        / get_sigma(burned_area_mu_sigma, "rh"),
      tair      = (tair      - get_mu(burned_area_mu_sigma, "tair"))      / get_sigma(burned_area_mu_sigma, "tair"),
      precip    = (precip    - get_mu(burned_area_mu_sigma, "precip"))    / get_sigma(burned_area_mu_sigma, "precip"),
      tair_2m   = (tair_2m   - get_mu(burned_area_mu_sigma, "tair_2m"))   / get_sigma(burned_area_mu_sigma, "tair_2m"),
      tair_5y   = (tair_5y   - get_mu(burned_area_mu_sigma, "tair_5y"))   / get_sigma(burned_area_mu_sigma, "tair_5y"),
      U         = (U         - get_mu(burned_area_mu_sigma, "U"))        / get_sigma(burned_area_mu_sigma, "U"),
      # predictors required by m20, not used by m3:
      wind       = (wind       - get_mu(burned_area_mu_sigma, "wind"))       / get_sigma(burned_area_mu_sigma, "wind"),
      C          = (C          - get_mu(burned_area_mu_sigma, "C"))          / get_sigma(burned_area_mu_sigma, "C"),
      precip_5y  = (precip_5y  - get_mu(burned_area_mu_sigma, "precip_5y"))  / get_sigma(burned_area_mu_sigma, "precip_5y"),
      precip_3m  = (precip_3m  - get_mu(burned_area_mu_sigma, "precip_3m")) / get_sigma(burned_area_mu_sigma, "precip_3m"),
      vpd        = (vpd        - get_mu(burned_area_mu_sigma, "vpd"))        / get_sigma(burned_area_mu_sigma, "vpd")
    )
  
  # model.matrix() against the stored FORMULAS -- necessary (rather than
  # plain column subsetting) so interaction terms (rh:urban_proximity_binary,
  # rh:U, rh:vpd, wind:rh) get correctly expanded into their own columns.
  X_classifier <- model.matrix(formula_classifier, data = scaled)
  X_extreme    <- model.matrix(formula_extreme,    data = scaled)
  X_typical    <- model.matrix(formula_typical,    data = scaled)
  
  a_logit <- logit_posterior$alpha[draw_idx];   b_logit <- logit_posterior$beta[draw_idx, ]
  a_ext   <- extreme_posterior$alpha[draw_idx]; b_ext   <- extreme_posterior$beta[draw_idx, ]; phi_ext <- extreme_posterior$phi[draw_idx]
  a_typ   <- normal_posterior$alpha[draw_idx];  b_typ   <- normal_posterior$beta[draw_idx, ];  phi_typ <- normal_posterior$phi[draw_idx]
  
  p_extreme  <- plogis(as.numeric(a_logit + X_classifier %*% b_logit))
  mu_extreme <- as.numeric(exp(a_ext + X_extreme %*% b_ext))
  mu_typical <- as.numeric(exp(a_typ + X_typical %*% b_typ))
  
  scaled %>%
    mutate(
      p_extreme    = p_extreme,   # raw, uncalibrated classifier probability -- no shift applied
      mu_extreme   = mu_extreme,
      mu_typical   = mu_typical,
      rate_extreme = phi_ext / mu_extreme,
      rate_typical = phi_typ / mu_typical,
      phi_extreme  = phi_ext,
      phi_typical  = phi_typ
    ) %>%
    dplyr::select(lat, lon, year, month, p_extreme, rate_extreme, rate_typical, phi_extreme, phi_typical)
}

## ---- 5e. Sample total burned area given fire_count -------------------------
# Extreme-component sampling is TRUNCATED at burned_area_threshold, via
# rejection sampling (draw ordinary rgamma(), keep only values above
# threshold, repeat) -- numerically stable regardless of how far into the
# tail the threshold sits, unlike inverse-CDF truncation.
rgamma_truncated <- function(n, shape, rate, lower, max_attempts = 10000) {
  out <- numeric(n)
  needed <- rep(TRUE, n)
  attempts <- 0
  while (any(needed) && attempts < max_attempts) {
    candidates <- rgamma(sum(needed), shape = shape, rate = rate)
    accepted <- candidates > lower
    out[which(needed)[accepted]] <- candidates[accepted]
    needed[which(needed)[accepted]] <- FALSE
    attempts <- attempts + 1
  }
  if (any(needed)) out[needed] <- lower  # bounded fallback, never an outlier
  out
}

sample_total_burned_area <- function(fire_count, p_extreme, rate_extreme, rate_typical,
                                     phi_extreme, phi_typical, threshold = burned_area_threshold,
                                     apply_truncation = APPLY_EXTREME_TRUNCATION) {
  n_rows <- length(fire_count)
  valid_idx <- which(!is.na(fire_count) & fire_count > 0)
  if (length(valid_idx) == 0) return(rep(0, n_rows))
  
  rep_idx      <- rep(valid_idx, times = fire_count[valid_idx])
  rep_p_ext    <- rep(p_extreme[valid_idx], times = fire_count[valid_idx])
  rep_rate_ext <- rep(rate_extreme[valid_idx], times = fire_count[valid_idx])
  rep_rate_typ <- rep(rate_typical[valid_idx], times = fire_count[valid_idx])
  
  is_extreme <- rbinom(length(rep_idx), size = 1, prob = rep_p_ext)
  
  n_ext <- sum(is_extreme == 1)
  ext_samples <- if (n_ext > 0) {
    if (apply_truncation) {
      rgamma_truncated(n_ext, shape = phi_extreme[1], rate = rep_rate_ext[is_extreme == 1], lower = threshold)
    } else {
      rgamma(n_ext, shape = phi_extreme[1], rate = rep_rate_ext[is_extreme == 1])
    }
  } else {
    numeric(0)
  }
  
  samples <- numeric(length(rep_idx))
  samples[is_extreme == 1] <- ext_samples
  samples[is_extreme == 0] <- rgamma(sum(is_extreme == 0), shape = phi_typical[1], rate = rep_rate_typ[is_extreme == 0])
  
  row_sums <- tapply(samples, rep_idx, sum)
  out <- rep(0, n_rows)
  out[as.integer(names(row_sums))] <- row_sums
  out
}

## ---- 5f. Gridcell-area cap (unchanged core logic from original scripts) ---
apply_area_cap <- function(df, raw_col = "total_burned_area", recovery_years = Inf) {
  n <- nrow(df)
  area <- df$area[1]
  capped <- numeric(n)
  cumulative <- 0
  recovery_until_year <- -Inf
  
  for (i in seq_len(n)) {
    this_year <- df$year[i]
    raw_val <- df[[raw_col]][i]
    
    if (this_year < recovery_until_year || (area - cumulative) <= 0) {
      capped[i] <- 0
      next
    }
    
    applied_val <- min(raw_val, area - cumulative)
    capped[i] <- applied_val
    cumulative <- cumulative + applied_val
    
    if (cumulative >= area) {
      cumulative <- 0
      recovery_until_year <- this_year + recovery_years
    }
  }
  df$total_burned_area_capped <- capped
  df
}

apply_cap_to_dataset <- function(data, area_lookup, recovery_years = Inf) {
  data <- data %>% mutate(lat = round(lat, 4), lon = round(lon, 4))
  area_lookup <- area_lookup %>%
    mutate(lat = round(lat, 4), lon = round(lon, 4)) %>%
    distinct(lat, lon, .keep_all = TRUE)
  
  data <- data %>% dplyr::select(-any_of("area")) %>% left_join(area_lookup, by = c("lat", "lon"))
  
  data %>%
    filter(!is.na(area)) %>%
    arrange(lat, lon, year, month) %>%
    group_by(lat, lon) %>%
    group_modify(~ apply_area_cap(.x, recovery_years = recovery_years)) %>%
    ungroup()
}

# =============================================================================
# 6. MAIN LOOP — N_DRAWS full-pipeline posterior replicates
# =============================================================================

cat(sprintf("Starting %d posterior replicates...\n", N_DRAWS))
t0 <- Sys.time()

# Write incrementally (append each draw to disk as it finishes) rather than
# holding all N_DRAWS results in memory -- for large datasets (paleo in
# particular), accumulating everything until the end can exhaust available
# RAM and cause severe slowdown from swapping, with no visible progress.
# This also lets you check real progress at any time with, from a
# terminal: wc -l <OUTPUT_FILE>
if (file.exists(OUTPUT_FILE)) file.remove(OUTPUT_FILE)
first_write <- TRUE

for (d in seq_len(N_DRAWS)) {
  
  t_draw_start <- Sys.time()
  
  # -- draw coefficients for this replicate --
  lightning_draw <- lightning_posterior[sample(n_lightning_draws, 1), ]
  nb_coefs       <- sample_nb_coefs()
  ignition_idx   <- sample(n_ignition_draws, 1)
  mixture_idx    <- sample(n_mixture_draws, 1)
  
  # -- 1. lightning rate --
  r_strike_df <- predict_lightning_rate(summer_df, lightning_draw)
  t1 <- Sys.time()
  
  # -- 2. strike count --
  strike_df <- predict_strike_count(month_df, r_strike_df, area_lookup, nb_coefs)
  t2 <- Sys.time()
  
  # -- 3. ignition efficiency + fire count --
  ignition_df <- predict_ignition(fire_df, ignition_idx)
  
  fire_rep <- fire_df %>%
    left_join(strike_df %>% dplyr::select(lat, lon, year, month, r_strike, n_strikes_raw, n_strikes),
              by = c("lat", "lon", "year", "month")) %>%
    left_join(ignition_df, by = c("lat", "lon", "year", "month")) %>%
    mutate(fire_count_pred = rbinom(n(), size = n_strikes, prob = p_ignite_rate_cal))
  t3 <- Sys.time()
  
  # -- 4. burned area --
  ba_df <- predict_burned_area(fire_df, mixture_idx)
  
  fire_rep <- fire_rep %>%
    left_join(ba_df, by = c("lat", "lon", "year", "month")) %>%
    mutate(total_burned_area = sample_total_burned_area(
      fire_count_pred, p_extreme, rate_extreme, rate_typical, phi_extreme, phi_typical
    ))
  t4 <- Sys.time()
  
  # -- 5. gridcell-area cap --
  fire_rep <- apply_cap_to_dataset(fire_rep, area_lookup, recovery_years = CAP_RECOVERY_YEARS)
  t5 <- Sys.time()
  
  # Per-stage timing, printed for the first 3 draws so a bottleneck shows up
  # immediately rather than after a long blind wait. Remove/comment out once
  # you've confirmed per-draw runtime is reasonable.
  if (d <= 3) {
    cat(sprintf(
      "  [draw %d timing] lightning: %.1fs | strike_count: %.1fs | ignition+fire: %.1fs | burned_area: %.1fs | cap: %.1fs | TOTAL: %.1fs\n",
      d,
      as.numeric(difftime(t1, t_draw_start, units = "secs")),
      as.numeric(difftime(t2, t1, units = "secs")),
      as.numeric(difftime(t3, t2, units = "secs")),
      as.numeric(difftime(t4, t3, units = "secs")),
      as.numeric(difftime(t5, t4, units = "secs")),
      as.numeric(difftime(t5, t_draw_start, units = "secs"))
    ))
    flush(stdout())
  }
  
  draw_result <- fire_rep %>%
    mutate(draw_id = d) %>%
    dplyr::select(draw_id, lat, lon, year, month, r_strike, n_strikes_raw, n_strikes,
                  p_ignite_rate_cal, fire_count_pred, total_burned_area,
                  total_burned_area_capped)
  
  write.table(draw_result, OUTPUT_FILE, sep = ",", row.names = FALSE,
              col.names = first_write, append = !first_write)
  first_write <- FALSE
  
  elapsed <- round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1)
  cat(sprintf("  draw %d / %d done (%.1f min elapsed)\n", d, N_DRAWS, elapsed))
  flush(stdout())
}

cat(sprintf("\nDone. Predictions written incrementally to %s\n", OUTPUT_FILE))