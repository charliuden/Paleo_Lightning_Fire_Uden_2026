# Run this once (or whenever a model is refit) to export posterior-mean
# coefficients from the .rds model objects into flat JSON files that
# run_monte_carlo.py reads. Avoids needing rpy2/pystan in Python.

library(jsonlite)

read_properties <- function(file_path) {
  lines <- readLines(file_path)
  lines <- lines[grepl("=", lines)]  # only keep lines with key-value pairs
  key_vals <- strsplit(lines, "=")
  props <- setNames(
    trimws(sapply(key_vals, `[`, 2)),
    trimws(sapply(key_vals, `[`, 1))
  )
  return(props)
}

# *** paleo_config.properties must be in Paleo_Lightning_Fire_Uden_2026
#location of this script:
current_path <- dirname(rstudioapi::getActiveDocumentContext()$path)
#get path to this project
base_path <- sub("(.*Paleo_Lightning_Fire_Uden_2026).*", "\\1", current_path)
#open 
config_file_path <- file.path(base_path, "paleo_config.properties")


out_dir <- file.path(config[["rds_root"]], "exported_params")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- lightning (B12) ----
B12 <- readRDS(file.path(config[["rds_root"]], "Lightning/B12_gamma_bayes.rds"))
post <- as.data.frame(B12)
lightning_coefs <- as.list(sapply(
  c("a_alpha","b_alpha","c_alpha","d_alpha","e_alpha","f_alpha",
    "a_beta","b_beta","c_beta","d_beta","e_beta","f_beta"),
  function(v) mean(post[[v]])
))
write_json(lightning_coefs, file.path(out_dir, "lightning_coefs.json"), auto_unbox = TRUE)

# ---- strike count (negative binomial theta model) ----
nb_model <- readRDS(file.path(config[["rds_root"]], "Lightning/neg_bin_strike_count_small.rds"))
theta_predictors <- c("swr", "tair", "rh", "precip", "sp", "wind")
gamma_vec <- nb_model$coef_means$gamma
if (is.null(names(gamma_vec))) {
  # gamma came back unnamed -- must match theta_predictors order exactly.
  # CHECK THIS against how the model matrix Z_fire was built in the Rmd
  # (predictors <- c("swr","tair","rh","precip","sp","wind")) before trusting it.
  names(gamma_vec) <- theta_predictors
}
strike_count_nb <- list(
  gamma_0 = nb_model$coef_means$gamma_0,
  gamma   = as.list(gamma_vec)  # now a NAMED list -> JSON object, not array
)
write_json(strike_count_nb, file.path(out_dir, "strike_count_nb.json"), auto_unbox = TRUE)

# ---- ignition efficiency ----
ignition_obj <- readRDS(file.path(config[["rds_root"]], "Ignition_Efficiency/m2_bayes_calibrated.rds"))
rate_cal <- readRDS(file.path(config[["rds_root"]], "Ignition_Efficiency/calibration_shifts.rds"))
predictors_m2 <- c("rh", "tair", "precip", "precip_1m", "precip_2m", "precip_3m",
                    "precip_5y", "tair_1m", "tair_2m", "tair_3m", "tair_5y", "B")
ignition_beta <- ignition_obj$coef_means$beta
if (is.null(names(ignition_beta))) {
  # CHECK against predictors_m2 in the Rmd's X_new model.matrix() call before trusting.
  names(ignition_beta) <- predictors_m2
}
ignition_coefs <- list(
  alpha = ignition_obj$coef_means$alpha,
  beta  = as.list(ignition_beta),
  cal_shift_bayes = ignition_obj$cal_shift,
  rate_cal_shift  = rate_cal$rate_cal_shift
)
write_json(ignition_coefs, file.path(out_dir, "ignition_coefs.json"), auto_unbox = TRUE)

# ---- burned area ----
m6_obj <- readRDS(file.path(config[["rds_root"]], "Burned_Area/m6_bayes_gamma_re.rds"))
post_ba <- rstan::extract(m6_obj$fit)
preds_final <- c("rh", "tair", "precip", "wind", "precip_1m", "precip_3m", "precip_5y",
                  "tair_2m", "tair_5y", "road_proximity_binary", "urban_proximity_binary")
burned_area_beta <- m6_obj$coef_means$beta
if (is.null(names(burned_area_beta))) {
  # CHECK against preds_final in the Rmd's X_new model.matrix() call before trusting.
  names(burned_area_beta) <- preds_final
}
burned_area_coefs <- list(
  alpha = m6_obj$coef_means$alpha,
  beta  = as.list(burned_area_beta),
  phi_mean = mean(post_ba$phi)
)
write_json(burned_area_coefs, file.path(out_dir, "burned_area_coefs.json"), auto_unbox = TRUE)

cat("Exported parameter files to:", out_dir, "\n")

# OPTIONAL: if you later want PARAMETER uncertainty (not just process
# uncertainty), also export the full posterior draws instead of just means,
# e.g. write_json(post[, c("a_alpha","b_alpha",...)], "lightning_posterior_draws.json")
# and sample a row index per Monte Carlo run in Python instead of using the
# fixed mean. Flagging this here so the two scripts stay in sync if you
# extend to that later.
