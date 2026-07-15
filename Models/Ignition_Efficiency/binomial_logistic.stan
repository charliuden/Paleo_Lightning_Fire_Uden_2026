// Bayesian binomial logistic regression for ignition efficiency
// Adapted from Uden et al. (2026) per-strike Bernoulli model to
// gridcell-month binomial aggregation.
//
// Model:
//   n_success_i ~ Binomial(n_trials_i, p_i)
//   logit(p_i)  = beta0 + X_i * beta
//   beta0       ~ Normal(0, sigma_beta0^2)
//   beta        ~ Normal(0, sigma_beta^2)

data {
  int<lower=0>            N;          // number of gridcell-months
  int<lower=0>            K;          // number of predictors
  matrix[N, K]            X;          // standardised predictor matrix
  int<lower=0>   n_trials[N];   // strike_count per gridcell-month
  int<lower=0>   n_success[N];  // fire_count per gridcell-month
  real<lower=0>           sigma_beta0; // prior sd for intercept
  real<lower=0>           sigma_beta;  // prior sd for coefficients
}

parameters {
  real        beta0;   // intercept
  vector[K]   beta;    // regression coefficients
}

model {
  // Priors
  beta0 ~ normal(0, sigma_beta0);
  beta  ~ normal(0, sigma_beta);

  // Likelihood
  n_success ~ binomial_logit(n_trials, beta0 + X * beta);
}

generated quantities {
  // Posterior predicted ignition probability for each gridcell-month
  vector[N] p_ignite;
  // Log-likelihood for LOO-CV
  vector[N] log_lik;

  for (i in 1:N) {
    real eta    = beta0 + dot_product(X[i], beta);
    p_ignite[i] = inv_logit(eta);
    log_lik[i]  = binomial_logit_lpmf(n_success[i] | n_trials[i], eta);
  }
}