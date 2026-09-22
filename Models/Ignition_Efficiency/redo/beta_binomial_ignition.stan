// beta_binomial_ignition.stan
data {
  int<lower=0> N;                          // number of gridcell-months
  int<lower=0> K;                          // number of predictors
  matrix[N, K] X;                          // standardized predictor matrix
  int<lower=0> fire_count[N];              // successes
  int<lower=0> strike_count[N];            // trials
}
parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> phi;                       // dispersion (concentration), analogous to glmmTMB's betabinomial dispersion
}
model {
  vector[N] p = inv_logit(alpha + X * beta);

  // Priors -- weakly informative, adjust if divergences/poor mixing occur
  alpha ~ normal(0, 5);
  beta  ~ normal(0, 2);
  phi   ~ exponential(0.02);               // mean 50; glmmTMB estimated ~28.7 as a reference point

  for (n in 1:N)
    fire_count[n] ~ beta_binomial(strike_count[n], p[n] * phi, (1 - p[n]) * phi);
}
generated quantities {
  vector[N] p_ignite = inv_logit(alpha + X * beta);
}