//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//
data {
  int<lower=0> N;           // Number of observations
  vector[N] S;              // Lightning strike rates (observations)
  vector[N] SWR;            // Climate variable 1
  vector[N] T;              // Climate variable 2
  vector[N] RH;             // Climate variable 3
  vector[N] W;              // Climate variable 4
  vector[N] P;              // Climate variable 5
  vector[N] SP;             // Climate variable 6
  real mu_a_beta;          // prior mean for a_beta intercept
  real<lower=0> sigma_intercepts;   // prior SD for intercepts
  real<lower=0> sigma_slopes;       // prior SD for slope coefficients
}

parameters {
  real a_alpha;             // Intercept for alpha
  real b_alpha;             // Coefficient for SWR in alpha
  real c_alpha;             // Coefficient for T in alpha
  real d_alpha;             // Coefficient for RH in alpha
  real e_alpha;             // Coefficient for W in alpha
  real f_alpha;             // Coefficient for P in alpha
  real g_alpha;             // Coefficient for SP in alpha
  real a_beta;              // Intercept for beta
  real b_beta;              // Coefficient for SWR in beta
  real c_beta;              // Coefficient for T in beta
  real d_beta;              // Coefficient for RH in beta
  real e_beta;              // Coefficient for W in beta
  real f_beta;              // Coefficient for P in beta
  real g_beta;              // Coefficient for SP in beta
}

model {
  vector[N] alpha;          // Shape parameter for the gamma distribution
  vector[N] beta;           // Rate parameter for the gamma distribution

  // Define alpha and beta as transformed functions of SWR, T, RH, W, and P
  alpha = exp(a_alpha + b_alpha * SWR + c_alpha * T + d_alpha * RH + e_alpha * W + f_alpha * P + g_alpha * SP);
  beta = exp(a_beta + b_beta * SWR + c_beta * T + d_beta * RH + e_beta * W + f_beta * P + g_beta * SP);

  // Likelihood
  S ~ gamma(alpha, beta);

  // Priors
  a_alpha ~ normal(0, sigma_intercepts);
  a_beta  ~ normal(mu_a_beta, sigma_intercepts);
  b_alpha ~ normal(0, sigma_slopes);
  c_alpha ~ normal(0, sigma_slopes);
  d_alpha ~ normal(0, sigma_slopes);
  e_alpha ~ normal(0, sigma_slopes);
  f_alpha ~ normal(0, sigma_slopes);
  g_alpha ~ normal(0, sigma_slopes);
  a_beta ~ normal(0, sigma_slopes);
  b_beta ~ normal(0, sigma_slopes);
  c_beta ~ normal(0, sigma_slopes);
  d_beta ~ normal(0, sigma_slopes);
  e_beta ~ normal(0, sigma_slopes);
  f_beta ~ normal(0, sigma_slopes);
  g_beta ~ normal(0, sigma_slopes);
}

generated quantities {
  vector[N] log_lik;        // Log-likelihood for each observation

  for (n in 1:N) {
    // Recalculate alpha and beta in the generated quantities block
    real alpha_n = exp(a_alpha + b_alpha * SWR[n] + c_alpha * T[n] + d_alpha * RH[n] + e_alpha * W[n] + f_alpha * P[n] + g_alpha * SP[n]);
    real beta_n = exp(a_beta + b_beta * SWR[n] + c_beta * T[n] + d_beta * RH[n] + e_beta * W[n] + f_beta * P[n] + g_beta * SP[n]);
    log_lik[n] = gamma_lpdf(S[n] | alpha_n, beta_n);
  }
}
