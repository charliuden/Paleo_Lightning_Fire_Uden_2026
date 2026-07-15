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
  a_alpha ~ normal(0, 1);
  b_alpha ~ normal(0, 1);
  c_alpha ~ normal(0, 1);
  d_alpha ~ normal(0, 1);
  e_alpha ~ normal(0, 1);
  f_alpha ~ normal(0, 1);
  g_alpha ~ normal(0, 1);
  a_beta ~ normal(0, 1);
  b_beta ~ normal(0, 1);
  c_beta ~ normal(0, 1);
  d_beta ~ normal(0, 1);
  e_beta ~ normal(0, 1);
  f_beta ~ normal(0, 1);
  g_beta ~ normal(0, 1);
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
