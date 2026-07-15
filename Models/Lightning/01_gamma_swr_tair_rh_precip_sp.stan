data {
  int<lower=0> N;
  vector[N] S;
  vector[N] SWR;
  vector[N] T;
  vector[N] RH;
  vector[N] P;
  vector[N] SP;
}
parameters {
  real a_alpha;   // Intercept for alpha
  real b_alpha;   // Coefficient for SWR in alpha
  real c_alpha;   // Coefficient for T in alpha
  real d_alpha;   // Coefficient for RH in alpha
  real e_alpha;   // Coefficient for P in alpha
  real f_alpha;   // Coefficient for SP in alpha
  real a_beta;    // Intercept for beta
  real b_beta;    // Coefficient for SWR in beta
  real c_beta;    // Coefficient for T in beta
  real d_beta;    // Coefficient for RH in beta
  real e_beta;    // Coefficient for P in beta
  real f_beta;    // Coefficient for SP in beta
}
model {
  vector[N] alpha;
  vector[N] beta;
  alpha = exp(a_alpha + b_alpha*SWR + c_alpha*T + d_alpha*RH + e_alpha*P + f_alpha*SP);
  beta  = exp(a_beta  + b_beta*SWR  + c_beta*T  + d_beta*RH  + e_beta*P  + f_beta*SP);
  S ~ gamma(alpha, beta);
  // Priors
  a_alpha ~ normal(0, 1);
  b_alpha ~ normal(0, 1);   
  c_alpha ~ normal(0, 1);
  d_alpha ~ normal(0, 1);
  e_alpha ~ normal(0, 1);
  f_alpha ~ normal(0, 1);
  a_beta  ~ normal(0, 1);
  b_beta  ~ normal(0, 1);
  c_beta  ~ normal(0, 1);
  d_beta  ~ normal(0, 1);
  e_beta  ~ normal(0, 1);
  f_beta  ~ normal(0, 1);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    real alpha_n = exp(a_alpha + b_alpha*SWR[n] + c_alpha*T[n] + d_alpha*RH[n] + e_alpha*P[n] + f_alpha*SP[n]);
    real beta_n  = exp(a_beta  + b_beta*SWR[n]  + c_beta*T[n]  + d_beta*RH[n]  + e_beta*P[n]  + f_beta*SP[n]);
    log_lik[n] = gamma_lpdf(S[n] | alpha_n, beta_n);
  }
}