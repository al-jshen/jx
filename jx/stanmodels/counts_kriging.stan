data {
  int<lower=0> N;
  int<lower=0, upper=N> N_obs;
  int obs_idx[N_obs];

  int<lower=0> N_edges;
  int<lower=1, upper=N> node1[N_edges]; // node1[i] adjacent to node2[i]
  int<lower=1, upper=N> node2[N_edges]; // and node1[i] < node2[i]
  
  real X1[N]; 
  real X2[N];
    
  int<lower=0> y[N_obs]; // count outcomes (observed ones)
  vector<lower=0>[N] E; // exposure
  vector[N] x; // predictor

}

transformed data {
  vector[N] log_E = log(E);
  real delta = 1e-9;
    
  row_vector[2] X[N];
  for (n in 1:N) {
    X[n, 1] = X1[n];
    X[n, 2] = X2[n];
  }
}

parameters {
  real beta0; // intercept
  real beta1; // slopes of predictors
  
  real<lower=0> sigma_theta; // standard deviation of theta
  vector[N] theta; // heterogeneous effects
    
  real<lower=0> rho;
  real<lower=0> alpha;
  vector[N] eta;
}

transformed parameters {
    
  vector[N] f;
  {
    matrix[N, N] L_K;
    matrix[N, N] K = cov_exp_quad(X, alpha, rho);

    // diagonal elements
    for (n in 1:N)
      K[n, n] = K[n, n] + delta;

    L_K = cholesky_decompose(K);
    f = L_K * eta;
  }
    
}

model {
    
  y ~ poisson_log(log_E[obs_idx] + beta0 + beta1 * x[obs_idx] + f[obs_idx] + theta[obs_idx] * sigma_theta);

  rho ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  eta ~ std_normal();

  beta0 ~ normal(0., 1.);
  beta1 ~ normal(0., 1.);
  theta ~ normal(0., 1.);
  sigma_theta ~ exponential(1.);

}

generated quantities {
  vector[N] mu = exp(log_E + beta0 + beta1 * x + f + theta * sigma_theta);
}
