data {
  int<lower=0> N;
  int<lower=0, upper=N> N_obs;
  int obs_idx[N_obs];

  int<lower=0> N_edges;
  int<lower=1, upper=N> node1[N_edges]; // node1[i] adjacent to node2[i]
  int<lower=1, upper=N> node2[N_edges]; // and node1[i] < node2[i]
    
  int<lower=0> y[N_obs]; // count outcomes (observed ones)
  vector<lower=0>[N] E; // exposure
  vector[N] x; // predictor

  real scaling_factor; // bym2 scaling factor
}

transformed data {
  vector[N] log_E = log(E);
}

parameters {
  real beta0; // intercept
  real beta1; // slopes of predictors
  
  real<lower=0> sigma; // overall std
  real<lower=0, upper=1> rho; // proportion of unstructured vs. spatially structured variance
  
  vector[N] theta; // heterogeneous effects
  vector[N] phi; // spatial effects
}

transformed parameters {
  vector[N] convolved_re = sqrt(1 - rho) * theta + sqrt(rho / scaling_factor) * phi;
}

model {
    y ~ poisson_log(log_E[obs_idx] + beta0 + x[obs_idx] * beta1 + convolved_re[obs_idx] * sigma);
    
    // icar prior on phi
    target += -0.5 * dot_self(phi[node1] - phi[node2]);
    // soft sum-to-zero constraint on phi
    sum(phi) ~ normal(0, 0.001 * N); 

    beta0 ~ normal(0., 5.);
    beta1 ~ normal(0., 5.);
    theta ~ normal(0., 1.);
    sigma ~ normal(0., 1.);
    rho ~ normal(0., 5.);

  /* } */
  
}

generated quantities {
  vector[N] mu;
  /* if (kriging) { */
  /*   mu = exp(log_E + beta0 + beta1 * x + f + theta * sigma_theta); */
  /* } else { */
  real logit_rho = log(rho / (1. - rho));
  vector[N] eta = log_E + beta0 + x * beta1 + convolved_re * sigma;
  mu = exp(eta);

  /* } */
}
