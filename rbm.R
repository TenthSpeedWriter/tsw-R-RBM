library(pracma)

# Assumes x is a vector of numeric probabilities in range(0, 1)
sbinsamp <- stochastic_binary_sample <- function(x) {
  random_probability_rolls <- runif(n = length(x),
                                    min = 0,
                                    max = 1)
  
  as.numeric(random_probability_rolls <= x)
}

RBM <- function(visible_size,
                hidden_size,
                initial_learn_rate = 1e-3) {
  new_weight <- function(in_size,
                         out_size,
                         mean = 0.0,
                         sd = 0.1) {
    flattened_size <- in_size * out_size
    initial_weights <- rnorm(flattened_size,
                             mean = mean,
                             sd = sd)
    
    matrix(initial_weights,
           ncol = in_size,
           nrow = out_size)
  }
  
  new_bias <- function(size,
                       initial = 0.1) {
    rep(initial, times = size)
  }
  
  hidden_bias <- new_bias(hidden_size)
  visible_bias <- new_bias(visible_size)
  weight <- new_weight(hidden_size, visible_size)
  
  list(
    visible_size = visible_size,
    weight = weight,
    hidden_bias = hidden_bias,
    visible_bias = visible_bias,
    learn_rate = initial_learn_rate
  )
}

# Performs a run of CD-1 optimization given a numeric data vector of length == visible_size
trainRBM <- function(rbm, data,
                     delta_learn_rate=0.0) {
  updated_weight <- function(weight, v, v_prime, h, h_prime, learn_rate) {
    positive_gradient <- outer(v, t(h))
    negative_gradient <- outer(v_prime, t(h_prime))
    net_gradient <- positive_gradient - negative_gradient
    delta_weight <- learn_rate * net_gradient
    
    # Remove redundant first and last dims of 1, correcting a dim mismatch with weight
    dim(delta_weight) <- c(dim(delta_weight)[2], dim(delta_weight)[3])
    
    weight + delta_weight
  }
  
  updated_bias <- function(bias, val, val_prime, learn_rate) {
    delta_bias <- learn_rate * (val - val_prime)
    bias + delta_bias
  }
  
  h = pracma::sigmoid((data %*% rbm$weight) + rbm$hidden_bias)
  v = pracma::sigmoid((h %*% t(rbm$weight)) + rbm$visible_bias)
  h_prime_sample = stochastic_binary_sample(h)
  
  v_reconstructed = pracma::sigmoid((h_prime_sample %*% t(rbm$weight)) + rbm$visible_bias)
  h_reconstructed = pracma::sigmoid((v_reconstructed %*% rbm$weight) + rbm$hidden_bias)
  
  learn_rate = rbm$learn_rate + delta_learn_rate
  
  list(
    visible_size = rbm$visible_size,
    weight = updated_weight(rbm$weight,
                            v, v_reconstructed,
                            h, h_reconstructed,
                            learn_rate),
    hidden_bias = updated_bias(rbm$hidden_bias, h, h_reconstructed, learn_rate),
    visible_bias = updated_bias(rbm$visible_bias, v, v_reconstructed, learn_rate),
    learn_rate = learn_rate
  )
}

encode_by_RBM <- function(rbm, data) {
  pracma::sigmoid((data %*% rbm$weight) + rbm$hidden_bias)
}

decode_by_RBM <- function(rbm, data) {
  pracma::sigmoid((data %*% t(rbm$weight)) + rbm$visible_bias)
}

dump_RBM_to_dir <- function(rbm, path = getwd()) {
  write.table(rbm$weight, file=file.path(path, "weight"))
  write.table(rbm$hidden_bias, file=file.path(path, "hidden_bias"))
  write.table(rbm$visible_bias, file=file.path(path, "visible_bias"))
}