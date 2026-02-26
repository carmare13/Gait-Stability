#pragma once
#include <armadillo>

// Returns posterior samples of H (length n_iter)
arma::vec bayesH_samples_core(const arma::vec& data,
                              unsigned int n_iter,
                              unsigned int seed = 42);