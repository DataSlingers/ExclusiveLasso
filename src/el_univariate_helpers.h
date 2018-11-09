#include "ExclusiveLasso.h"

double soft_thresh(double x, double lambda);

arma::vec inv_logit(const arma::vec& x);
double norm_sq(const arma::vec& x);
double norm_sq(const arma::vec& x, const arma::vec& w);
double exclusive_lasso_penalty(const arma::vec& x, const arma::ivec& groups);
arma::vec exclusive_lasso_prox(const arma::vec& z,
                               const arma::ivec& groups,
                               double lambda,
                               const arma::vec& lower_bound,
                               const arma::vec& upper_bound,
                               double thresh);
