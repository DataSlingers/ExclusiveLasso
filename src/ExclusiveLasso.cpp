// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#define EXLASSO_CHECK_USER_INTERRUPT_RATE 20

// We only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

double soft_thresh(double x, double lambda){
    if(x > lambda){
        return x - lambda;
    } else if(x < - lambda){
        return x + lambda;
    } else {
        return 0;
    }
}

// [[Rcpp::export]]
arma::vec exclusive_lasso_prox(arma::vec z, const arma::ivec& groups,
                               double lambda, double thresh=1e-7){

    arma::uword p = z.n_elem;
    arma::vec beta(p);

    // TODO -- parallelize?
    // Loop over groups
    for(arma::uword g = arma::min(groups); g <= arma::max(groups); g++){
        // Identify elements in group
        arma::uvec g_ix = find(g == groups);
        int g_n_elem = g_ix.n_elem;

        arma::vec z_g = z(g_ix);
        arma::vec beta_g  = z_g;

        arma::vec beta_g_old;

        do{
            beta_g_old = beta_g;
            for(int i=0; i<g_n_elem; i++){
                double thresh_level = arma::norm(beta_g, 1) - fabs(beta_g(i));
                beta_g(i) = 1/(lambda + 1) * soft_thresh(z_g(i), lambda * thresh_level);
            }
        } while (arma::norm(beta_g - beta_g_old) > thresh);

        beta(g_ix) = beta_g;
    }
    return beta;
}

