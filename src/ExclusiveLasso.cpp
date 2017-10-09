// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#define EXLASSO_CHECK_USER_INTERRUPT_RATE 20
#define EXLASSO_MAX_ITERATIONS_PROX 100
#define EXLASSO_MAX_ITERATIONS_PG 100000

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
            int k = 0;
            beta_g_old = beta_g;
            for(int i=0; i<g_n_elem; i++){
                double thresh_level = arma::norm(beta_g, 1) - fabs(beta_g(i));
                beta_g(i) = 1/(lambda + 1) * soft_thresh(z_g(i), lambda * thresh_level);
            }

            k++;
            if(k >= EXLASSO_MAX_ITERATIONS_PROX){
                Rcpp::stop("[ExclusiveLasso::exclusive_lasso_prox] Maximum number of iterations reached.");
            }
        } while (arma::norm(beta_g - beta_g_old) > thresh);

        beta(g_ix) = beta_g;
    }
    return beta;
}

// [[Rcpp::export]]
arma:: mat exclusive_lasso_gaussian(const arma::mat& X, const arma::vec& y,
                                    const arma::ivec& groups, const arma::vec& lambda,
                                    const arma::vec& w, const arma::vec& o,
                                    double thresh_pg=1e-7, double thresh_prox=1e-7){

    arma::uword n = X.n_rows;
    arma::uword p = X.n_cols;
    arma::uword n_lambda = lambda.n_elem;

    arma::mat XtX = X.t() * arma::diagmat(w/n) * X;
    arma::vec Xty = X.t() * (w % (y - o)/n);
    double L = arma::max(arma::eig_sym(XtX));

    arma::mat Beta(p, n_lambda); Beta.fill(NA_REAL);

    // Number of prox gradient iterations -- used to check for interrupts
    uint k = 0;

    // Iterate from highest to smallest lambda
    // to take advantage of
    // warm starts for sparsity
    for(int i=n_lambda - 1; i >= 0; i--){
        arma::vec beta(p);
        if(i == n_lambda - 1){
            beta.zeros(p);
        } else {
            beta = Beta.col(i + 1);
        }

        arma::vec beta_old(p);

        do {
            beta_old = beta;
            arma::vec z = beta + 1/L * (Xty - XtX * beta);
            beta = exclusive_lasso_prox(z, groups, lambda(i)/L, thresh_prox);

            k++;
            if((k % EXLASSO_CHECK_USER_INTERRUPT_RATE) == 0){
                Rcpp::checkUserInterrupt();
            }
            if(k >= EXLASSO_MAX_ITERATIONS_PG){
                Rcpp::stop("[ExclusiveLasso::exclusive_lasso_gaussian] Maximum number of iterations reached.");
            }

        } while(arma::norm(beta - beta_old) > thresh_pg);

        Beta.col(i) = beta;
    }

    return Beta;
}
