#include "ExclusiveLasso.h"
#include "el_univariate_helpers.h"

double soft_thresh(double x, double lambda){
    if(x > lambda){
        return x - lambda;
    } else if(x < - lambda){
        return x + lambda;
    } else {
        return 0;
    }
}

arma::vec inv_logit(const arma::vec& x){
    return 1.0 / (1.0 + arma::exp(-x));
}

double norm_sq(const arma::vec& x){
    return arma::dot(x, x);
}

double norm_sq(const arma::vec& x, const arma::vec& w){
    return arma::dot(x, w % x);
}

// [[Rcpp::export]]
double exclusive_lasso_penalty(const arma::vec& x, const arma::ivec& groups){
    double ans = 0;

    for(arma::uword g = arma::min(groups); g <= arma::max(groups); g++){
        ans += pow(arma::norm(x(arma::find(g == groups)), 1), 2);
    }

    return ans / 2.0;
}

// [[Rcpp::export]]
arma::vec exclusive_lasso_prox(const arma::vec& z,
                               const arma::ivec& groups,
                               double lambda,
                               const arma::vec& lower_bound,
                               const arma::vec& upper_bound,
                               double thresh=1e-7){

    bool apply_box_constraints = arma::any(lower_bound != -EXLASSO_INF) || arma::all(upper_bound != EXLASSO_INF);

    arma::uword p = z.n_elem;
    arma::vec beta(p);

    // TODO -- parallelize?
    // Loop over groups
    for(arma::uword g = arma::min(groups); g <= arma::max(groups); g++){
        // Identify elements in group
        arma::uvec g_ix = arma::find(g == groups);
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

                // Impose box constraints
                if(apply_box_constraints){
                    beta_g(i) = std::fmax(beta_g(i), lower_bound(i));
                    beta_g(i) = std::fmin(beta_g(i), upper_bound(i));
                }
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
