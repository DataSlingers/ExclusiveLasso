#include "ExclusiveLasso.h"

// Calculate degrees of freedom using Theorem 5 of Campbell and Allen (EJS, 2017)
// NB: Since we use a 1/2n scaling instead of a 1/2 scaling for the EL problem,
//     we multiply the lambda in the denominator by a extra n (X.n_rows)
//
// [[Rcpp::export]]
arma::mat calculate_exclusive_lasso_df(const arma::mat& X,
                                       const arma::vec& lambda_vec,
                                       const arma::ivec& groups,
                                       const arma::sp_mat& coefs){

    arma::vec res(lambda_vec.n_elem, arma::fill::zeros);

    for(arma::uword ix = 0; ix < lambda_vec.size(); ix++){
        const double lambda  = lambda_vec(ix);

        // Pull out the associated column of coefs as a (dense) vector
        // For whatever reason, coefs.col(ix) doesn't work here...
        arma::vec coef(X.n_cols, arma::fill::zeros);
        for(arma::uword j = 0; j < X.n_cols; j++){
            coef(j) = coefs(j, ix);
        }

        const arma::uvec S  = arma::find(coef != 0.0);
        arma::mat M(X.n_cols, X.n_cols, arma::fill::zeros);

        // Loop over groups to build M matrix
        for(arma::uword g = arma::min(groups); g <= arma::max(groups); g++){
            // Identify elements in group
            arma::uvec g_ix = arma::find(g == groups);
            arma::vec  s_g  = arma::sign(coef(g_ix));
            M.submat(g_ix, g_ix) = s_g * s_g.t();
        }

        M.diag() += 0.0001; // For numerical stability of inverses
        const arma::mat X_S = X.cols(S);
        const arma::mat proj_mat = X_S * arma::solve(X_S.t() * X_S + X.n_rows * lambda * M.submat(S, S), X_S.t());
        res(ix) = arma::trace(proj_mat);
        Rcpp::checkUserInterrupt();
    }

    return res;
}
