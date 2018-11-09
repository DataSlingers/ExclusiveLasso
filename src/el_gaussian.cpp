#include "ExclusiveLasso.h"
#include "el_univariate_helpers.h"

// [[Rcpp::export]]
Rcpp::List exclusive_lasso_gaussian_pg(const arma::mat& X,
                                       const arma::vec& y,
                                       const arma::ivec& groups,
                                       const arma::vec& lambda,
                                       const arma::vec& w,
                                       const arma::vec& o,
                                       const arma::vec& lower_bound,
                                       const arma::vec& upper_bound,
                                       double thresh=1e-7,
                                       double thresh_prox=1e-7,
                                       bool intercept=true){

    arma::uword n = X.n_rows;
    arma::uword p = X.n_cols;
    arma::uword n_lambda = lambda.n_elem;

    arma::mat XtX = X.t() * arma::diagmat(w/n) * X;
    arma::vec Xty = X.t() * (w % (y - o)/n);
    double L = arma::max(arma::eig_sym(XtX));

    uint beta_nnz_approx = EXLASSO_PREALLOCATION_FACTOR * p;
    uint beta_nnz = 0;
    arma::umat Beta_storage_ind(2, beta_nnz_approx);
    arma::vec  Beta_storage_vec(beta_nnz_approx);

    // Storage for intercepts
    arma::vec Alpha(n_lambda, arma::fill::zeros);

    // Number of prox gradient iterations -- used to check for interrupts
    uint k = 0;

    arma::vec beta(p, arma::fill::zeros);
    double alpha = 0;

    arma::vec N = X.t() * arma::diagmat(w/n) * arma::colvec(n, arma::fill::ones);

    // Iterate from highest to smallest lambda
    // to take advantage of
    // warm starts for sparsity

    for(int i=n_lambda - 1; i >= 0; i--){
        arma::vec beta_old(p);

        do {
            beta_old = beta;
            arma::vec z = beta + 1/L * (Xty - N * alpha - XtX * beta);
            beta = exclusive_lasso_prox(z,
                                        groups,
                                        lambda(i)/L,
                                        lower_bound,
                                        upper_bound,
                                        thresh_prox);

            if(intercept){
                alpha = arma::dot(w/n, y - o - X * beta);
            }

            k++;
            if((k % EXLASSO_CHECK_USER_INTERRUPT_RATE) == 0){
                Rcpp::checkUserInterrupt();
            }
            if(k >= EXLASSO_MAX_ITERATIONS_PG * n_lambda){
                Rcpp::stop("[ExclusiveLasso::exclusive_lasso_gaussian] Maximum number of proximal gradient iterations reached.");
            }

        } while(arma::norm(beta - beta_old) > thresh);

        // Extend sparse matrix storage if needed
        if(beta_nnz >= Beta_storage_ind.n_cols - p){
            Beta_storage_ind.resize(2, EXLASSO_RESIZE_FACTOR * Beta_storage_ind.n_cols);
            Beta_storage_vec.resize(EXLASSO_RESIZE_FACTOR * Beta_storage_vec.n_elem);
        }

        // Load sparse matrix storage
        for(uint j=0; j < p; j++){
            if(beta(j) != 0){
                // We want to have a coefficient matrix
                // where rows are features and columns are values of lambda
                //
                // Armadillo's batch constructor takes (row, column) pairs
                // so we build as  (j = feature, i = lambda_index)
                Beta_storage_ind(0, beta_nnz) = j;
                Beta_storage_ind(1, beta_nnz) = i;
                Beta_storage_vec(beta_nnz) = beta(j);
                beta_nnz += 1;
            }
        }

        if(intercept){
            Alpha(i) = alpha;
        }
    }

    arma::sp_mat Beta(Beta_storage_ind.cols(0, beta_nnz - 1),
                      Beta_storage_vec.subvec(0, beta_nnz - 1),
                      p, n_lambda);

    if(intercept){
        return Rcpp::List::create(Rcpp::Named("intercept")=Alpha,
                                  Rcpp::Named("coef")=Beta);
    } else {
        return Rcpp::List::create(Rcpp::Named("intercept")=R_NilValue,
                                  Rcpp::Named("coef")=Beta);
    }
}

// [[Rcpp::export]]
Rcpp::List exclusive_lasso_gaussian_cd(const arma::mat& X,
                                       const arma::vec& y,
                                       const arma::ivec& groups,
                                       const arma::vec& lambda,
                                       const arma::vec& w,
                                       const arma::vec& o,
                                       const arma::vec& lower_bound,
                                       const arma::vec& upper_bound,
                                       double thresh=1e-7,
                                       bool intercept=true){

    arma::uword n = X.n_rows;
    arma::uword p = X.n_cols;
    arma::uword n_lambda = lambda.n_elem;

    bool apply_box_constraints = arma::any(lower_bound != -EXLASSO_INF) || arma::all(upper_bound != EXLASSO_INF);

    arma::vec r = y - o;
    arma::vec beta_working(p, arma::fill::zeros);
    double alpha = 0;

    arma::vec u(p);
    for(uint i=0; i<p; i++){
        u(i) = arma::sum(arma::square(X.col(i)) % w);
    }

    // Like the working residual, we also store a working copy of
    // the groupwise norms to speed up calculating the soft-threshold
    // level
    //
    // If called via exclusive_lasso groups will be from 0 to `num_groups - 1`
    // so this is tight, but should be safe generally.
    arma::vec g_norms(arma::max(groups) + 1, arma::fill::zeros);

    uint beta_nnz_approx = EXLASSO_PREALLOCATION_FACTOR * p;
    uint beta_nnz = 0;
    arma::umat Beta_storage_ind(2, beta_nnz_approx);
    arma::vec  Beta_storage_vec(beta_nnz_approx);

    arma::vec beta_old(p, arma::fill::zeros);

    arma::vec Alpha(n_lambda, arma::fill::zeros); // Storage for intercepts

    // Number of cd iterations -- used to check for interrupts
    uint k = 0;

    // For first iteration we want to loop over all variables since
    // we haven't identified the active set yet
    bool full_loop = true;
    uint full_loop_count = 0; // Number of full loops completed
    // We require at least EXLASSO_FULL_LOOP_MIN
    // full loops before moving to the next value
    // of lambda to ensure convergence

    // Iterate from highest to smallest lambda
    // to take advantage of
    // warm starts for sparsity
    for(int i=n_lambda - 1; i >= 0; i--){
        double nl = n * lambda(i);

        do {
            beta_old = beta_working;
            for(int j=0; j < p; j++){
                double beta = beta_working(j);

                if((!full_loop) && (beta == 0)){
                    continue;
                }

                const arma::vec xj = X.col(j);
                r += xj * beta;

                uint g = groups(j);
                g_norms(g) -= fabs(beta);

                double z = arma::dot(r % w, xj);
                double lambda_til = nl * g_norms(g);

                beta = soft_thresh(z, lambda_til) / (u(j) + nl);

                // Box constraints
                if(apply_box_constraints){
                    beta = std::fmax(beta, lower_bound(j));
                    beta = std::fmin(beta, upper_bound(j));
                }

                r -= xj * beta;
                g_norms(g) += fabs(beta);

                beta_working(j) = beta;
            }

            if(intercept){
                r += alpha;
                alpha = arma::dot(r, w/n);
                r -= alpha;
            }

            k++; // Increment loop counter

            if((k % EXLASSO_CHECK_USER_INTERRUPT_RATE) == 0){
                Rcpp::checkUserInterrupt();
            }
            if(k >= EXLASSO_MAX_ITERATIONS_CD * n_lambda){
                Rcpp::stop("[ExclusiveLasso::exclusive_lasso_gaussian] Maximum number of coordinate descent iterations reached.");
            }

            if(full_loop){
                full_loop_count++;
                // Only check this if not already in full loop mode
            } else if(arma::norm(beta_working - beta_old) < EXLASSO_FULL_LOOP_FACTOR * thresh){
                // If it looks like we're closing in on a solution,
                // switch to looping over all variables
                full_loop = true;
            }

        } while((full_loop_count < EXLASSO_FULL_LOOP_MIN) || arma::norm(beta_working - beta_old) > thresh);

        // Switch back to active set loops for next iteration
        full_loop = false;
        full_loop_count = 0;

        // Extend sparse matrix storage if needed
        if(beta_nnz >= Beta_storage_ind.n_cols - p){
            Beta_storage_ind.resize(2, EXLASSO_RESIZE_FACTOR * Beta_storage_ind.n_cols);
            Beta_storage_vec.resize(EXLASSO_RESIZE_FACTOR * Beta_storage_vec.n_elem);
        }

        // Load sparse matrix storage
        for(uint j=0; j < p; j++){
            if(beta_working(j) != 0){
                // We want to have a coefficient matrix
                // where rows are features and columns are values of lambda
                //
                // Armadillo's batch constructor takes (row, column) pairs
                // so we build as  (j = feature, i = lambda_index)
                Beta_storage_ind(0, beta_nnz) = j;
                Beta_storage_ind(1, beta_nnz) = i;
                Beta_storage_vec(beta_nnz) = beta_working(j);
                beta_nnz += 1;
            }
        }

        Alpha(i) = alpha;
    }

    arma::sp_mat Beta(Beta_storage_ind.cols(0, beta_nnz - 1),
                      Beta_storage_vec.subvec(0, beta_nnz - 1),
                      p,
                      n_lambda);

    if(intercept){
        return Rcpp::List::create(Rcpp::Named("intercept")=Alpha,
                                  Rcpp::Named("coef")=Beta);
    } else {
        return Rcpp::List::create(Rcpp::Named("intercept")=R_NilValue,
                                  Rcpp::Named("coef")=Beta);
    }
}
