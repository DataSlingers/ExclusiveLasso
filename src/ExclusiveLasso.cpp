// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#define EXLASSO_CHECK_USER_INTERRUPT_RATE 50
#define EXLASSO_CHECK_USER_INTERRUPT_RATE_GLM 10
#define EXLASSO_MAX_ITERATIONS_PROX 100
#define EXLASSO_MAX_ITERATIONS_PG 50000
#define EXLASSO_MAX_ITERATIONS_CD 10000
#define EXLASSO_MAX_ITERATIONS_GLM 100
#define EXLASSO_PREALLOCATION_FACTOR 10
#define EXLASSO_RESIZE_FACTOR 2
#define EXLASSO_FULL_LOOP_FACTOR 10
#define EXLASSO_FULL_LOOP_MIN 2
#define EXLASSO_GLM_FAMILY_GAUSSIAN 0
#define EXLASSO_GLM_FAMILY_LOGISTIC 1
#define EXLASSO_GLM_FAMILY_POISSON  2
#define EXLASSO_BACKTRACK_BETA 0.8

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
arma::vec exclusive_lasso_prox(arma::vec z, const arma::ivec& groups,
                               double lambda, double thresh=1e-7){

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
Rcpp::List exclusive_lasso_gaussian_pg(const arma::mat& X, const arma::vec& y,
                                       const arma::ivec& groups, const arma::vec& lambda,
                                       const arma::vec& w, const arma::vec& o,
                                       double thresh=1e-7, double thresh_prox=1e-7,
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
            beta = exclusive_lasso_prox(z, groups, lambda(i)/L, thresh_prox);

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
Rcpp::List exclusive_lasso_glm_pg(const arma::mat& X, const arma::vec& y,
                                  const arma::ivec& groups, const arma::vec& lambda,
                                  const arma::vec& w, const arma::vec& o,
                                  int family, double thresh=1e-7,
                                  double thresh_prox=1e-7,
                                  bool intercept=true){

    // It's a bit tricky to handle intercepts directly in proximal gradient,
    // so we add a un penalized column of ones to X if intercept == true.
    //
    // If no intercept, then X1 is just X.

    arma::mat X1;
    if(intercept){
        arma::colvec one_vec = arma::ones(X.n_rows);
        X1 = arma::join_rows(X, one_vec);
    } else {
        X1 = X;
    }

    arma::uword n = X1.n_rows;
    arma::uword p = X1.n_cols;
    arma::uword n_lambda = lambda.n_elem;

    uint beta_nnz_approx = EXLASSO_PREALLOCATION_FACTOR * p;
    uint beta_nnz = 0;
    arma::umat Beta_storage_ind(2, beta_nnz_approx);
    arma::vec  Beta_storage_vec(beta_nnz_approx);

    arma::mat XtX = X1.t() * arma::diagmat(w/n) * X1;
    double L = arma::max(arma::eig_sym(XtX));

    // Storage for intercepts
    arma::vec Alpha(n_lambda, arma::fill::zeros);

    // Number of prox gradient iterations -- used to check for interrupts
    uint k = 0;

    arma::vec beta(p, arma::fill::zeros);
    arma::vec beta_old(p);

    // Define helper functions for GLM Prox Gradient
    //
    // 1) f -- smooth portion of objective function.
    //         Used in back-tracking choice of step-size
    // 2) g -- gradient of smooth (loss) portion
    //         Used in gradient update
    // In many of these, we force evaluation of the return object
    // (which would otherwise be calculated lazily) to avoid a nasty segfault
    // resulting from un-mapped memory

    std::function<double(const arma::vec&)> f;
    std::function<arma::vec(const arma::vec&)> g;

    if(family == EXLASSO_GLM_FAMILY_GAUSSIAN){
        f = [&](const arma::vec& beta){
            arma::vec linear_predictor = X1*beta + o;
            return 1/(2.0 * n) * norm_sq(y - linear_predictor, w);
        };

        g = [&](const arma::vec& beta){
            arma::vec linear_predictor = X1*beta + o;
            return (-X1.t() * arma::diagmat(w/n) * (y - linear_predictor)).eval();
        };
    } else if(family == EXLASSO_GLM_FAMILY_LOGISTIC) {
        // TODO -- can we write this in a way that doesn't compute
        //         linear_predictor redundantly?
        f = [&](const arma::vec& beta){
            arma::vec linear_predictor = X1 * beta + o;
            return arma::dot(w/n, arma::log(1 + arma::exp(linear_predictor)) - y % linear_predictor);
        };

        g = [&](const arma::vec& beta){
            arma::vec linear_predictor = X1 * beta + o;
            return (-X1.t() * arma::diagmat(w/n) * (y - inv_logit(linear_predictor))).eval();
        };
    } else if(family == EXLASSO_GLM_FAMILY_POISSON){
        // TODO -- can we write this in a way that doesn't compute
        //         linear_predictor redundantly?
        f = [&](const arma::vec& beta){
            arma::vec linear_predictor = X1 * beta + o;
            return arma::dot(w/n, y % linear_predictor - arma::exp(linear_predictor));
        };

        g = [&](const arma::vec& beta){
            arma::vec linear_predictor = X1 * beta + o;
            return (-X1.t() * arma::diagmat(w/n) * (y - arma::exp(linear_predictor))/n).eval();
        };
    } else {
        Rcpp::stop("[exclusive_lasso_glm] Unrecognized GLM family code.");
    }

    // To simplify the code, we define a wrapped proximal function
    // which remembers the group structure and the intercept for us
    std::function<arma::vec(const arma::vec&, double)> prox;

    if(intercept){
        prox = [&](const arma::vec& beta, double lambda){
            arma::vec ret(p);
            ret.head(p-1) = exclusive_lasso_prox(beta.head(p-1), groups,
                                                 lambda, thresh_prox);
            ret(p-1) = beta(p-1);
            return ret;
        };
    } else {
        prox = [&](const arma::vec& beta, double lambda){
            return exclusive_lasso_prox(beta, groups, lambda, thresh_prox);
        };
    }

    // Iterate from highest to smallest lambda
    // to take advantage of warm starts for quick convergence
    // beta = 0 is a better guess for high lambda than small lambda

    for(int i=n_lambda - 1; i >= 0; i--){
        double t = L;
        do {
            beta_old = beta;
            double f_old = f(beta);

            arma::vec grad_beta_old = g(beta);

            bool descent_achieved = false;

            while(!descent_achieved){
                // This is the back-tracking search of Parikh and Boyd
                // Proximal Algorithms, Section 4.2, who cite Beck and Teboulle
                // except we reset t to L at each iteration (lambda)
                arma::vec z = prox(beta - t * grad_beta_old, lambda(i) * t);
                double f_z = f(z);

                if(f_z <= f_old + arma::dot(grad_beta_old, z - beta) + 1 / (2.0 * t) * norm_sq(z-beta)){
                    descent_achieved = true;
                    beta = z;
                } else {
                    t *= EXLASSO_BACKTRACK_BETA;
                }
            }

            k++;
            if((k % EXLASSO_CHECK_USER_INTERRUPT_RATE_GLM) == 0){
                Rcpp::checkUserInterrupt();
            }
            if(k >= EXLASSO_MAX_ITERATIONS_PG * EXLASSO_MAX_ITERATIONS_GLM * n_lambda){
                Rcpp::stop("[ExclusiveLasso::exclusive_lasso_glm_pg] Maximum number of proximal gradient iterations reached.");
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
                if(intercept && (j == p - 1)){
                    // Handle intercept specially
                    Alpha(i) = beta(j);
                } else {
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
        }
    }

    arma::sp_mat Beta(Beta_storage_ind.cols(0, beta_nnz - 1),
                      Beta_storage_vec.subvec(0, beta_nnz - 1),
                      intercept ? p - 1 : p, // Drop column of X if intercept
                      n_lambda);

    if(intercept){
        return Rcpp::List::create(Rcpp::Named("intercept")=Alpha,
                                  Rcpp::Named("coef")=Beta);
    } else {
        return Rcpp::List::create(Rcpp::Named("intercept")=R_NilValue,
                                  Rcpp::Named("coef")=Beta);
    }
}


// [[Rcpp::export]]
Rcpp::List exclusive_lasso_gaussian_cd(const arma::mat& X, const arma::vec& y,
                                       const arma::ivec& groups, const arma::vec& lambda,
                                       const arma::vec& w, const arma::vec& o,
                                       double thresh=1e-7, bool intercept=true){

    arma::uword n = X.n_rows;
    arma::uword p = X.n_cols;
    arma::uword n_lambda = lambda.n_elem;

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


// [[Rcpp::export]]
Rcpp::List exclusive_lasso_glm_cd(const arma::mat& X, const arma::vec& y,
                                  const arma::ivec& groups, const arma::vec& lambda,
                                  const arma::vec& w, const arma::vec& o,
                                  int family,
                                  double thresh=1e-7,
                                  double thresh_prox=1e-7,
                                  bool intercept=true){

    arma::mat X1;

    if(intercept){
        arma::colvec one_vec = arma::ones(X.n_rows);
        X1 = arma::join_rows(X, one_vec);
    } else {
        X1 = X;
    }

    arma::uword n = X1.n_rows;
    arma::uword p = X1.n_cols;
    arma::uword n_lambda = lambda.n_elem;

    // Define helper functions for GLM SQA+CD
    //
    // 1) l1 -- gradient of smooth portion of objective function.
    //          Used in calculation of working response
    // 2) l2 -- diagonal of Hessian of smooth portion of objective function.
    //          Used to weight entries
    //
    // Both take a vector (eta == linear predictor = X*beta + o) as input
    //
    // In many of these, we force evaluation of the return object
    // (which would otherwise be calculated lazily) to avoid a nasty segfault
    // resulting from un-mapped memory

    std::function<arma::vec(const arma::vec&)> l1;
    std::function<arma::vec(const arma::vec&)> l2;

    if(family == EXLASSO_GLM_FAMILY_GAUSSIAN){
        l1 = [&](const arma::vec& eta){
            return ((eta - y)).eval();
        };

        l2 = [&](const arma::vec& eta){
            return (arma::vec(eta.n_elem, arma::fill::ones)).eval();
        };
    } else if(family == EXLASSO_GLM_FAMILY_LOGISTIC) {
        l1 = [&](const arma::vec& eta){
            return ((inv_logit(eta) - y)).eval();
        };

        l2 = [&](const arma::vec& eta){
            return ((inv_logit(eta)) % (1 - inv_logit(eta))).eval();
        };
    } else if(family == EXLASSO_GLM_FAMILY_POISSON){
        l1 = [&](const arma::vec& eta){
            return ((arma::exp(eta) - y)).eval();
        };

        l2 = [&](const arma::vec& eta){
            return (arma::exp(eta)).eval();
        };
    } else {
        Rcpp::stop("[exclusive_lasso_glm] Unrecognized GLM family code.");
    }

    arma::vec beta_working(p, arma::fill::zeros);
    arma::vec eta = o;

    arma::vec l1_vec = l1(eta);
    arma::vec l2_vec = l2(eta);

    arma::vec z = o - l1_vec / l2_vec;
    arma::vec omega = l2_vec % w;
    arma::vec r = z - o;

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

    arma::vec beta_cd_old(p, arma::fill::zeros);
    arma::vec beta_sqa_old(p, arma::fill::zeros);

    arma::vec Alpha(n_lambda, arma::fill::zeros); // Storage for intercepts

    // Number of cd iterations -- used to check for interrupts
    uint k = 0;

    // Number of SQA iterations
    uint K = 0;

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

        do{ // Outer SQA Loop
            beta_sqa_old = beta_working;
            arma::vec u(p);
            for(uint i=0; i<p; i++){
                u(i) = arma::sum(arma::square(X1.col(i)) % omega);
            }

            do { // Inner CD Loop
                beta_cd_old = beta_working;
                for(int j=0; j < p; j++){
                    double beta = beta_working(j);

                    if((!full_loop) && (beta == 0)){
                        continue;
                    }

                    if(intercept & (j == p - 1)){
                        // Don't need to threshold here, so simpler update
                        const arma::vec xj = X1.col(j);

                        r += xj * beta;
                        beta = arma::dot(r % omega, xj) / n;
                        r -= xj * beta;

                        beta_working(j) = beta;
                        continue;
                    }

                    const arma::vec xj = X1.col(j);
                    r += xj * beta;

                    uint g = groups(j);
                    g_norms(g) -= fabs(beta);

                    double z1 = arma::dot(r % omega, xj);
                    double lambda_til = nl * g_norms(g);

                    beta = soft_thresh(z1, lambda_til) / (u(j) + nl);
                    r -= xj * beta;
                    g_norms(g) += fabs(beta);

                    beta_working(j) = beta;
                }

                k++; // Increment CD loop counter
                if((k % EXLASSO_CHECK_USER_INTERRUPT_RATE) == 0){
                    Rcpp::checkUserInterrupt();
                }
                if(k >= EXLASSO_MAX_ITERATIONS_CD * EXLASSO_MAX_ITERATIONS_GLM * n_lambda){
                    Rcpp::stop("[ExclusiveLasso::exclusive_lasso_glm_cd] Maximum number of coordinate descent iterations reached.");
                }

                if(full_loop){
                    full_loop_count++;
                    // Only check this if not already in full loop mode
                } else if(arma::norm(beta_working - beta_cd_old) < EXLASSO_FULL_LOOP_FACTOR * thresh){
                    // If it looks like we're closing in on a solution,
                    // switch to looping over all variables
                    full_loop = true;
                }

            } while((full_loop_count < EXLASSO_FULL_LOOP_MIN) || arma::norm(beta_working - beta_cd_old) > thresh_prox);

            // Switch back to active set loops for next iteration
            full_loop = false;
            full_loop_count = 0;

            // Update SQA
            eta = X1 * beta_working + o;
            l1_vec = l1(eta);
            l2_vec = l2(eta);

            z = eta - l1_vec / l2_vec;
            omega = l2_vec % w;
            r = (z - eta);

            K += 1;
            if((K % EXLASSO_CHECK_USER_INTERRUPT_RATE_GLM) == 0){
                Rcpp::checkUserInterrupt();
            }
            if(K >= EXLASSO_MAX_ITERATIONS_GLM * n_lambda){
                Rcpp::stop("[ExclusiveLasso::exclusive_lasso_glm_cd] Maximum number of sequential quadratic approximations iterations reached.");
            }

        } while (arma::norm(beta_working - beta_sqa_old) > thresh);

        // Extend sparse matrix storage if needed
        if(beta_nnz >= Beta_storage_ind.n_cols - p){
            Beta_storage_ind.resize(2, EXLASSO_RESIZE_FACTOR * Beta_storage_ind.n_cols);
            Beta_storage_vec.resize(EXLASSO_RESIZE_FACTOR * Beta_storage_vec.n_elem);
        }

        // Load sparse matrix storage
        for(uint j=0; j < p; j++){
            if(beta_working(j) != 0){
                if(intercept && (j == p - 1)){
                    // Handle intercept specially
                    Alpha(i) = beta_working(j);
                } else {
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
        }
    }

    arma::sp_mat Beta(Beta_storage_ind.cols(0, beta_nnz - 1),
                      Beta_storage_vec.subvec(0, beta_nnz - 1),
                      intercept ? p - 1 : p, // Drop column of X if intercept
                      n_lambda);

    if(intercept){
        return Rcpp::List::create(Rcpp::Named("intercept")=Alpha,
                                  Rcpp::Named("coef")=Beta);
    } else {
        return Rcpp::List::create(Rcpp::Named("intercept")=R_NilValue,
                                  Rcpp::Named("coef")=Beta);
    }
}
