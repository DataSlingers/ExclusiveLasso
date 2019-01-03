// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#include <limits>
#include <cmath>

#define EXLASSO_CHECK_USER_INTERRUPT_RATE 50
#define EXLASSO_CHECK_USER_INTERRUPT_RATE_GLM 10
#define EXLASSO_MAX_ITERATIONS_PROX 100
#define EXLASSO_MAX_ITERATIONS_PG 50000
#define EXLASSO_MAX_ITERATIONS_CD 10000
#define EXLASSO_MAX_ITERATIONS_GLM 500
#define EXLASSO_PREALLOCATION_FACTOR 10
#define EXLASSO_RESIZE_FACTOR 2
#define EXLASSO_FULL_LOOP_FACTOR 10
#define EXLASSO_FULL_LOOP_MIN 2
#define EXLASSO_GLM_FAMILY_GAUSSIAN 0
#define EXLASSO_GLM_FAMILY_LOGISTIC 1
#define EXLASSO_GLM_FAMILY_POISSON  2
#define EXLASSO_BACKTRACK_ALPHA 0.5
#define EXLASSO_BACKTRACK_BETA 0.8
#define EXLASSO_INF std::numeric_limits<double>::infinity()

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

    for(arma::sword g = arma::min(groups); g <= arma::max(groups); g++){
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
    for(arma::sword g = arma::min(groups); g <= arma::max(groups); g++){
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
                double thresh_level = arma::norm(beta_g, 1) - std::abs(beta_g(i));
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

    arma::uword beta_nnz_approx = EXLASSO_PREALLOCATION_FACTOR * p;
    arma::uword beta_nnz = 0;
    arma::umat Beta_storage_ind(2, beta_nnz_approx);
    arma::vec  Beta_storage_vec(beta_nnz_approx);

    // Storage for intercepts
    arma::vec Alpha(n_lambda, arma::fill::zeros);

    // Number of prox gradient iterations -- used to check for interrupts
    arma::uword k = 0;

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
        for(arma::uword j=0; j < p; j++){
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
Rcpp::List exclusive_lasso_glm_pg(const arma::mat& X,
                                  const arma::vec& y,
                                  const arma::ivec& groups,
                                  const arma::vec& lambda,
                                  const arma::vec& w,
                                  const arma::vec& o,
                                  int family,
                                  const arma::vec& lower_bound,
                                  const arma::vec& upper_bound,
                                  double thresh=1e-7,
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

    arma::uword beta_nnz_approx = EXLASSO_PREALLOCATION_FACTOR * p;
    arma::uword beta_nnz = 0;
    arma::umat Beta_storage_ind(2, beta_nnz_approx);
    arma::vec  Beta_storage_vec(beta_nnz_approx);

    arma::mat XtX = X1.t() * arma::diagmat(w/n) * X1;
    double L = arma::max(arma::eig_sym(XtX));

    // Storage for intercepts
    arma::vec Alpha(n_lambda, arma::fill::zeros);

    // Number of prox gradient iterations -- used to check for interrupts
    arma::uword k = 0;

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
            return arma::dot(w/n, arma::exp(linear_predictor) - y % linear_predictor);
        };

        g = [&](const arma::vec& beta){
            arma::vec linear_predictor = X1 * beta + o;
            return (-X1.t() * arma::diagmat(w/n) * (y - arma::exp(linear_predictor))).eval();
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
            ret.head(p-1) = exclusive_lasso_prox(beta.head(p-1),
                                                 groups,
                                                 lambda,
                                                 lower_bound,
                                                 upper_bound,
                                                 thresh_prox);
            ret(p-1) = beta(p-1);
            return ret;
        };
    } else {
        prox = [&](const arma::vec& beta, double lambda){
            return exclusive_lasso_prox(beta, groups, lambda, lower_bound, upper_bound, thresh_prox);
        };
    }

    // Iterate from highest to smallest lambda
    // to take advantage of warm starts for quick convergence
    // beta = 0 is a better guess for high lambda than small lambda

    for(int i=n_lambda - 1; i >= 0; i--){
        double t = L;
        do {
            beta_old = beta;
            const double f_old = f(beta);
            const arma::vec grad_old = g(beta);

            bool descent_achieved = false;

            while(!descent_achieved){
                // This is the back-tracking search of Parikh and Boyd
                // Proximal Algorithms, Section 4.2, who cite Beck and Teboulle
                // except we reset t to L at each iteration (lambda)
                const arma::vec z = prox(beta - t * grad_old, lambda(i) * t);
                const double f_z = f(z);

                if(f_z <= f_old + arma::dot(grad_old, z - beta) + 1 / (2.0 * t) * norm_sq(z-beta)){
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
        for(arma::uword j=0; j < p; j++){
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
    for(arma::uword i=0; i<p; i++){
        u(i) = arma::sum(arma::square(X.col(i)) % w);
    }

    // Like the working residual, we also store a working copy of
    // the groupwise norms to speed up calculating the soft-threshold
    // level
    //
    // If called via exclusive_lasso groups will be from 0 to `num_groups - 1`
    // so this is tight, but should be safe generally.
    arma::vec g_norms(arma::max(groups) + 1, arma::fill::zeros);

    arma::uword beta_nnz_approx = EXLASSO_PREALLOCATION_FACTOR * p;
    arma::uword beta_nnz = 0;
    arma::umat Beta_storage_ind(2, beta_nnz_approx);
    arma::vec  Beta_storage_vec(beta_nnz_approx);

    arma::vec beta_old(p, arma::fill::zeros);

    arma::vec Alpha(n_lambda, arma::fill::zeros); // Storage for intercepts

    // Number of cd iterations -- used to check for interrupts
    arma::uword k = 0;

    // For first iteration we want to loop over all variables since
    // we haven't identified the active set yet
    bool full_loop = true;
    arma::uword full_loop_count = 0; // Number of full loops completed
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
            for(arma::uword j=0; j < p; j++){
                double beta = beta_working(j);

                if((!full_loop) && (beta == 0)){
                    continue;
                }

                const arma::vec xj = X.col(j);
                r += xj * beta;

                arma::sword g = groups(j);
                g_norms(g) -= std::abs(beta);

                double z = arma::dot(r % w, xj);
                double lambda_til = nl * g_norms(g);

                beta = soft_thresh(z, lambda_til) / (u(j) + nl);

                // Box constraints
                if(apply_box_constraints){
                    beta = std::fmax(beta, lower_bound(j));
                    beta = std::fmin(beta, upper_bound(j));
                }

                r -= xj * beta;
                g_norms(g) += std::abs(beta);

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
        for(arma::uword j=0; j < p; j++){
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
Rcpp::List exclusive_lasso_glm_cd(const arma::mat& X,
                                  const arma::vec& y,
                                  const arma::ivec& groups,
                                  const arma::vec& lambda,
                                  const arma::vec& w,
                                  const arma::vec& o,
                                  int family,
                                  const arma::vec& lower_bound,
                                  const arma::vec& upper_bound,
                                  double thresh=1e-7,
                                  double thresh_prox=1e-7,
                                  bool intercept=true){

    arma::uword n = X.n_rows;
    arma::uword p = X.n_cols;
    arma::uword n_lambda = lambda.n_elem;

    bool apply_box_constraints = arma::any(lower_bound != -EXLASSO_INF) || arma::all(upper_bound != EXLASSO_INF);

    // Define helper functions for GLM SQA+CD
    //
    // 1) `f` -- negative log-likelihood (smooth part of objective function)
    // 2) `f_prime` -- derivative of `f` with respect to the input (beta)
    //
    // 3) `g` -- inverse link function (maps linear predictor to conditional mean)
    // 4) `g_prime` -- derivative of `g` (maps to the conditional variance, though
    //                 the SQA+CD algorithm doesn't use that fact)
    //
    // Note that all of these functions take the linear predictor
    // (eta := X*beta + alpha + offset) as input even when they take derivatives w.r.t. _beta_
    //
    // In many of these, we force evaluation of the return object
    // (which would otherwise be calculated lazily) to avoid a nasty segfault
    // resulting from un-mapped memory

    std::function<double(const arma::vec&)> f;
    std::function<arma::vec(const arma::vec&)> f_prime;
    std::function<arma::vec(const arma::vec&)> g;
    std::function<arma::vec(const arma::vec&)> g_prime;

    double alpha = 0; // Put this declaration early for lambda capture

    if(family == EXLASSO_GLM_FAMILY_GAUSSIAN){
        f = [&](const arma::vec& eta){
            return 1/(2.0 * n) * norm_sq(y - eta, w);
        };

        f_prime = [&](const arma::vec& eta){
            return (-X.t() * arma::diagmat(w/n) * (y - eta)).eval();
        };

        g = [&](const arma::vec& eta){
            return eta;
        };

        g_prime = [&](const arma::vec& eta){
            return (arma::vec(eta.n_elem, arma::fill::ones)).eval();
        };
    } else if(family == EXLASSO_GLM_FAMILY_LOGISTIC) {
        f = [&](const arma::vec& eta){
            return arma::dot(w/n, arma::log(1 + arma::exp(eta)) - y % eta);
        };

        f_prime = [&](const arma::vec& eta){
            return (-X.t() * arma::diagmat(w/n) * (y - inv_logit(eta))).eval();
        };

        g = [&](const arma::vec& eta){
            return (inv_logit(eta)).eval();
        };

        g_prime = [&](const arma::vec& eta){
            return ((inv_logit(eta)) % (1 - inv_logit(eta))).eval();
        };
    } else if(family == EXLASSO_GLM_FAMILY_POISSON){
        f = [&](const arma::vec& eta){
            return arma::dot(w/n, arma::exp(eta) - y % eta);
        };

        f_prime = [&](const arma::vec& eta){
            return (-X.t() * arma::diagmat(w/n) * (y - arma::exp(eta))).eval();
        };

        g = [&](const arma::vec& eta){
            return (arma::exp(eta)).eval();
        };

        g_prime = [&](const arma::vec& eta){
            return (arma::exp(eta)).eval();
        };
    } else {
        Rcpp::stop("[exclusive_lasso_glm] Unrecognized GLM family code.");
    }

    arma::vec beta_working(p, arma::fill::zeros);
    arma::vec eta = o;
    arma::vec mu  = g(eta);
    arma::vec fitting_weights = g_prime(eta);

    arma::vec z = (y - mu) / fitting_weights + eta;
    arma::vec combined_weights = fitting_weights % w;

    arma::vec r = z - o;

    // Like the working residual, we also store a working copy of
    // the groupwise norms to speed up calculating the soft-threshold
    // level
    //
    // If called via exclusive_lasso groups will be from 0 to `num_groups - 1`
    // so this is tight, but should be safe generally.
    arma::vec g_norms(arma::max(groups) + 1, arma::fill::zeros);

    arma::uword beta_nnz_approx = EXLASSO_PREALLOCATION_FACTOR * p;
    arma::uword beta_nnz = 0;
    arma::umat Beta_storage_ind(2, beta_nnz_approx);
    arma::vec  Beta_storage_vec(beta_nnz_approx);

    arma::vec beta_cd_old(p, arma::fill::zeros);
    arma::vec beta_sqa_old(p, arma::fill::zeros);
    double    alpha_old;

    arma::vec Alpha(n_lambda, arma::fill::zeros); // Storage for intercepts

    // Number of cd iterations -- used to check for interrupts
    arma::uword k = 0;

    // Number of SQA iterations
    arma::uword K = 0;

    // For first iteration we want to loop over all variables since
    // we haven't identified the active set yet
    bool full_loop = true;
    arma::uword full_loop_count = 0; // Number of full loops completed

    // We require at least EXLASSO_FULL_LOOP_MIN
    // full loops before moving to the next value
    // of lambda to ensure convergence

    // Iterate from highest to smallest lambda
    // to take advantage of
    // warm starts for sparsity
    for(int i=n_lambda - 1; i >= 0; i--){

        double nl = n * lambda(i);

        K = 0; // Reset SQA loop counter

        do{ // Outer SQA Loop
            beta_sqa_old = beta_working; // Save these for PN back-tracking
            alpha_old    = alpha;
            arma::vec u(p);
            for(arma::uword i=0; i<p; i++){
                u(i) = arma::sum(arma::square(X.col(i)) % combined_weights);
            }

            k = 0; // Reset CD loop counter

            do { // Inner CD Loop
                beta_cd_old = beta_working;
                for(arma::uword j=0; j < p; j++){
                    double beta = beta_working(j);

                    if((!full_loop) && (beta == 0)){
                        continue;
                    }

                    const arma::vec xj = X.col(j);
                    r += xj * beta;

                    arma::sword g = groups(j);
                    g_norms(g) -= std::abs(beta);

                    const double zeta = arma::dot(r % combined_weights, xj);
                    const double lambda_til = nl * g_norms(g);

                    beta = soft_thresh(zeta, lambda_til) / (u(j) + nl);

                    // Box constraints
                    if(apply_box_constraints){
                        beta = std::fmax(beta, lower_bound(j));
                        beta = std::fmin(beta, upper_bound(j));
                    }

                    r -= xj * beta;

                    g_norms(g) += std::abs(beta);
                    beta_working(j) = beta;
                }

                if(intercept){
                    r += alpha;
                    // It's safe (and numerically stable) to renormalize weights here
                    //
                    // We don't generally normalize weights since we need the quadratic
                    // approximation and the penalty to keep the correct ratio, but the
                    // intercept does not appear in the penalty
                    alpha = arma::dot(r, combined_weights) / arma::sum(combined_weights);
                    r -= alpha;
                }

                k++; // Increment CD loop counter

                if((k % EXLASSO_CHECK_USER_INTERRUPT_RATE) == 0){
                    Rcpp::checkUserInterrupt();
                }
                if(k >= EXLASSO_MAX_ITERATIONS_CD){
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

            // PN Back-tracking search
            // See Algorithm 2.1 of Byrd, Nocedal, Oztoprak (2016) or
            // Slide 9 of http://www.stat.cmu.edu/~ryantibs/convexopt-F15/lectures/17-prox-newton.pdf
            const arma::vec delta_beta  = beta_working - beta_sqa_old;
            const double    delta_alpha = alpha - alpha_old;
            double t = 1;

            const double penalty_old   = lambda(i) * exclusive_lasso_penalty(beta_sqa_old, groups);
            const double objective_old = f(eta) + penalty_old;
            const arma::vec grad_old   = f_prime(eta);

            // FIXME? -- This seems to get better results but I think mu - y is the actual gradient...
            const double  grad_int_old = arma::dot(w, y - mu)/n;

            int k_backtrack = 0;
            double descent_achieved;

            do {
                k_backtrack++;

                const arma::vec beta_new = (beta_sqa_old + t * delta_beta);
                const double   alpha_new = (alpha_old + t * delta_alpha);
                const arma::vec  eta_new = X * beta_new + o + alpha_new;

                const double penalty_new   = lambda(i) * exclusive_lasso_penalty(beta_new, groups);
                const double objective_new = f(eta_new) + penalty_new;

                if(objective_new < objective_old + EXLASSO_BACKTRACK_ALPHA * (t * arma::dot(delta_beta, grad_old) + grad_int_old * delta_alpha + penalty_new - penalty_old)){
                    descent_achieved = true;
                } else {
                    t *= EXLASSO_BACKTRACK_BETA;
                }

                if(k_backtrack > 20){
                    descent_achieved = true;
                    t = 0;
                }
            } while (!descent_achieved);

            beta_working = (beta_sqa_old + t * delta_beta);
            alpha = (alpha_old + t * delta_alpha);
            eta = X * beta_working + o + alpha;

            // Update SQA
            mu  = g(eta);
            fitting_weights = g_prime(eta);

            z = (y - mu) / fitting_weights + eta;
            combined_weights = fitting_weights % w;
            r = (z - eta);

            K += 1;

            if((K % EXLASSO_CHECK_USER_INTERRUPT_RATE_GLM) == 0){
                Rcpp::checkUserInterrupt();
            }
            if(K >= EXLASSO_MAX_ITERATIONS_GLM){
                Rcpp::stop("[ExclusiveLasso::exclusive_lasso_glm_cd] Maximum number of sequential quadratic approximations iterations reached.");
            }

        } while (arma::norm(beta_working - beta_sqa_old) > thresh);

        // Extend sparse matrix storage if needed
        if(beta_nnz >= Beta_storage_ind.n_cols - p){
            Beta_storage_ind.resize(2, EXLASSO_RESIZE_FACTOR * Beta_storage_ind.n_cols);
            Beta_storage_vec.resize(EXLASSO_RESIZE_FACTOR * Beta_storage_vec.n_elem);
        }

        // Load sparse matrix storage
        for(arma::uword j=0; j < p; j++){
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
        for(arma::sword g = arma::min(groups); g <= arma::max(groups); g++){
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
