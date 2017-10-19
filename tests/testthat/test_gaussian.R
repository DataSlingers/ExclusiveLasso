context("Gaussian response GLM works")

test_that("Input validation works", {
    set.seed(123)
    ## Good inputs
    n <- 200
    p <- 500
    g <- 10

    X <- matrix(rnorm(n * p), ncol=p)
    y <- rnorm(n)
    groups <- rep(1:g, length.out=p)

    weights <- runif(n, 0.5, 1.5); weights <- weights / sum(weights) * n;
    offset <- runif(n, -0.5, 0.5);

    lambda <- seq(0.5, 5, length.out=100)
    thresh <- 1e-2; thresh_prox <- 1e-2 ## Low values here speed up tests

    expect_silent(exclusive_lasso(X, y, groups=groups,
                                  weights=weights, offset=offset,
                                  lambda=lambda, family="gaussian",
                                  thresh=thresh, thresh_prox=thresh_prox))

    ## X, y match
    expect_error(exclusive_lasso(X, rep(y, 2), groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="gaussian",
                                 thresh=thresh, thresh_prox=thresh_prox))

    expect_error(exclusive_lasso(X, rep(y, length.out=n-1), groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="gaussian",
                                 thresh=thresh, thresh_prox=thresh_prox))

    ## groups check
    expect_error(exclusive_lasso(X, y,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="gaussian",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=rep(groups, 2),
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="gaussian",
                                 thresh=thresh, thresh_prox=thresh_prox))

    ## Weights check
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=rep(weights, 3), offset=offset,
                                 lambda=lambda, family="gaussian",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=groups,
                                  weights= -1 * weights, offset=offset,
                                  lambda=lambda, family="gaussian",
                                  thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights= 0 * weights, offset=offset,
                                 lambda=lambda, family="gaussian",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_warning(exclusive_lasso(X, y, groups=groups,
                                 weights= 2 * weights, offset=offset,
                                 lambda=lambda, family="gaussian",
                                 thresh=thresh, thresh_prox=thresh_prox))

    ## Offsets check
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=rep(offset, 2),
                                 lambda=lambda, family="gaussian",
                                 thresh=thresh, thresh_prox=thresh_prox))

    ## Convergence thresholds check
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="gaussian",
                                 thresh=-1 * thresh, thresh_prox=thresh_prox))

    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="gaussian",
                                 thresh=thresh, thresh_prox=-1 * thresh_prox))

    ## Lambda check
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 nlambda=-30, family="gaussian",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 nlambda=0, family="gaussian",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=-lambda, family="gaussian",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 lambda.min.ratio=2, family="gaussian",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_warning(exclusive_lasso(X, y, groups=groups,
                                   weights=weights, offset=offset,
                                   lambda=rev(lambda), family="gaussian",
                                   thresh=thresh, thresh_prox=thresh_prox))

})

test_that("Dynamic defaults work", {
    set.seed(100)

    n <- 200
    p <- 500
    g <- 10

    X <- matrix(rnorm(n * p), ncol=p)
    y <- rnorm(n)
    groups <- rep(1:g, length.out=p)

    elfit <- exclusive_lasso(X, y, groups,
                             thresh_prox=1e-2, thresh=1e-2)

    expect_true(all(elfit$weights == 1))
    expect_true(all(elfit$offset == 0))
    expect_equal(length(elfit$lambda),
                 formals(exclusive_lasso)$nlambda)
    expect_equal(min(elfit$lambda)/max(elfit$lambda),
                 eval(formals(exclusive_lasso)$lambda.min.ratio,
                      envir=data.frame(nobs=n, nvars=p)))

})

test_that("Preserves column names", {
    set.seed(500)

    n <- 200
    p <- 500
    g <- 10

    X <- matrix(rnorm(n * p), ncol=p);
    colnames(X) <- paste0("A", 1:p)
    y <- rnorm(n)
    groups <- rep(1:g, length.out=p)

    elfit <- exclusive_lasso(X, y, groups,
                             thresh_prox=1e-2,
                             thresh=1e-2)

    expect_equal(rownames(elfit$coef),
                 colnames(X))
})

test_that("Standardization works", {
    set.seed(538)

    n <- 200
    p <- 500
    g <- 10

    X <- matrix(rnorm(n * p), ncol=p) %*% diag(seq(from=1, to=20, length.out=p))
    X[] <- scale(X, center=TRUE, scale=FALSE)

    colnames(X) <- paste0("A", 1:p)

    beta <- rep(0, p); beta[1:g] <- 2
    y <- X %*% beta + rnorm(n)
    groups <- rep(1:g, length.out=p)

    X_sc <- X; X_sc[] <- scale(X_sc)

    elfit <- exclusive_lasso(X, y, groups,
                             thresh_prox=1e-8,
                             thresh=1e-8)

    elfit_sc <- exclusive_lasso(X_sc, y, groups, lambda=elfit$lambda,
                                standardize=FALSE,
                                thresh_prox=1e-8,
                                thresh=1e-8)

    expect_equal(elfit$intercept, elfit_sc$intercept)
    expect_equal(scale(elfit$X), elfit_sc$X, check.attributes=FALSE)
    ## If Xsc gets multiplied by attr(... "scale")
    ## then elfit_sc$coef gets divided
    expect_equal(elfit$coef,
                 elfit_sc$coef / attr(scale(X), "scaled:scale"))

})

test_that("Returns prox for orthogonal case", {
    ## If X is orthogonal then
    ##
    ## argmin_{beta} \frac{1}{2n} |y - X*\beta|_2^2 + \lambda P(\beta) \\
    ## argmin_{beta} \frac{1}{2n} |X^Ty - X^T * X*\beta|_2^2 + \lambda P(\beta) \\
    ## argmin_{beta} \frac{1}{2n} |X^Ty - \beta|_2^2 + \lambda P(\beta) \\
    ## argmin_{beta} \frac{1}{2} |X^Ty - \beta|_2^2 + n * \lambda P(\beta) \\
    ## = prox_{n * lambda * P()}(X^Ty)
    ##
    set.seed(240)

    n <- 100
    p <- 100
    g <- 10

    X <- matrix(rnorm(n * p), ncol=p); X <- qr.Q(qr(X))
    beta <- numeric(p); beta[1:g] <- 2
    y <- X %*% beta + rnorm(n)

    groups <- rep(1:g, length.out=p)
    lambda <- 0.75

    elfit <- exclusive_lasso(X, y, groups, lambda=lambda,
                             standardize=FALSE, intercept=FALSE)
    prox_coefs <- ExclusiveLasso:::exclusive_lasso_prox(crossprod(X, y),
                                                        groups, lambda * n)

    expect_equal(as.matrix(elfit$coef), prox_coefs, check.attributes=FALSE)
})

test_that("Returns ridge with trivial group structure", {
    set.seed(5454)

    ## Low-Dimensional
    n <- 200
    p <- 20
    groups <- 1:p

    beta <- rep(1, p)
    X <- matrix(rnorm(n * p), ncol=p); X[] <- scale(X)
    y <- X %*% beta + rnorm(n)


    nlambda <- 10

    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             intercept=FALSE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14)

    for(i in seq_len(nlambda)){
        expect_equal(solve(crossprod(X)/n + elfit$lambda[i] * diag(1, p, p),
                           crossprod(X, y)/n),
                     as.matrix(elfit$coef[,i, drop=FALSE]),
                     check.attributes=FALSE)
    }

    ## High-Dimensional
    n <- 200
    p <- 500
    groups <- 1:p

    beta <- numeric(p); beta[1:5] <- 3

    X <- matrix(rnorm(n * p), ncol=p); X[] <- scale(X)
    y <- X %*% beta + rnorm(n)


    nlambda <- 10

    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             intercept=FALSE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14)

    for(i in seq_len(nlambda)){
        expect_equal(solve(crossprod(X)/n + elfit$lambda[i] * diag(1, p, p),
                           crossprod(X, y)/n),
                     as.matrix(elfit$coef[, i, drop=FALSE]),
                     check.attributes=FALSE)
    }
})

## TODO -- Add tests for KKT checks
##
## check_kkt <- function(X, y, groups, lambda){
##     ### Generalization of the prox KKT conditions
##
## }
##
## test_that("Satisfies KKT conditions in general case", {
##
## })

test_that("Matches closed form solution", {
    make_M <- function(groups, beta_hat){
        ### Assumes beta_hat is sorted by groups
        ### and that only exact solution is given
        M <- matrix(0, length(beta_hat), length(beta_hat))

        for(g in unique(groups)){
            g_init <- min(which(groups == g))
            g_end  <- max(which(groups == g))

            s <- sign(beta_hat)[g_init:g_end]

            M[g_init:g_end, g_init:g_end] <- s %o% s
        }
        M
    }

    set.seed(3003)

    ## Low-Dimensional
    n <- 200
    p <- 20
    g <- 5
    groups <- rep(1:g, each=p/g)

    beta <- rep(0, p)
    beta[seq(1, by=p/g, length.out=g)] <- 2

    X <- matrix(rnorm(n * p), ncol=p); X[] <- scale(X)
    y <- X %*% beta + rnorm(n)


    nlambda <- 20

    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             intercept=FALSE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14)

    for(i in seq_len(nlambda)){
        beta_hat <- elfit$coef[, i]
        supp <- which(beta_hat != 0)

        M <- make_M(groups[supp], beta_hat[supp])
        X_supp <- X[,supp]

        beta_analytic <- solve(crossprod(X_supp)/n + elfit$lambda[i] * M,
                               crossprod(X_supp, y)/n)

        expect_equal(as.vector(beta_analytic), beta_hat[supp])
    }

    ## High-Dimensional
    n <- 100
    p <- 200
    g <- 5
    groups <- rep(1:g, each=p/g)

    beta <- rep(0, p)
    beta[seq(1, by=p/g, length.out=g)] <- 2

    X <- matrix(rnorm(n * p), ncol=p); X[] <- scale(X)
    y <- X %*% beta + rnorm(n)

    nlambda <- 20

    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             intercept=FALSE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14)

    for(i in seq_len(nlambda)){
        beta_hat <- elfit$coef[, i]
        supp <- which(beta_hat != 0)

        M <- make_M(groups[supp], beta_hat[supp])
        X_supp <- X[,supp]

        beta_analytic <- solve(crossprod(X_supp)/n + elfit$lambda[i] * M,
                               crossprod(X_supp, y)/n)

        expect_equal(as.vector(beta_analytic), beta_hat[supp])
    }
})

test_that("Two algorithms give the same result",{
    set.seed(1559)
    n <- 200
    p <- 500

    g <- 50
    groups <- rep(1:p, length.out=p)

    ## Basic case
    X <- matrix(rnorm(n * p), ncol=p)
    beta <- numeric(p); beta[1:g] <- 2 * sample(c(-1, 1), g, replace=TRUE)

    y <- X %*% beta + rnorm(n)

    f1 <- exclusive_lasso(X, y, groups, algorithm="cd", thresh=1e-12)
    f2 <- exclusive_lasso(X, y, groups, algorithm="pg", thresh=1e-12, thresh_prox=1e-12)

    expect_equal(coef(f1), coef(f2))

    ## + Weights
    w <- runif(n, 1, 2); w <- n * w / sum(w)
    y <- X %*% beta + rnorm(n)

    f1 <- exclusive_lasso(X, y, groups, weights=w,
                          algorithm="cd", thresh=1e-12)
    f2 <- exclusive_lasso(X, y, groups, weights=w,
                          algorithm="pg", thresh=1e-12, thresh_prox=1e-12)

    expect_equal(coef(f1), coef(f2))

    ## + Offsets
    o <- runif(n, -0.5, 0.5)
    y <- X %*% beta + o + rnorm(n)

    f1 <- exclusive_lasso(X, y, groups, offset=o,
                          algorithm="cd", thresh=1e-12)
    f2 <- exclusive_lasso(X, y, groups, offset=o,
                          algorithm="pg", thresh=1e-12, thresh_prox=1e-12)

    expect_equal(coef(f1), coef(f2))

    ## Speed up test => use fewer lambda values
    ##
    ## We need to use very high precision to get the values to match,
    ## so check fewer lambdas to get this done in a reasonable amount of time
    f <- exclusive_lasso(X, y, groups, offset=o, weights=w, nlambda=10,
                         algorithm="cd", thresh=1e-14)
    f2 <- exclusive_lasso(X, y, groups, offset=o, weights=w, nlambda=10,
                         thresh=1e-14, thresh_prox=1e-14, algorithm="pg")
    expect_equal(coef(f), coef(f2))
})

test_that("Intercepts work", {
    set.seed(1945)
    n <- 100
    p <- 200

    g <- 50
    groups <- rep(1:p, length.out=p)

    ## Basic case
    X <- matrix(rnorm(n * p), ncol=p)
    beta <- numeric(p); beta[1:g] <- 2 * sample(c(-1, 1), g, replace=TRUE)

    y <- X %*% beta + rnorm(n)
    ym <- matrix(y, nrow=length(y), ncol=100, byrow=FALSE)

    f <- exclusive_lasso(X, y, groups, standardize=TRUE, thresh=1e-12)
    expect_equal(colMeans(ym - X %*% coef(f)[-1, ]), coef(f)[1,])
    expect_equal(colMeans(predict(f)), rep(mean(y), 100))

    f <- exclusive_lasso(X, y, groups, standardize=FALSE, thresh=1e-12)
    expect_equal(colMeans(ym - X %*% coef(f)[-1, ]), coef(f)[1,])
    expect_equal(colMeans(predict(f)), rep(mean(y), 100))

    ## + Weights
    w <- runif(n, 0.5, 1.5); w <- w * n / sum(w);

    f <- exclusive_lasso(X, y, groups, standardize=TRUE, weights=w, thresh=1e-12)
    expect_equal(apply(ym - X %*% coef(f)[-1, ], 2, weighted.mean, w), coef(f)[1,])
    expect_equal(apply(predict(f), 2, weighted.mean, w), rep(weighted.mean(y, w), 100))

    f <- exclusive_lasso(X, y, groups, standardize=FALSE, weights=w, thresh=1e-12)
    expect_equal(apply(ym - X %*% coef(f)[-1, ], 2, weighted.mean, w), coef(f)[1,])
    expect_equal(apply(predict(f), 2, weighted.mean, w), rep(weighted.mean(y, w), 100))

    ## + Offsets
    o <- runif(n, 1.5, 2.5);

    f <- exclusive_lasso(X, y, groups, standardize=TRUE, offset=o, thresh=1e-12)
    expect_equal(colMeans(ym - o - X %*% coef(f)[-1, ]), coef(f)[1,])
    expect_equal(colMeans(predict(f)), rep(mean(y), 100))

    f <- exclusive_lasso(X, y, groups, standardize=FALSE, offset=o, thresh=1e-12)
    expect_equal(colMeans(ym - o - X %*% coef(f)[-1, ]), coef(f)[1,])
    expect_equal(colMeans(predict(f)), rep(mean(y), 100))

    ## + Weights + Offsets
    f <- exclusive_lasso(X, y, groups, standardize=TRUE, offset=o, weights=w, thresh=1e-12)
    expect_equal(apply(ym - o - X %*% coef(f)[-1, ], 2, weighted.mean, w), coef(f)[1,])
    expect_equal(apply(predict(f), 2, weighted.mean, w), rep(weighted.mean(y, w), 100))

    f <- exclusive_lasso(X, y, groups, standardize=FALSE, offset=o, weights=w, thresh=1e-12)
    expect_equal(apply(ym - o - X %*% coef(f)[-1, ], 2, weighted.mean, w), coef(f)[1,])
    expect_equal(apply(predict(f), 2, weighted.mean, w), rep(weighted.mean(y, w), 100))

    ## Different distribution of X
    X <- matrix(rnorm(n * p, mean=2, sd=0.5), ncol=p)
    beta <- numeric(p); beta[1:g] <- 2 * sample(c(-1, 1), g, replace=TRUE)
    y <- X %*% beta + rnorm(n) + 1
    ym <- matrix(y, nrow=length(y), ncol=100, byrow=FALSE)

    f <- exclusive_lasso(X, y, groups, standardize=TRUE, offset=o, weights=w, thresh=1e-12)
    expect_equal(apply(ym - o - X %*% coef(f)[-1, ], 2, weighted.mean, w), coef(f)[1,])
    expect_equal(apply(predict(f), 2, weighted.mean, w), rep(weighted.mean(y, w), 100))

    ## This one has convergence problems, so only check 10 lambdas
    ym <- matrix(y, nrow=length(y), ncol=10, byrow=FALSE)
    f <- exclusive_lasso(X, y, groups, standardize=FALSE, nlambda=10, offset=o, weights=w, thresh=1e-8)
    expect_equal(apply(ym - o - X %*% coef(f)[-1, ], 2, weighted.mean, w), coef(f)[1,])
    expect_equal(apply(predict(f), 2, weighted.mean, w), rep(weighted.mean(y, w), 10))
})


test_that("Estimation works in low-dim + low-penalty case", {
    set.seed(672)
    n <- 500
    p <- 20

    groups <- rep(1:p, length.out=p)

    ## Basic case
    X <- matrix(rnorm(n * p, sd=3, mean=-1), ncol=p)
    beta <- rep(2, p)

    y <- X %*% beta + 3 ## No noise

    f <- exclusive_lasso(X, y, groups, thresh=1e-12, lambda=1e-14)
    expect_equal(coef(f)[1,,drop=TRUE], 3, check.attributes=FALSE)
    expect_equal(coef(f)[-1,,drop=TRUE], beta, check.attributes=FALSE)

    ## + Weights
    w <- runif(n, 2, 3); w <- w / sum(w) * n
    f <- exclusive_lasso(X, y, groups, weights=w, thresh=1e-12, lambda=1e-14)
    expect_equal(coef(f)[1,,drop=TRUE], 3, check.attributes=FALSE)
    expect_equal(coef(f)[-1,,drop=TRUE], beta, check.attributes=FALSE)
})
