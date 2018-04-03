context("Logistic GLM works")

test_that("Input validation works", {
    set.seed(123)
    ## Good inputs
    n <- 200
    p <- 500
    g <- 10

    X <- matrix(rnorm(n * p), ncol=p)
    y <- runif(n)
    groups <- rep(1:g, length.out=p)

    weights <- runif(n, 0.5, 1.5); weights <- weights / sum(weights) * n;
    offset <- runif(n, -0.5, 0.5);

    lambda <- seq(0.5, 5, length.out=100)
    thresh <- 1e-2; thresh_prox <- 1e-2 ## Low values here speed up tests

    expect_silent(exclusive_lasso(X, y, groups=groups,
                                  weights=weights, offset=offset,
                                  lambda=lambda, family="binomial", algorithm="pg",
                                  thresh=thresh, thresh_prox=thresh_prox))

    ## Wrong domain for y
    expect_error(exclusive_lasso(X, y  + 3, groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="binomial", algorithm="pg",
                                 thresh=thresh, thresh_prox=thresh_prox))

    expect_error(exclusive_lasso(X, y  - 3, groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="binomial", algorithm="pg",
                                 thresh=thresh, thresh_prox=thresh_prox))

    ## X, y match
    expect_error(exclusive_lasso(X, rep(y, 2), groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="binomial", algorithm="pg",
                                 thresh=thresh, thresh_prox=thresh_prox))

    expect_error(exclusive_lasso(X, rep(y, length.out=n-1), groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="binomial", algorithm="pg",
                                 thresh=thresh, thresh_prox=thresh_prox))

    ## groups check
    expect_error(exclusive_lasso(X, y,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="binomial", algorithm="pg",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=rep(groups, 2),
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="binomial", algorithm="pg",
                                 thresh=thresh, thresh_prox=thresh_prox))

    ## Weights check
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=rep(weights, 3), offset=offset,
                                 lambda=lambda, family="binomial", algorithm="pg",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights= -1 * weights, offset=offset,
                                 lambda=lambda, family="binomial", algorithm="pg",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights= 0 * weights, offset=offset,
                                 lambda=lambda, family="binomial", algorithm="pg",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_warning(exclusive_lasso(X, y, groups=groups,
                                   weights= 2 * weights, offset=offset,
                                   lambda=lambda, family="binomial", algorithm="pg",
                                   thresh=thresh, thresh_prox=thresh_prox))

    ## Offsets check
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=rep(offset, 2),
                                 lambda=lambda, family="binomial", algorithm="pg",
                                 thresh=thresh, thresh_prox=thresh_prox))

    ## Convergence thresholds check
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="binomial", algorithm="pg",
                                 thresh=-1 * thresh, thresh_prox=thresh_prox))

    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="binomial", algorithm="pg",
                                 thresh=thresh, thresh_prox=-1 * thresh_prox))

    ## Lambda check
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 nlambda=-30, family="binomial", algorithm="pg",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 nlambda=0, family="binomial", algorithm="pg",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=-lambda, family="binomial", algorithm="pg",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 lambda.min.ratio=2, family="binomial", algorithm="pg",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_warning(exclusive_lasso(X, y, groups=groups,
                                   weights=weights, offset=offset,
                                   lambda=rev(lambda), family="binomial", algorithm="pg",
                                   thresh=thresh, thresh_prox=thresh_prox))

})

test_that("Dynamic defaults work", {
    set.seed(100)

    n <- 200
    p <- 500
    g <- 10

    X <- matrix(rnorm(n * p), ncol=p)
    y <- runif(n)
    groups <- rep(1:g, length.out=p)

    elfit <- exclusive_lasso(X, y, groups,
                             family="binomial", algorithm="pg",
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
    y <- runif(n)
    groups <- rep(1:g, length.out=p)

    elfit <- exclusive_lasso(X, y, groups,
                             family="binomial",
                             algorithm="pg",
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
    y <- plogis(X %*% beta + rnorm(n))
    groups <- rep(1:g, length.out=p)

    X_sc <- X; X_sc[] <- scale(X_sc)

    elfit <- exclusive_lasso(X, y, groups,
                             family="binomial",
                             algorithm="pg",
                             thresh_prox=1e-8,
                             thresh=1e-8)

    elfit_sc <- exclusive_lasso(X_sc, y, groups,
                                lambda=elfit$lambda,
                                family="binomial",
                                algorithm="pg",
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

test_that("GLM PG works correctly", {
    ## Compare to Gaussian case
    set.seed(45)
    n <- 200
    p <- 20

    groups <- rep(1:4, 5)

    X <- matrix(rnorm(n * p), ncol=p)
    beta <- rep(0, p); beta[1:4] <- 3
    y <- X %*% beta + rnorm(n)

    options(ExclusiveLasso.gaussian_fast_path = TRUE)

    fit1 <- exclusive_lasso(X, y, groups, algorithm="pg",
                            thresh=1e-10, thresh_prox=1e-10)
    fit2 <- exclusive_lasso(X, y, groups, algorithm="cd",
                            thresh=1e-10, thresh_prox=1e-10)

    options(ExclusiveLasso.gaussian_fast_path = FALSE)

    fit3 <- exclusive_lasso(X, y, groups, algorithm="pg",
                            thresh=1e-10, thresh_prox=1e-10)

    ## Reset this
    options(Exclusive_lasso.gaussian_fast_path = TRUE)

    expect_equal(coef(fit1), coef(fit2))
    expect_equal(coef(fit1), coef(fit3))
})
