context("Gaussian GLM works")
## Tests that the Gaussian-specific and the GLM-impelementation + Gaussian family
## get the same answer
##
## This gives us some additional assurance that the GLM implementation is accurate

test_that("Gaussian GLM works with coordinate descent", {
    set.seed(545)
    n <- 100
    p <- 40

    groups <- rep(1:4, length.out=p)

    X <- matrix(rnorm(n * p), ncol=p)
    beta <- rep(0, p); beta[1:4] <- 3
    y <- X %*% beta + rnorm(n)

    options(ExclusiveLasso.gaussian_fast_path = TRUE)
    fit1 <- exclusive_lasso(X, y, groups, algorithm="cd",
                            thresh=1e-12, thresh_prox=1e-12)

    options(ExclusiveLasso.gaussian_fast_path = FALSE)
    fit2 <- exclusive_lasso(X, y, groups, algorithm="cd",
                            thresh=1e-12, thresh_prox=1e-12)

    ## Reset this
    options(ExclusiveLasso.gaussian_fast_path = TRUE)

    expect_equal(coef(fit1), coef(fit2))
})

test_that("Gaussian GLM works with proximal gradient", {
    set.seed(454)
    n <- 100
    p <- 40

    groups <- rep(1:4, length.out=p)

    X <- matrix(rnorm(n * p), ncol=p)
    beta <- rep(0, p); beta[1:4] <- 3
    y <- X %*% beta + rnorm(n)

    options(ExclusiveLasso.gaussian_fast_path = TRUE)
    fit1 <- exclusive_lasso(X, y, groups, algorithm="pg",
                            thresh=1e-14, thresh_prox=1e-14)

    options(ExclusiveLasso.gaussian_fast_path = FALSE)
    fit2 <- exclusive_lasso(X, y, groups, algorithm="pg",
                            thresh=1e-14, thresh_prox=1e-14)

    ## Reset this
    options(ExclusiveLasso.gaussian_fast_path = TRUE)

    ## Need a slightly looser tolerance here since inexact PG is not super accurate
    expect_equal(coef(fit1), coef(fit2), tolerance = 5e-8)
})
