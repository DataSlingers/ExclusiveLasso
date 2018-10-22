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
                                  lambda=lambda, family="binomial",
                                  thresh=thresh, thresh_prox=thresh_prox))

    ## Wrong domain for y
    expect_error(exclusive_lasso(X, y  + 3, groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="binomial",
                                 thresh=thresh, thresh_prox=thresh_prox))

    expect_error(exclusive_lasso(X, y  - 3, groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="binomial",
                                 thresh=thresh, thresh_prox=thresh_prox))

    ## X, y match
    expect_error(exclusive_lasso(X, rep(y, 2), groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="binomial",
                                 thresh=thresh, thresh_prox=thresh_prox))

    expect_error(exclusive_lasso(X, rep(y, length.out=n-1), groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="binomial",
                                 thresh=thresh, thresh_prox=thresh_prox))

    ## groups check
    expect_error(exclusive_lasso(X, y,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="binomial",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=rep(groups, 2),
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="binomial",
                                 thresh=thresh, thresh_prox=thresh_prox))

    ## Weights check
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=rep(weights, 3), offset=offset,
                                 lambda=lambda, family="binomial",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights= -1 * weights, offset=offset,
                                 lambda=lambda, family="binomial",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights= 0 * weights, offset=offset,
                                 lambda=lambda, family="binomial",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_warning(exclusive_lasso(X, y, groups=groups,
                                   weights= 2 * weights, offset=offset,
                                   lambda=lambda, family="binomial",
                                   thresh=thresh, thresh_prox=thresh_prox))

    ## Offsets check
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=rep(offset, 2),
                                 lambda=lambda, family="binomial",
                                 thresh=thresh, thresh_prox=thresh_prox))

    ## Convergence thresholds check
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="binomial",
                                 thresh=-1 * thresh, thresh_prox=thresh_prox))

    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=lambda, family="binomial",
                                 thresh=thresh, thresh_prox=-1 * thresh_prox))

    ## Lambda check
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 nlambda=-30, family="binomial",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 nlambda=0, family="binomial",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 lambda=-lambda, family="binomial",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_error(exclusive_lasso(X, y, groups=groups,
                                 weights=weights, offset=offset,
                                 lambda.min.ratio=2, family="binomial",
                                 thresh=thresh, thresh_prox=thresh_prox))
    expect_warning(exclusive_lasso(X, y, groups=groups,
                                   weights=weights, offset=offset,
                                   lambda=rev(lambda), family="binomial",
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
                             family="binomial",
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
                             thresh_prox=1e-8,
                             thresh=1e-8)

    elfit_sc <- exclusive_lasso(X_sc, y, groups,
                                lambda=elfit$lambda,
                                family="binomial",
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


test_that("Logistic returns ridge with trivial group structure", {
    skip_if_not_installed("glmnet") ## For a logistic ridge implementation
    library(glmnet)
    set.seed(200)

    ## Low-Dimensional
    n <- 200
    p <- 20
    groups <- 1:p

    beta <- rep(1, p)
    X <- matrix(rnorm(n * p), ncol=p); X[] <- scale(X)
    y <- round(plogis(X %*% beta + rnorm(n)))

    nlambda <- 10

    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             family = "binomial",
                             intercept=FALSE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14)

    glfit <- glmnet(X, y, family = "binomial", intercept = FALSE,
                    lambda = rev(elfit$lambda), standardize = FALSE,
                    alpha = 0, thresh = 1e-20)

    for(i in seq_len(nlambda)){
        ## Glmnet returns coefficients 'reversed' compared to how we do it
        expect_equal(coef(elfit)[, i], coef(glfit)[,nlambda - i + 1],
                     check.attributes = FALSE)
    }

    ## High-Dimensional
    n <- 200
    p <- 500
    groups <- 1:p

    beta <- numeric(p); beta[1:5] <- c(-3, 3, 1, -2, 1)

    X <- matrix(rnorm(n * p), ncol=p); X[] <- scale(X)
    y <- round(plogis(X %*% beta + rnorm(n)))

    nlambda <- 10

    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             family = "binomial",
                             intercept=FALSE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14)

    glfit <- glmnet(X, y, family = "binomial", intercept = FALSE,
                    lambda = rev(elfit$lambda), standardize = FALSE,
                    alpha = 0, thresh = 1e-20)

    for(i in seq_len(nlambda)){
        ## Glmnet returns coefficients 'reversed' compared to how we do it
        expect_equal(coef(elfit)[, i], coef(glfit)[,nlambda - i + 1], check.attributes = FALSE)
    }
})

test_that("CD and PG solvers get the same result for Logistic GLMs", {
    set.seed(55)
    n <- 400
    p <- 40

    groups <- rep(1:4, length.out=p)

    X <- matrix(rnorm(n * p), ncol=p); X[] <- scale(X)
    beta <- rep(0, p); beta[1:4] <- 5
    y <- plogis(X %*% beta)

    fit1 <- exclusive_lasso(X, y, groups, algorithm="cd", family="binomial",
                            thresh=1e-14, thresh_prox=1e-14, intercept=TRUE)
    fit2 <- exclusive_lasso(X, y, groups, algorithm="pg", family="binomial",
                            thresh=1e-14, thresh_prox=1e-14, intercept=TRUE)

    ## Be a bit looser here than strictly necessary since we are using inexact PG
    expect_equal(coef(fit1), coef(fit2), tolerance = 1e-6)
})
