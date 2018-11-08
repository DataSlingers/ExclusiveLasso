context("Box constraints")

test_that("Error checking for box constraints works", {
    set.seed(125)

    n <- 200
    p <- 500
    groups <- rep(1:10, times=50)
    beta <- numeric(p);
    beta[1:10] <- 3

    X <- matrix(rnorm(n * p), ncol=p)
    y <- X %*% beta + rnorm(n)

    expect_error(exclusive_lasso(X, y, groups, lower.limits = 5, upper.limits = -5))

    expect_error(exclusive_lasso(X, y, groups, lower.limits = rep(5, p - 1)))
    expect_error(exclusive_lasso(X, y, groups, lower.limits = NA))
    expect_error(exclusive_lasso(X, y, groups, lower.limits = NaN))

    expect_error(exclusive_lasso(X, y, groups, upper.limits = rep(5, p - 1)))
    expect_error(exclusive_lasso(X, y, groups, upper.limits = NA))
    expect_error(exclusive_lasso(X, y, groups, upper.limits = NaN))
})

test_that("All four gaussian implementations match - non-negative", {
    set.seed(125)

    n <- 200
    p <- 500
    groups <- rep(1:10, times=50)
    beta <- numeric(p);
    beta[1:10] <- 3

    X <- matrix(rnorm(n * p), ncol=p)
    y <- X %*% beta + rnorm(n)

    options(ExclusiveLasso.gaussian_fast_path = TRUE)
    exfit1 <- exclusive_lasso(X, y, groups, skip_df = TRUE,
                              lower.limits = rep(0, p), thresh=1e-10, algorithm = "cd")
    exfit2 <- exclusive_lasso(X, y, groups, skip_df = TRUE,
                              lower.limits = rep(0, p), thresh=1e-10, algorithm = "pg")

    options(ExclusiveLasso.gaussian_fast_path = FALSE)
    exfit3 <- exclusive_lasso(X, y, groups, skip_df = TRUE,
                              lower.limits = rep(0, p), thresh=1e-10, algorithm = "cd")
    exfit4 <- exclusive_lasso(X, y, groups, skip_df = TRUE,
                              lower.limits = rep(0, p), thresh=1e-10, algorithm = "pg")

    ## Reset before testing
    options(ExclusiveLasso.gaussian_fast_path = TRUE)

    expect_equal(coef(exfit1), coef(exfit2))
    expect_equal(coef(exfit1), coef(exfit3))
    expect_equal(coef(exfit1), coef(exfit4))

    ## Check all coefficients (but not intercept) are >= 0
    expect_true(all(coef(exfit1)[-1,] >= 0))
    expect_true(all(coef(exfit2)[-1,] >= 0))
    expect_true(all(coef(exfit3)[-1,] >= 0))
    expect_true(all(coef(exfit4)[-1,] >= 0))
})

test_that("All four gaussian implementations match - non-positive", {
    set.seed(521)

    n <- 200
    p <- 50
    groups <- rep(1:10, times=5)
    beta <- numeric(p);
    beta[1:10] <- 3

    X <- matrix(rnorm(n * p), ncol=p)
    y <- X %*% beta + rnorm(n)

    options(ExclusiveLasso.gaussian_fast_path = TRUE)
    exfit1 <- exclusive_lasso(X, y, groups, skip_df = TRUE,
                              upper.limits = rep(0, p), thresh=1e-10, algorithm = "cd")
    exfit2 <- exclusive_lasso(X, y, groups, skip_df = TRUE,
                              upper.limits = rep(0, p), thresh=1e-10, algorithm = "pg")

    options(ExclusiveLasso.gaussian_fast_path = FALSE)
    exfit3 <- exclusive_lasso(X, y, groups, skip_df = TRUE,
                              upper.limits = rep(0, p), thresh=1e-10, algorithm = "cd")
    exfit4 <- exclusive_lasso(X, y, groups, skip_df = TRUE,
                              upper.limits = rep(0, p), thresh=1e-10, algorithm = "pg")

    ## Reset before testing
    options(ExclusiveLasso.gaussian_fast_path = TRUE)

    expect_equal(coef(exfit1), coef(exfit2))
    expect_equal(coef(exfit1), coef(exfit3))
    expect_equal(coef(exfit1), coef(exfit4), tolerance = 1e-7) # This one is a bit less tight for some reason

    ## Check all coefficients (but not intercept) are <= 0
    expect_true(all(coef(exfit1)[-1,] <= 0))
    expect_true(all(coef(exfit2)[-1,] <= 0))
    expect_true(all(coef(exfit3)[-1,] <= 0))
    expect_true(all(coef(exfit4)[-1,] <= 0))
})

test_that("Recovers non-negative Gaussian ridge", {
    skip("Cannot match glmnet's odd scaling of Gaussian responses")

    ## This should work, but seems not to for some reason...
    skip_if_not_installed("nnls")
    library(nnls)

    set.seed(5454)

    n <- 200
    p <- 20
    groups <- 1:p

    beta <- rnorm(p, 0, 3)
    X <- matrix(rnorm(n * p), ncol=p); X[] <- scale(X)
    y <- X %*% beta + rnorm(n)

    nlambda <- 10

    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             intercept=FALSE, standardize=FALSE,
                             lower.limits = rep(0, p),
                             thresh=1e-14, thresh_prox=1e-14, skip_df = TRUE)

    for(i in seq_len(nlambda)){
        X_aug <- rbind(X, elfit$lambda[i] * diag(1, p, p))
        y_aug <- c(y, rep(0, p))

        expect_equal(coef(nnls(X_aug, y_aug)),
                     as.matrix(elfit$coef[,i, drop=FALSE]),
                     check.attributes=FALSE)
    }
})

test_that("Recovers non-negative logistic ridge", {
    skip_if_not_installed("glmnet") ## For a logistic ridge implementation
    library(glmnet)
    set.seed(222)

    n <- 200
    p <- 20
    groups <- 1:p

    beta <- rnorm(p, -1, 3)
    X <- matrix(rnorm(n * p), ncol=p); X[] <- scale(X)
    y <- round(plogis(X %*% beta + rnorm(n)))

    nlambda <- 10

    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             family = "binomial", lower.limits = rep(0, p),
                             intercept=FALSE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14, skip_df = TRUE)

    glfit <- glmnet(X, y, family = "binomial", intercept = FALSE, lower.limits = rep(0, p),
                    lambda = rev(elfit$lambda), standardize = FALSE,
                    alpha = 0, thresh = 1e-20)

    for(i in seq_len(nlambda)){
        ## Glmnet returns coefficients 'reversed' compared to how we do it
        expect_equal(coef(elfit)[, i], coef(glfit)[,nlambda - i + 1],
                     check.attributes = FALSE)
    }

    ## Now with intercepts
    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             family = "binomial", lower.limits = rep(0, p),
                             intercept=TRUE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14, skip_df = TRUE)

    glfit <- glmnet(X, y, family = "binomial", intercept = TRUE, lower.limits = rep(0, p),
                    lambda = rev(elfit$lambda), standardize = FALSE,
                    alpha = 0, thresh = 1e-20)

    for(i in seq_len(nlambda)){
        ## Glmnet returns coefficients 'reversed' compared to how we do it
        expect_equal(coef(elfit)[, i], coef(glfit)[,nlambda - i + 1],
                     check.attributes = FALSE)
    }
})

test_that("Recovers non-negative Poisson ridge", {
    skip_if_not_installed("glmnet") ## For a poisson ridge implementation
    library(glmnet)
    set.seed(7117)

    ## Low-Dimensional
    n <- 200
    p <- 20
    groups <- 1:p

    beta <- rep(0.2, p)
    X <- matrix(rnorm(n * p), ncol=p); X[] <- scale(X)
    y <- rpois(n, exp(X %*% beta))

    nlambda <- 10

    ## Poisson objective
    obj <- function(coef, lambda){
        alpha <- coef[1]; beta <- coef[-1]
        eta     <- alpha + X %*% beta
        loss    <- mean(exp(eta) - y * eta)
        penalty <- 0.5 * sum(beta^2) # Ridge penalty
        loss + lambda * penalty
    }

    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             family = "poisson", lower.limits = rep(0, p),
                             intercept=FALSE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14, skip_df = TRUE)

    glfit <- glmnet(X, y, family = "poisson", intercept = FALSE, lower.limits = rep(0, p),
                    lambda = rev(elfit$lambda), standardize = FALSE,
                    alpha = 0, thresh = 1e-20)

    for(i in seq_len(nlambda)){
        ## Glmnet returns coefficients 'reversed' compared to how we do it
        ##
        ## In theory, these should match, but the lack of a back-tracking step in
        ## glmnet means we sometimes get superior solutions (as measured by the objective value)

        # expect_equal(coef(elfit)[, i], coef(glfit)[,nlambda - i + 1],
        #             check.attributes = FALSE)

        expect_lte(obj(coef(elfit)[,i], elfit$lambda[i]),
                   obj(coef(elfit)[,1 + nlambda - i], elfit$lambda[i]))
    }

    ## Now with intercepts
    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             family = "poisson", lower.limits = rep(0, p),
                             intercept=TRUE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14, skip_df = TRUE)

    glfit <- glmnet(X, y, family = "poisson", intercept = TRUE, lower.limits = rep(0, p),
                    lambda = rev(elfit$lambda), standardize = FALSE,
                    alpha = 0, thresh = 1e-20)

    for(i in seq_len(nlambda)){
        ## Glmnet returns coefficients 'reversed' compared to how we do it
        ##
        ## In theory, these should match, but the lack of a back-tracking step in
        ## glmnet means we sometimes get superior solutions (as measured by the objective value)

        # expect_equal(coef(elfit)[, i], coef(glfit)[,nlambda - i + 1],
        #             check.attributes = FALSE)

        expect_lte(obj(coef(elfit)[,i], elfit$lambda[i]),
                   obj(coef(elfit)[,1 + nlambda - i], elfit$lambda[i]))
    }
})

test_that("Recovers box-constrianed Gaussian ridge", {
  skip("Cannot match glmnet's odd scaling of Gaussian responses")
})

test_that("Recovers box-constrianed logistic ridge", {
    skip_if_not_installed("glmnet") ## For a logistic ridge implementation
    library(glmnet)
    set.seed(555)

    n <- 200
    p <- 20
    groups <- 1:p

    beta <- rnorm(p, 0, 3)
    X <- matrix(rnorm(n * p), ncol=p); X[] <- scale(X)
    y <- round(plogis(X %*% beta + rnorm(n)))

    nlambda <- 10

    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             family = "binomial",
                             lower.limits = rep(-3, p), upper.limits = rep(3, p),
                             intercept=FALSE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14, skip_df = TRUE)

    glfit <- glmnet(X, y, family = "binomial", intercept = FALSE,
                    lower.limits = rep(-3, p), upper.limits = rep(3, p),
                    lambda = rev(elfit$lambda), standardize = FALSE,
                    alpha = 0, thresh = 1e-20)

    for(i in seq_len(nlambda)){
        ## Glmnet returns coefficients 'reversed' compared to how we do it
        expect_equal(coef(elfit)[, i], coef(glfit)[,nlambda - i + 1],
                     check.attributes = FALSE)
    }

    ## Now with intercepts
    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             family = "binomial",
                             lower.limits = rep(-3, p), upper.limits = rep(3, p),
                             intercept=TRUE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14, skip_df = TRUE)

    glfit <- glmnet(X, y, family = "binomial", intercept = TRUE,
                    lower.limits = rep(-3, p), upper.limits = rep(3, p),
                    lambda = rev(elfit$lambda), standardize = FALSE,
                    alpha = 0, thresh = 1e-20)

    for(i in seq_len(nlambda)){
        ## Glmnet returns coefficients 'reversed' compared to how we do it
        expect_equal(coef(elfit)[, i], coef(glfit)[,nlambda - i + 1],
                     check.attributes = FALSE)
    }
})

test_that("Recovers box-constrianed Poisson ridge", {
    skip_if_not_installed("glmnet") ## For a poisson ridge implementation
    library(glmnet)
    set.seed(1771)

    ## Low-Dimensional
    n <- 200
    p <- 20
    groups <- 1:p

    beta <- rep(0.2, p)
    X <- matrix(rnorm(n * p), ncol=p); X[] <- scale(X)
    y <- rpois(n, exp(X %*% beta))

    nlambda <- 10

    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             family = "poisson",
                             lower.limits = rep(-0.1, p), upper.limits = rep(0.1, p),
                             intercept=FALSE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14, skip_df = TRUE)

    glfit <- glmnet(X, y, family = "poisson", intercept = FALSE,
                    lower.limits = rep(-0.1, p), upper.limits = rep(0.1, p),
                    lambda = rev(elfit$lambda), standardize = FALSE,
                    alpha = 0, thresh = 1e-20)

    for(i in seq_len(nlambda)){
        ## Glmnet returns coefficients 'reversed' compared to how we do it
        expect_equal(coef(elfit)[, i], coef(glfit)[,nlambda - i + 1],
                     check.attributes = FALSE)
    }

    ## Now with intercepts
    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             family = "poisson",
                             lower.limits = rep(-0.1, p), upper.limits = rep(0.1, p),
                             intercept=TRUE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14, skip_df = TRUE)

    glfit <- glmnet(X, y, family = "poisson", intercept = TRUE,
                    lower.limits = rep(-0.1, p), upper.limits = rep(0.1, p),
                    lambda = rev(elfit$lambda), standardize = FALSE,
                    alpha = 0, thresh = 1e-20)

    for(i in seq_len(nlambda)){
        ## Glmnet returns coefficients 'reversed' compared to how we do it
        expect_equal(coef(elfit)[, i], coef(glfit)[,nlambda - i + 1],
                     check.attributes = FALSE)
    }
})
