context("Helper functions work")

test_that("coef() works with original lambda", {
    set.seed(5454)

    n <- 200
    p <- 500
    groups <- rep(1, 10, length.out=p)

    beta <- numeric(p); beta[1:max(groups)] <- 2
    X <- matrix(rnorm(n * p), ncol=p)
    y <- X %*% beta + rnorm(n)

    nlambda <- 10

    ## With intercept
    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             intercept=TRUE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14)

    c_elfit <- coef(elfit)
    expect_equal(NROW(c_elfit), p + 1)
    expect_equal(NCOL(c_elfit), nlambda)

    expect_equal(c_elfit[1,,drop=TRUE], elfit$intercept)
    expect_equal(c_elfit[-1,,drop=FALSE], elfit$coef,
                 check.attributes=FALSE)

    ## Repeat without intercept
    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             intercept=FALSE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14)

    c_elfit <- coef(elfit)
    expect_equal(NROW(c_elfit), p + 1)
    expect_equal(NCOL(c_elfit), nlambda)

    expect_equal(c_elfit[1,,drop=TRUE], rep(0, nlambda))
    expect_equal(c_elfit[-1,,drop=FALSE], elfit$coef,
                 check.attributes=FALSE)
})

test_that("coef() works with new lambda (exact=FALSE)", {
    set.seed(4114)

    n <- 200
    p <- 500
    groups <- rep(1, 10, length.out=p)

    beta <- numeric(p); beta[1:max(groups)] <- 2
    X <- matrix(rnorm(n * p), ncol=p)
    y <- X %*% beta + rnorm(n)

    nlambda <- 30

    ## With intercept
    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             intercept=TRUE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14)

    ### Truly new value
    c_elfit <- coef(elfit, lambda=mean(elfit$lambda[23:24]))

    expect_true(all(abs(elfit$coef[, 23]) >= abs(c_elfit[-1])))
    expect_true(all(abs(elfit$coef[, 24]) <= abs(c_elfit[-1])))

    ### Old min endpoint
    c_elfit <- coef(elfit, lambda=min(elfit$lambda))
    expect_equal(c_elfit[-1], elfit$coef[,1], check.attributes=FALSE)

    ### Old max endpoint
    c_elfit <- coef(elfit, lambda=max(elfit$lambda))
    expect_equal(c_elfit[-1], elfit$coef[,nlambda], check.attributes=FALSE)

    ### Exact match to old internal value
    c_elfit <- coef(elfit, lambda=elfit$lambda[15])
    expect_equal(c_elfit[-1], elfit$coef[,15], check.attributes=FALSE)
})

test_that("coef() works with new lambda (exact=TRUE)", {
    set.seed(6772)

    n <- 200
    p <- 500
    groups <- rep(1, 10, length.out=p)

    beta <- numeric(p); beta[1:max(groups)] <- 2
    X <- matrix(rnorm(n * p), ncol=p)
    y <- X %*% beta + rnorm(n)

    nlambda <- 30

    ## Without intercept
    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             intercept=FALSE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14)

    ### Truly new value
    c_elfit <- coef(elfit, lambda=mean(elfit$lambda[23:24]),
                    exact=TRUE)

    expect_true(all(abs(elfit$coef[, 23]) >= abs(c_elfit[-1])))
    expect_true(all(abs(elfit$coef[, 24]) <= abs(c_elfit[-1])))

    ### Need to refit with high precision to match original fit

    ### Old min endpoint
    c_elfit <- coef(elfit, lambda=min(elfit$lambda),
                    exact=TRUE, thresh=1e-14, thresh_prox=1e-14)
    expect_equal(c_elfit[-1], elfit$coef[,1], check.attributes=FALSE)

    ### Old max endpoint
    c_elfit <- coef(elfit, lambda=max(elfit$lambda),
                    exact=TRUE, thresh=1e-14, thresh_prox=1e-14)
    expect_equal(c_elfit[-1], elfit$coef[,nlambda], check.attributes=FALSE)

    ### Exact match to old internal value
    c_elfit <- coef(elfit, lambda=elfit$lambda[15],
                    exact=TRUE, thresh=1e-14, thresh_prox=1e-14)
    expect_equal(c_elfit[-1], elfit$coef[,15], check.attributes=FALSE)

})

test_that("predict() works -- training data", {
    set.seed(8119)

    n <- 200
    p <- 500
    groups <- rep(1, 50, length.out=p)

    beta <- numeric(p); beta[1:max(groups)] <- 2
    X <- matrix(rnorm(n * p), ncol=p)
    y <- X %*% beta + rnorm(n)

    nlambda <- 30

    ## With intercept
    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             intercept=TRUE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14)

    expect_equal(predict(elfit, lambda=1),
                 cbind(1, X) %*% as.matrix(coef(elfit, lambda=1)),
                 check.attributes=FALSE)

    expect_equal(predict(elfit),
                 cbind(1, X) %*% as.matrix(coef(elfit)),
                 check.attributes=FALSE)

    expect_equal(predict(elfit, type="response"), ## Doesn't change for Gaussian
                 cbind(1, X) %*% as.matrix(coef(elfit)),
                 check.attributes=FALSE)

    ## Without intercept + with offset
    o <- runif(n, 0.5, 1.5)
    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             intercept=FALSE, offset=o,
                             thresh=1e-14, thresh_prox=1e-14)

    expect_equal(predict(elfit, lambda=1),
                 X %*% as.matrix(coef(elfit, lambda=1))[-1,] + o,
                 check.attributes=FALSE)

    expect_equal(predict(elfit),
                 X %*% as.matrix(coef(elfit))[-1,] + o,
                 check.attributes=FALSE)

    expect_equal(predict(elfit, type="response"), ## Doesn't change for Gaussian
                 X %*% as.matrix(coef(elfit))[-1,] + o,
                 check.attributes=FALSE)
})

test_that("predict() works -- test data", {
    set.seed(8119)

    n <- 200
    p <- 500
    groups <- rep(1, 50, length.out=p)

    beta <- numeric(p); beta[1:max(groups)] <- 2
    X <- matrix(rnorm(n * p), ncol=p); X2 <- matrix(rnorm(n * p), ncol=p)
    y <- X %*% beta + rnorm(n)

    nlambda <- 30

    ## With intercept
    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             intercept=TRUE, standardize=FALSE,
                             thresh=1e-14, thresh_prox=1e-14)

    expect_equal(predict(elfit, newx=X2, lambda=1),
                 cbind(1, X2) %*% as.matrix(coef(elfit, lambda=1)),
                 check.attributes=FALSE)

    expect_equal(predict(elfit, newx=X2),
                 cbind(1, X2) %*% as.matrix(coef(elfit)),
                 check.attributes=FALSE)

    expect_equal(predict(elfit, newx=X2, type="response"), ## Doesn't change for Gaussian
                 cbind(1, X2) %*% as.matrix(coef(elfit)),
                 check.attributes=FALSE)

    ## Without intercept + with offset
    o <- runif(n, 0.5, 1.5)
    elfit <- exclusive_lasso(X, y, groups, nlambda=nlambda,
                             intercept=FALSE, offset=o,
                             thresh=1e-14, thresh_prox=1e-14)

    expect_equal(predict(elfit, lambda=1, newx=X2, offset=o),
                 X2 %*% as.matrix(coef(elfit, lambda=1))[-1,] + o,
                 check.attributes=FALSE)

    expect_equal(predict(elfit, newx=X2, offset=o),
                 X2 %*% as.matrix(coef(elfit))[-1,] + o,
                 check.attributes=FALSE)

    expect_equal(predict(elfit, newx=X2, offset=o, type="response"), ## Doesn't change for Gaussian
                 X2 %*% as.matrix(coef(elfit))[-1,] + o,
                 check.attributes=FALSE)

    expect_error(predict(elfit, newx=X2)) ## Offset required
})
