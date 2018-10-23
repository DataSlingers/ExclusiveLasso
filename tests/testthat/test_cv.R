context("Cross-Validation")

test_that("CV works in default mode", {
    skip_on_cran()

    set.seed(125)
    n <- 200
    p <- 500
    groups <- rep(1:10, times=50)
    beta <- numeric(p);
    beta[1:10] <- 3

    X <- matrix(rnorm(n * p), ncol=p)
    y <- X %*% beta + rnorm(n)

    exfit_cv <- cv.exclusive_lasso(X, y, groups, nfolds=5)

    ## This is an easy problem, so we should get the right answer
    expect_equal(which(beta != 0),
                 which(coef(exfit_cv, lambda = "lambda.1se", group_threshold = TRUE)[-1] != 0))
})

test_that("CV checks loss type fits family", {
    set.seed(125)
    n <- 20
    p <- 50
    groups <- rep(1:10, times=5)
    beta <- numeric(p);
    beta[1:10] <- 3 * c(-1, 1)

    X <- matrix(rnorm(n * p), ncol=p)
    y <- plogis(X %*% beta + rnorm(n))

    ## AUC & Misclassificaiton error are only defined for binomials with 0/1 response
    expect_error(cv.exclusive_lasso(X, y, groups, type.measure = "class", nfolds=5))
    expect_error(cv.exclusive_lasso(X, round(y), groups, type.measure = "class", nfolds=5))
    expect_error(cv.exclusive_lasso(X, y, family = "binomial", groups, type.measure = "class", nfolds=5))

    expect_error(cv.exclusive_lasso(X, y, groups, type.measure = "auc", nfolds=5))
    expect_error(cv.exclusive_lasso(X, round(y), groups, type.measure = "auc", nfolds=5))
    expect_error(cv.exclusive_lasso(X, y, family = "binomial", groups, type.measure = "auc", nfolds=5))
})

test_that("predict() and coef() work with cv selected lambda", {
    skip_on_cran()

    set.seed(50)
    n <- 20
    p <- 50
    groups <- rep(1:10, times=5)
    beta <- numeric(p);
    beta[1:10] <- 3

    X <- matrix(rnorm(n * p), ncol=p)
    y <- X %*% beta + rnorm(n)

    exfit_cv <- cv.exclusive_lasso(X, y, groups, nfolds=5)

    expect_equal(coef(exfit_cv, lambda = "lambda.min"), coef(exfit_cv$fit, lambda = exfit_cv$lambda.min))
    expect_equal(coef(exfit_cv, lambda = "lambda.1se"), coef(exfit_cv$fit, lambda = exfit_cv$lambda.1se))
    expect_equal(coef(exfit_cv, s = "lambda.min"), coef(exfit_cv$fit, s = exfit_cv$lambda.min))
    expect_equal(coef(exfit_cv, s = "lambda.1se"), coef(exfit_cv$fit, s = exfit_cv$lambda.1se))

    expect_equal(predict(exfit_cv, lambda = "lambda.min"), predict(exfit_cv$fit, lambda = exfit_cv$lambda.min))
    expect_equal(predict(exfit_cv, lambda = "lambda.1se"), predict(exfit_cv$fit, lambda = exfit_cv$lambda.1se))
    expect_equal(predict(exfit_cv, s = "lambda.min"), predict(exfit_cv$fit, s = exfit_cv$lambda.min))
    expect_equal(predict(exfit_cv, s = "lambda.1se"), predict(exfit_cv$fit, s = exfit_cv$lambda.1se))
})
