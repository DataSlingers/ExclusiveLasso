#' CV for the Exclusive Lasso
#'
#' @rdname cv.exclusive_lasso
#' @export
#' @importFrom foreach foreach %do% %dopar%
#' @importFrom stats sd
#' @param X The matrix of predictors (\eqn{X \in \R^{n \times p}}{X})
#' @param y The response vector (\eqn{y})
#' @param groups An integer vector of length \eqn{p} indicating group membership.
#'     (Cf. the \code{index} argument of \code{\link[grplasso]{grpreg}})
#' @param ... Additional arguments passed to \code{\link{exclusive_lasso}}.
#' @param type.measure The loss function to be used for cross-validation.
#' @param nfolds The number of folds (\eqn{K}) to be used for K-fold CV
#' @param parallel Should CV run in parallel? If a parallel back-end for the
#'    \code{foreach} package is registered, it will be used. See the
#'    \code{foreach} documentation for details of different backends.
#' @param family The GLM response type. (Cf. the \code{family} argument of
#'               \code{\link[stats]{glm}})
#' @param weights Weights applied to individual
#'     observations. If not supplied, all observations will be equally
#'     weighted. Will be re-scaled to sum to \eqn{n} if
#'     necessary. (Cf. the \code{weight} argument of
#'     \code{\link[stats]{lm}})
#' @param offset A vector of length \eqn{n} included in the linear
#'     predictor.
#' @details As discussed in Appendix F of Campbell and Allen [1], cross-validation
#'    can be quite unstable for exclusive lasso problems. Model selection by BIC
#'    or EBIC tends to perform better in practice.
#' @references
#' Campbell, Frederick and Genevera I. Allen. "Within Group Variable Selection
#'     with the Exclusive Lasso." Electronic Journal of Statistics 11(2),
#'     pp.4220-4257. 2017. \doi{10.1214/EJS-1317}
#' @examples
#' n <- 200
#' p <- 500
#' groups <- rep(1:10, times=50)
#' beta <- numeric(p);
#' beta[1:10] <- 3
#'
#' X <- matrix(rnorm(n * p), ncol=p)
#' y <- X %*% beta + rnorm(n)
#'
#' exfit_cv <- cv.exclusive_lasso(X, y, groups, nfolds=5)
#' print(exfit_cv)
#' plot(exfit_cv)
#'
#' # coef() and predict() work just like
#' # corresponding methods for exclusive_lasso()
#' # but can also specify lambda="lambda.min" or "lambda.1se"
#' coef(exfit_cv, lambda="lambda.1se")
cv.exclusive_lasso <- function(X, y, groups, ...,
                               family = c("gaussian", "binomial", "poisson"),
                               offset = rep(0, NROW(X)),
                               weights = rep(1, NROW(X)),
                               type.measure=c("mse", "deviance", "class", "auc", "mae"),
                               nfolds=10, parallel=FALSE){

    tic <- Sys.time()

    family       <- match.arg(family)
    type.measure <- match.arg(type.measure)

    ## The full data fit below will handle most input checking, but this needs to
    ## be done here to be reflected in the CV fits
    if(sum(weights) != NROW(X)){
        weights <- weights * NROW(X) / sum(weights)
        warning(sQuote("sum(weights)"), " is not equal to ", sQuote("NROW(X)."), " Renormalizing...")
    }

    if ( (family != "binomial") && (type.measure %in% c("auc", "class")) ){
        stop("Loss type ", sQuote(type.measure), " only defined for family = \"binomial.\"")
    }

    if ( (type.measure %in% c("auc", "class")) && (!all(y %in% c(0, 1))) ) {
        stop("Loss type ", sQuote(type.measure), " is only defined for 0/1 responses.")
    }

    fit <- exclusive_lasso(X = X, y = y,
                           groups = groups, family = family,
                           offset = offset, weights = weights, ...)

    lambda <- fit$lambda

    fold_ids <- split(sample(NROW(X)), rep(1:nfolds, length.out=NROW(X)))

    `%my.do%` <- if(parallel) `%dopar%` else `%do%`

    i <- NULL ## Hack to avoid global variable warning in foreach call below

    loss_func <- switch(type.measure,
                        mse = function(test_true, test_pred, w) weighted.mean((test_true - test_pred)^2, w),
                        mae = function(test_true, test_pred, w) weighted.mean(abs(test_true - test_pred), w),
                        class = function(test_true, test_pred, w) weighted.mean(round(test_pred) == test_true, w),
                        deviance = function(test_true, test_pred, w) weighted.mean(deviance_loss(test_true, test_pred, family), w),
                        stop(sQuote(type.measure), "loss has not yet been implemented."))

    cv_err <- foreach(i=1:nfolds, .inorder=FALSE,
                      .packages=c("ExclusiveLasso", "Matrix")) %my.do% {

            X_tr <- X[-fold_ids[[i]], ];           X_te <- X[fold_ids[[i]], ]
            y_tr <- y[-fold_ids[[i]]];             y_te <- y[fold_ids[[i]]]
            offset_tr <- offset[-fold_ids[[i]]];   offset_te <- offset[fold_ids[[i]]]
            weights_tr <- weights[-fold_ids[[i]]]; weights_te <- weights[fold_ids[[i]]]

            my_fit <- exclusive_lasso(X = X_tr, y = y_tr, groups = groups, lambda=lambda,
                                      ..., family = family, offset = offset_tr,
                                      weights = weights_tr)

            apply(predict(my_fit, newx=X_te, offset = offset_te), 2,
                  function(y_hat) loss_func(y_te, y_hat, weights_te))
    }

    cv_err <- do.call(cbind, cv_err)

    cv_res <- apply(cv_err, 1,
                    function(x){
                        m <- mean(x)
                        se <- sd(x) / (length(x) - 1)
                        up <- m + se
                        lo <- m - se
                        c(m, se, up, lo)
                    })

    min_ix <- which.min(cv_res[1,])
    lambda.min <- lambda[min_ix]

    ## The "One-standard error" rule is defined as the largest lambda such that
    ## CV(lambda) <= CV(lambda_min) + SE(CV(lambda_min)) where lambda_min is the
    ## lambda giving minimal CV error

    lambda_min_plus_1se <- cv_res[3, min_ix]
    oneSE_ix <- max(which(cv_res[1,] <= lambda_min_plus_1se))
    lambda.1se <- lambda[oneSE_ix]

    r <- list(fit=fit,
              lambda=lambda,
              cvm=cv_res[1,],
              cvsd=cv_res[2,],
              cvup=cv_res[3,],
              cvlo=cv_res[4,],
              lambda.min=lambda.min,
              lambda.1se=lambda.1se,
              name=type.measure,
              time=Sys.time() - tic)

    class(r) <- c("ExclusiveLassoFit_cv", class(r))

    r
}

#' @noRd
#' Deviance Loss for CV -- we define this here for readability, but it's only used
#' one small place inside CV
deviance_loss <- function(y, mu, family = c("gaussian", "binomial", "poisson")){
    family <- match.arg(family)

    switch(family,
           gaussian = (y - mu)^2,
           binomial = y * mu - log(1 + exp(mu)),
           poisson  = y * log(mu) - mu)
}

#' @export
print.ExclusiveLassoFit_cv <- function(x, ...){
    cat("Exclusive Lasso CV", "\n")
    cat("------------------", "\n")
    cat("\n")
    cat("Lambda (Min Rule):", x$lambda.min, "\n")
    cat("Lambda (1SE Rule):", x$lambda.1se, "\n")
    cat("\n")
    cat("Loss Function:", x$name, "\n")
    cat("\n")
    cat("Time: ", sprintf("%2.3f %s", x$time, attr(x$time, "units")), "\n")
    cat("\n")
    cat("Full Data Fit", "\n")
    cat("-------------", "\n")
    print(x$fit, indent=2)

    invisible(x)
}

#' @export
#' @importFrom stats predict
predict.ExclusiveLassoFit_cv <- function(object, ...){
    dots <- list(...)
    if("s" %in% names(dots)){
        s <- dots$s
        if(s == "lambda.min"){
            s <- object$lambda.min
        }
        if(s == "lambda.1se"){
            s <- object$lambda.1se
        }
        dots$s <- s
    }
    if("lambda" %in% names(dots)){
        lambda <- dots$lambda
        if(lambda == "lambda.min"){
            lambda <- object$lambda.min
        }
        if(lambda == "lambda.1se"){
            lambda  <- object$lambda.1se
        }
        dots$lambda  <- lambda
    }

    do.call(predict, c(list(object$fit), dots))
}


#' @export
#' @importFrom stats coef
coef.ExclusiveLassoFit_cv <- function(object, ...){
    dots <- list(...)
    if("s" %in% names(dots)){
        s <- dots$s
        if(s == "lambda.min"){
            s <- object$lambda.min
        }
        if(s == "lambda.1se"){
            s <- object$lambda.1se
        }
        dots$s <- s
    }
    if("lambda" %in% names(dots)){
        lambda <- dots$lambda
        if(lambda == "lambda.min"){
            lambda <- object$lambda.min
        }
        if(lambda == "lambda.1se"){
            lambda  <- object$lambda.1se
        }
        dots$lambda  <- lambda
    }

    do.call(coef, c(list(object$fit), dots))
}
