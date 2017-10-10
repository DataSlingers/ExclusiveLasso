#' CV for the Exclusive Lasso
#'
#' @rdname cv.exclusive_lasso
#' @importFrom foreach foreach %do% %dopar%
#' @importFrom stats sd
#' @param X The matrix of predictors (\eqn{X \in \R^{n \times p}}{X})
#' @param y The response vector (\eqn{y})
#' @param groups An integer vector of length \eqn{p} indicating group membership.
#'     (Cf. the \code{index} argument of \code{\link[grplasso]{grplasso}})
#' @param ... Additional arguments passed to \code{exclusive_lasso}.
#' @param type.measure The loss function to be used for cross-validation.
#' @param nfolds The number of folds (\eqn{K}) to be used for K-fold CV
#' @param parallel Should CV run in parallel? If a parallel back-end for the
#'    \code{foreach} package is registered, it will be used. See the
#'    \code{foreach} documentation for details of different backends.
cv.exclusive_lasso <- function(X, y, groups, ...,
                               type.measure=c("mse", "deviance", "class", "auc", "mae"),
                               nfolds=10, parallel=TRUE){

    tic <- Sys.time()

    type.measure <- match.arg(type.measure)

    if(type.measure != "mse"){
        stop("Only", sQuote("mse"), "loss currently supported.")
    }

    fit <- exclusive_lasso(X=X, y=y, groups=groups, ...)
    lambdas <- fit$lambda

    fold_ids <- split(sample(NROW(X)), rep(1:nfolds, length.out=NROW(X)))

    `%my.do%` <- if(parallel) `%dopar%` else `%do%`

    i <- NULL ## Hack to avoid global variable warning in foreach call below

    cv_err <- foreach(i=1:nfolds, .inorder=FALSE,
                      .packages="ExclusiveLasso") %my.do% {

            X_tr <- X[-fold_ids[[i]], ]; X_te <- X[fold_ids[[i]], ]
            y_tr <- y[-fold_ids[[i]]]; y_te <- y[fold_ids[[i]]]

            my_fit <- exclusive_lasso(X=X_tr, y=y_tr, groups=groups, ...)

            apply(predict(my_fit, newx=X_te), 2, function(y_hat) mean((y_te - y_hat)^2))
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

    lambda.min <- lambdas[which.min(cv_res[1,])]
    lambda.1se <- lambdas[which.min(cv_res[3,])]

    r <- list(fit=fit,
              lambda=lambdas,
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
