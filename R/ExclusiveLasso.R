#' Fit a GLM with Exclusive Lasso Regularization
#'
#' Fit a generalized linear model via maximum penalized likelihood
#' using the exclusive lasso penalty. The regularization path is computed
#' along a grid of values for the regularization parameter (lambda).
#' The interface is intentionally similar to that of \code{\link[glmnet]{glmnet}} in
#' the package of the same name.
#'
#' Note that unlike Campbell and Allen (2017), we use the "1/n"-scaling of the
#' loss function.
#'
#' For the Gaussian case:
#' \deqn{\frac{1}{2n}|y - X\beta|_2^2 + \lambda P(\beta, G)}
#'
#' For other GLMs:
#' \deqn{-\frac{1}{n}\ell(y, X\beta)+ \lambda P(\beta, G)}
#'
#' @param X The matrix of predictors (\eqn{X \in \R^{n \times p}}{X})
#' @param y The response vector (\eqn{y})
#' @param groups An integer vector of length \eqn{p} indicating group membership.
#'     (Cf. the \code{index} argument of \code{\link[grplasso]{grplasso}})
#' @param family The GLM response type. Currently only \code{family="gaussian"}
#'     is implemented. (Cf. the \code{family} argument of \code{\link[stats]{glm}})
#' @param weights Weights applied to individual
#'     observations. If not supplied, all observations will be equally
#'     weighted. Will be re-scaled to sum to \eqn{n} if
#'     necessary. (Cf. the \code{weight} argument of
#'     \code{\link[stats]{lm}})
#' @param offset A vector of length \eqn{n} included in the linear
#'     predictor.
#' @param nlambda The number of lambda values to use in computing the
#'    regularization path. Note that the time to run is typically sublinear
#'    in the grid size due to the use of warm starts.
#' @param lambda.min.ratio The smallest value of lambda to be used, as a fraction
#'      of the largest value of lambda used. Unlike the lasso, there is no
#'      value of lambda such that the solution is wholly sparse, but we still
#'      use lambda_max from the lasso.
#' @param lambda A user-specified sequence of lambdas to use.
#' @param standardize Should \code{X} be centered and scaled before fitting?
#' @param intercept Should the fitted model have an (unpenalized) intercept term?
#' @param thresh The convergence threshold used for the proximal
#'    gradient or coordinate-descent algorithm used to solve the
#'    penalized regression problem.
#' @param thresh_prox The convergence threshold used for the
#'    coordinate-descent algorithm used to evaluate the proximal operator.
#' @param algorithm Which algorithm to use, proximal gradient (\code{"pg"}) or
#'    coordinate descent (\code{"cd"})? Empirically, coordinate descent appears
#'    to be faster for most problems (consistent with Campbell and Allen), but
#'    proximal gradient may be faster for certain problems with many small groups
#'    where the proximal operator may be evaluated quickly and to high precision.
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
#' exfit <- exclusive_lasso(X, y, groups)
#' @importFrom stats median weighted.mean
#' @importFrom Matrix Matrix
#' @references
#' Campbell, Frederick and Genevera I. Allen. "Within Group Variable Selection
#'     with the Exclusive Lasso". Electronic Journal of Statistics (to appear).
#' @export
exclusive_lasso <- function(X, y, groups, family=c("gaussian", "binomial", "poisson"),
                            weights, offset, nlambda=100,
                            lambda.min.ratio=ifelse(nobs < nvars, 0.01, 1e-04),
                            lambda, standardize=TRUE, intercept=TRUE,
                            thresh=1e-07, thresh_prox=thresh,
                            algorithm=c("cd", "pg")){

    tic <- Sys.time()

    ####################
    ##
    ## Input Validation
    ##
    ####################

    nobs <- NROW(X); nvars <- NCOL(X);

    if(length(y) != nobs){
        stop(sQuote("NROW(X)"), " and ", sQuote("length(y)"), " must match.")
    }
    if(length(groups) != nvars){
        stop(sQuote("NCOL(X)"), " and ", sQuote("length(groups)"), " must match.")
    }

    if(anyNA(X) || anyNA(y)){
        stop(sQuote("exclusive_lasso"), " does not support missing data.")
    }

    ## Index groups from 0 to `num_unique(groups) - 1` to represent
    ## in a arma::ivec
    groups <- match(groups, unique(groups)) - 1

    family <- match.arg(family)

    if(family != "gaussian"){
        stop(sQuote("exclusive_lasso"), " currently only supports the ", sQuote("gaussian"), " family.")
    }

    if(missing(weights)){
        weights <- rep(1, nobs)
    }
    if(length(weights) != nobs){
        stop(sQuote("NROW(X)"), " and ", sQuote("length(weights)"), " must match.")
    }
    if(any(weights <= 0)){
        stop("Observation weights must be strictly positive.")
    }
    if(sum(weights) != nobs){
        weights <- weights * nobs / sum(weights)
        warning(sQuote("sum(weights)"), " is not equal to ", sQuote("NROW(X)."), " Renormalizing...")
    }

    if(missing(offset)){
        offset <- rep(0, nobs)
    }
    if(length(offset) != nobs){
        stop(sQuote("NROW(X)"), " and ", sQuote("length(offset)"), " must match.")
    }

    nlambda <- as.integer(nlambda)

    if((lambda.min.ratio <= 0) || (lambda.min.ratio >= 1)){
        stop(sQuote("lambda.min.ratio"), " must be in the interval (0, 1).")
    }

    if(standardize){
        ## FIXME -- This form of standardizing X isn't quite right with observation weights
        Xsc <- scale(X, center=TRUE, scale=TRUE)
        X_scale <- attr(Xsc, "scaled:scale", exact=TRUE)
    } else {
        Xsc <- X
        X_scale <- rep(1, nobs)
    }

    if(missing(lambda)){
        lambda_max <- max(abs(crossprod(X, y - offset - weighted.mean(y, weights) * intercept)/nobs))
        lambda <- logspace(lambda.min.ratio * lambda_max, lambda_max, length.out=nlambda)
    }

    if(length(lambda) < 1){
        stop("Must solve for at least one value of lambda.")
    }

    if(any(lambda <= 0)){
        stop("All values of ", sQuote("lambda"), " must be positive.")
    }

    if(is.unsorted(lambda)){
        warning("User-supplied ", sQuote("lambda"), " is not increasing Sorting for maximum performance.")
        lambda <- sort(lambda)
    }

    if(thresh_prox <= 0){
        stop(sQuote("thresh_prox"), " must be positive.")
    }

    if(thresh <= 0){
        stop(sQuote("thresh"), " must be positive.")
    }

    algorithm <- match.arg(algorithm)

    if(algorithm == "cd"){
        coef <- exclusive_lasso_gaussian_cd(X=Xsc, y=y, groups=groups,
                                           lambda=lambda, w=weights, o=offset,
                                           thresh=thresh)
    } else {
        coef <- exclusive_lasso_gaussian_pg(X=Xsc, y=y, groups=groups,
                                            lambda=lambda, w=weights, o=offset,
                                            thresh_prox=thresh_prox, thresh=thresh)
    }

    coef <- Matrix(coef, sparse=TRUE)

    ## Convert coefficients back to original scale
    if(standardize){
        coef <- coef * X_scale
    }

    if(!is.null(colnames(X))){
        rownames(coef) <- colnames(X)
    }

    if(intercept){
        err <- matrix(y - offset, nrow=length(y), ncol=length(lambda)) - X %*% coef
        intercept <- apply(err, 2, function(x) weighted.mean(x, weights))
    } else {
        intercept <- NULL
    }

    result <- list(coef=coef,
                   intercept=intercept,
                   y=y,
                   X=X,
                   standardize=standardize,
                   groups=groups,
                   lambda=lambda,
                   weights=weights,
                   offset=offset,
                   family=family,
                   time=Sys.time() - tic)

    class(result) <- c("ExclusiveLassoFit", class(result))

    result
}

has_intercept <- function(x){
    !is.null(x$intercept)
}

has_offset <- function(x){
    any(x$offset != 0)
}

#' @export
print.ExclusiveLassoFit <- function(x, ..., indent=0){
    icat("Exclusive Lasso Fit", "\n", indent=indent)
    icat("-------------------", "\n", indent=indent)
    icat("\n", indent=indent)
    icat("N: ", NROW(x$X), ". P: ", NCOL(x$X), ".\n", sep="", indent=indent)
    icat(length(unique(x$groups)), "groups. Median size", median(table(x$groups)), "\n", indent=indent)
    icat("\n", indent=indent)
    icat("Grid:", length(x$lambda), "values of lambda. \n", indent=indent)
    icat("  Miniumum:", min(x$lambda), "\n", indent=indent)
    icat("  Maximum: ", max(x$lambda), "\n", indent=indent)
    icat("\n", indent=indent)
    icat("Fit Options:\n", indent=indent)
    icat("  - Family:        ", capitalize_string(x$family), "\n", indent=indent)
    icat("  - Intercept:     ", has_intercept(x), "\n", indent=indent)
    icat("  - Standardize X: ", x$standardize, "\n", indent=indent)
    icat("\n", indent=indent)
    icat("Time: ", sprintf("%2.3f %s", x$time, attr(x$time, "units")), "\n", indent=indent)
    icat("\n", indent=indent)

    invisible(x)
}

# Refit exclussive lasso on new lambda grid
# Used internally by predict(exact=TRUE)
#' @importFrom utils modifyList
update_fit <- function(object, lambda, ...){
    ARGS <- list(X=object$X,
                 y=object$y,
                 groups=object$groups,
                 weights=object$weights,
                 offset=object$offset,
                 family=object$gamily,
                 standardize=object$standardize,
                 intercept=has_intercept(object),
                 lambda=lambda)

    ARGS <- modifyList(ARGS, list(...))

    do.call(exclusive_lasso, ARGS)
}

#' @rdname predict.ExclusiveLassoFit
#' @export
#' @importFrom stats predict
coef.ExclusiveLassoFit <- function(object, lambda=s, s=NULL, exact=FALSE, ...){
    predict(object, lambda=lambda, type="coefficients", exact=exact, ...)
}

#' Make predictions using the exclusive lasso.
#'
#' Make predictions using the exclusive lasso. Similar to \code{\link[glmnet]{predict.glmnet}}.
#' \code{coef(...)} is a wrapper around \code{predict(..., type="coefficients")}.
#'
#' @rdname predict.ExclusiveLassoFit
#' @export
#' @param object An \code{ExclusiveLassoFit} object produced by \code{\link{exclusive_lasso}}.
#' @param newx New data \eqn{X \in R^{m \times p}}{X} on which to make predictions. If not
#'    supplied, predictions are made on trainng data.
#' @param s An alternate argument that may be used to supply \code{lambda}. Included for
#'    compatability with \code{\link[glmnet]{glmnet}}.
#' @param lambda The value of the regularization paramter (\eqn{lambda}) at which to
#'    return the fitted coefficients or predicted values. If not supplied, results for
#'    the entire regularization path are returned. Can be a vector.
#' @param type The type of "prediction" to return. If \code{type="link"}, returns
#'    the linear predictor. If \code{type="response"}, returns the expected
#'    value of the response. If \code{type="coefficients"}, returns the coefficients
#'    used to calculate the linear predictor. (Cf. the \code{type} argument
#'    of \code{\link[glmnet]{predict.glmnet}})
#' @param exact Should the exclusive lasso be re-run for provided values of \code{lambda}?
#'    If \code{FALSE}, approximate values obtained by linear interpolation on grid points
#'    are used instead. (Cf. the \code{exact} argument of \code{\link[glmnet]{predict.glmnet}})
#' @param offset An offset term used in predictions. If not supplied, all offets are
#'    taken to be zero. If the original fit was made with an offset, \code{offset} will
#'    be required.
#' @param ... Additional arguments passed to \code{\link{exclusive_lasso}} if
#'    \code{exact=TRUE} and ignored otherwise.
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
#' exfit <- exclusive_lasso(X, y, groups)
#' coef(exfit, lambda=1)
#' predict(exfit, lambda=1, newx = -X)
predict.ExclusiveLassoFit <- function(object, newx, lambda=s, s=NULL,
                                      type=c("link", "response", "coefficients"),
                                      exact=FALSE, offset, ...){
    type <- match.arg(type)

    ## Get coefficients first
    if(!is.null(lambda)){
        if(exact){
            object <- update_fit(object, lambda=lambda, ...)

            if(has_intercept(object)){
               int <- Matrix(object$intercept, nrow=1,
                             sparse=TRUE, dimnames=list("(Intercept)", NULL))
            } else {
                int <- Matrix(0, nrow=1, ncol=length(object$lambda),
                              sparse=TRUE, dimnames=list("(Intercept)", NULL))
            }

            coef <- rbind(int, object$coef)
        } else {
            if(has_intercept(object)){
               int <- Matrix(object$intercept, nrow=1,
                             sparse=TRUE, dimnames=list("(Intercept)", NULL))
            } else {
                int <- Matrix(0, nrow=1, ncol=length(object$lambda),
                              sparse=TRUE, dimnames=list("(Intercept)", NULL))
            }

            coef <- rbind(int, object$coef)
            lambda <- clamp(lambda, range=range(object$lambda))

            coef <- lambda_interp(coef,
                                  old_lambda=object$lambda,
                                  new_lambda=lambda)
        }
    } else {
        if(has_intercept(object)){
               int <- Matrix(object$intercept, nrow=1,
                             sparse=TRUE, dimnames=list("(Intercept)", NULL))
            } else {
                int <- Matrix(0, nrow=1, ncol=length(object$lambda),
                              sparse=TRUE, dimnames=list("(Intercept)", NULL))
            }

        coef <- rbind(int, object$coef)
    }

    if(type == "coefficients"){
        return(coef) ## Done
    }

    if(missing(newx)){
        link <- object$offset + cbind(1, object$X) %*% coef
    } else {
        if(missing(offset)){
            if(has_offset(object)){
                stop("Original fit had an offset term but", sQuote("offset"), "not supplied.")
            } else {
                offset <- rep(0, NROW(newx))
            }
        }
        link <- offset + cbind(1, newx) %*% coef
    }

    link <- as.matrix(link)

    if(type == "link"){
        link
    } else {
        ## Returning response
        switch(object$family,
           gaussian=link,
           binomial=inv_logit(link),
           poisson=exp(link))
    }
}
