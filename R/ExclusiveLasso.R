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
#'      use lambda_max from the lasso
#' @param lambda A user-specified sequence of lambdas to use.
#' @param standardize Should \code{X} be centered and scaled before fitting?
#' @param intercept Should the fitted model have an (unpenalized) intercept term?
#' @param thresh_prox The convergence threshold used for the
#'    coordinate-descent algorithm used to evaluate the proximal operator.
#' @param thresh_pg The convergence threshold used for the proximal
#'    gradient algorithm used to solve the penalized regression problem.
#'
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
#' @export
exclusive_lasso <- function(X, y, groups, family=c("gaussian", "binomial", "poisson"),
                            weights, offset, nlambda=100,
                            lambda.min.ratio=ifelse(nobs < nvars, 0.01, 1e-04),
                            lambda, standardize=TRUE,
                            intercept=TRUE, thresh_prox=1e-07, thresh_pg=1e-07){

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

    groups <- match(groups, unique(groups))

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
        lambda <- seq(lambda.min.ratio * lambda_max, lambda_max, length.out=nlambda)
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

    if(thresh_pg <= 0){
        stop(sQuote("thresh_pg"), " must be positive.")
    }

    coef <- exclusive_lasso_gaussian(X=Xsc, y=y, groups=groups,
                                     lambda=lambda, w=weights, o=offset,
                                     thresh_prox=thresh_prox, thresh_pg=thresh_pg)

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

#' @export
print.ExclusiveLassoFit <- function(x, ...){
    cat("Exclusive Lasso Fit", "\n")
    cat("-------------------", "\n")
    cat("\n")
    cat("N: ", NROW(x$X), ". P: ", NCOL(x$X), ".\n", sep="")
    cat(length(unique(x$groups)), "groups. Median size", median(table(x$groups)), "\n")
    cat("\n")
    cat("Grid:", length(x$lambda), "values of lambda. \n")
    cat("  Miniumum:", min(x$lambda), "\n")
    cat("  Maximum: ", max(x$lambda), "\n")
    cat("\n")
    cat("Fit Options:\n")
    cat("  - Family:        ", capitalize_string(x$family), "\n")
    cat("  - Intercept:     ", has_intercept(x), "\n")
    cat("  - Standardize X: ", x$standardize, "\n")
    cat("\n")
    cat("Time: ", sprintf("%2.3f %s", x$time, attr(x$time, "units")), "\n")
    cat("\n")

    invisible(x)
}
