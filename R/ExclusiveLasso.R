GLM_FAMILIES <- c(gaussian=0,
                  binomial=1,
                  poisson=2)

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
#'     (Cf. the \code{group} argument of \code{\link[grpreg]{grpreg}})
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
#' @param skip_df Should the DF calculations be skipped? They are often slower
#'    than the actual model fitting; if calling \code{exclusive_lasso} repeatedly
#'    it may be useful to skip these calculations.
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
#' @importMethodsFrom Matrix colMeans colSums
#' @importClassesFrom Matrix dgCMatrix
#' @return An object of class \code{ExclusiveLassoFit} containing \itemize{
#' \item \code{coef} - A matrix of estimated coefficients
#' \item \code{intercept} - A vector of estimated intercepts if \code{intercept=TRUE}
#' \item \code{X, y, groups, weights, offset} - The data used to fit the model
#' \item \code{lambda} - The vector of \eqn{\lambda}{lambda} used
#' \item \code{df} - An unbiased estimate of the degrees of freedom (see Theorem
#'       5 in [1])
#' \item \code{nnz} - The number of non-zero coefficients at each value of
#'       \eqn{\lambda}{lambda}
#' }
#' @references
#' Campbell, Frederick and Genevera I. Allen. "Within Group Variable Selection
#'     with the Exclusive Lasso". Electronic Journal of Statistics 11(2),
#'     pp.4220-4257. 2017. \doi{10.1214/17-EJS1317}
#' @export
exclusive_lasso <- function(X, y, groups, family=c("gaussian", "binomial", "poisson"),
                            weights, offset, nlambda=100,
                            lambda.min.ratio=ifelse(nobs < nvars, 0.01, 1e-04),
                            lambda, standardize=TRUE, intercept=TRUE,
                            thresh=1e-07, thresh_prox=thresh,
                            skip_df=FALSE,
                            algorithm=c("cd", "pg")){

    tic <- Sys.time()

    ####################
    ##
    ## Input Validation
    ##
    ####################

    nobs <- NROW(X);
    nvars <- NCOL(X);

    if(length(y) != nobs){
        stop(sQuote("NROW(X)"), " and ", sQuote("length(y)"), " must match.")
    }

    if(length(groups) != nvars){
        stop(sQuote("NCOL(X)"), " and ", sQuote("length(groups)"), " must match.")
    }

    if(anyNA(X) || anyNA(y)){
        stop(sQuote("exclusive_lasso"), " does not support missing data.")
    }

    if(!all(is.finite(X))){
        stop("All elements of ", sQuote("X"), " must be finite.")
    }

    if(!all(is.finite(y))){
        stop("All elements of ", sQuote("y"), " must be finite.")
    }

    ## Index groups from 0 to `num_unique(groups) - 1` to represent
    ## in a arma::ivec
    groups <- match(groups, unique(groups)) - 1

    family <- match.arg(family)

    if(family == "poisson"){
        stop("Poisson not yet implemented.")
    }

    if(family == "poisson"){
        if(any(y < 0)){
            stop(sQuote("y"), " must be non-negative for Poisson regression.")
        }
    }

    if(family == "binomial"){
        if(any(y < 0) || any(y > 1)){
            stop(sQuote("y"), " must be in [0, 1] for logistic regression.")
        }
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
        X_center <- attr(Xsc, "scaled:center", exact=TRUE)

        if(!all(is.finite(Xsc))){
            stop("Non-finite ", sQuote("X"), " found after standardization.")
        }
    } else {
        Xsc <- X
        X_scale <- rep(1, nvars)
        X_center <- rep(0, nvars)
    }

    if(missing(lambda)){
        lambda_max <- max(abs(crossprod(Xsc, y - offset - weighted.mean(y, weights) * intercept)/nobs))
        lambda <- logspace(lambda.min.ratio * lambda_max, lambda_max, length.out=nlambda)
    }

    if(length(lambda) < 1){
        stop("Must solve for at least one value of lambda.")
    }

    if(any(lambda <= 0)){
        stop("All values of ", sQuote("lambda"), " must be positive.")
    }

    if(is.unsorted(lambda)){
        warning("User-supplied ", sQuote("lambda"), " is not increasing. Sorting for maximum performance.")
        lambda <- sort(lambda)
    }

    if(thresh_prox <= 0){
        stop(sQuote("thresh_prox"), " must be positive.")
    }

    if(thresh <= 0){
        stop(sQuote("thresh"), " must be positive.")
    }

    algorithm <- match.arg(algorithm)

    if((family == "gaussian") && getOption("ExclusiveLasso.gaussian_fast_path", TRUE)){
        if(algorithm == "cd"){
            res <- exclusive_lasso_gaussian_cd(X=Xsc, y=y, groups=groups,
                                               lambda=lambda, w=weights, o=offset,
                                               thresh=thresh, intercept=intercept)
        } else {
            res <- exclusive_lasso_gaussian_pg(X=Xsc, y=y, groups=groups,
                                               lambda=lambda, w=weights, o=offset,
                                               thresh_prox=thresh_prox, thresh=thresh,
                                               intercept=intercept)
        }
    } else {
        if(algorithm == "cd"){
            stop("Coordinate Descent for GLMs not yet implemented!")
        }

        res <- exclusive_lasso_glm_pg(X=Xsc, y=y, group=groups,
                                      lambda=lambda, w=weights, o=offset,
                                      family=GLM_FAMILIES[family],
                                      thresh=thresh, thresh_prox=thresh_prox,
                                      intercept=intercept)
    }

    ## Convert intercept to R vector (arma::vec => R column vector)
    res$intercept <- as.vector(res$intercept)

    ## Convert coefficients and intercept back to original scale
    if(standardize){
        ## To get back to the original X, we multiply by X_scale,
        ## so we divide beta to keep things on the same unit
        res$coef <- res$coef / X_scale
        if(intercept){
            ## Map back to original X (undo scale + center)
            ##
            ## We handled the scaling above, now we adjust for the
            ## centering of X: beta(X - colMeans(X)) = beta * X - beta * colMeans(X)
            ## To uncenter we add back in beta * colMeans(X), summed over all observations
            res$intercept <- res$intercept - colSums(res$coef * X_center)
        }
    }

    ## Degrees of freedom -- calculated using original scale matrix
    ## (though it shouldn't really matter)
    ##
    ## Loop over each solution and calculate M matrix
    ## and from there DF
    if(!skip_df){
        df <- rep(NA, length(lambda))
        for(ix in seq(length(lambda), 1, by=-1)){
            l <- lambda[ix]
            S <- which(res$coef[,ix] != 0)
            M <- matrix(0, nrow=nvars, ncol=nvars)

            for(g in unique(groups)){
                g_ix <- which(groups == g)
                s_g <- sign(res$coef[g_ix, ix])
                M[g_ix, g_ix] <- outer(s_g, s_g)
            }

            diag(M) <- diag(M) + 1e-4 ## For numerical stability in inverse step
            X_S <- X[,S]
            projection_mat <- X_S %*% solve(crossprod(X_S) + nobs * l * M[S, S], t(X_S))
            df[ix] <- sum(diag(projection_mat))
        }
    } else {
      df <- NULL
    }

    if(!is.null(colnames(X))){
        rownames(res$coef) <- colnames(X)
    }

    result <- list(coef=res$coef,
                   intercept=res$intercept,
                   y=y,
                   X=X,
                   standardize=standardize,
                   groups=groups,
                   lambda=lambda,
                   weights=weights,
                   offset=offset,
                   family=family,
                   df=df,
                   algorithm=algorithm,
                   nnz=apply(res$coef, 2, function(x) sum(x != 0)),
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
    if(!is.null(x$df)){
      icat("  Degrees of freedom: ", min(x$df), " --> ", max(x$df), "\n", indent=indent)
    }
    icat("  Number of selected variables:", min(x$nnz), " --> ", max(x$nnz), "\n", indent=indent)
    icat("\n", indent=indent)
    icat("Fit Options:\n", indent=indent)
    icat("  - Family:        ", capitalize_string(x$family), "\n", indent=indent)
    icat("  - Intercept:     ", has_intercept(x), "\n", indent=indent)
    icat("  - Standardize X: ", x$standardize, "\n", indent=indent)
    icat("  - Algorithm:     ", switch(x$algorithm, pg="Proximal Gradient", cd="Coordinate Descent"),
         "\n", indent=indent)
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
coef.ExclusiveLassoFit <- function(object, lambda=s, s=NULL,
                                   exact=FALSE, group_threshold=FALSE, ...){

    predict(object, lambda=lambda, type="coefficients",
            exact=exact, group_threshold=group_threshold, ...)
}

#' Make predictions using the exclusive lasso.
#'
#' Make predictions using the exclusive lasso. Similar to \code{\link[glmnet]{predict.glmnet}}.
#' \code{coef(...)} is a wrapper around \code{predict(..., type="coefficients")}.
#'
#' @rdname predict.ExclusiveLassoFit
#' @importFrom Matrix Matrix
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
#' @param group_threshold If \code{TRUE}, (hard-)threshold coefficients so that
#'    there is exactly one non-zero coefficient in each group.
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
                                      group_threshold=FALSE,
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

    if(group_threshold){
            coef[-1,,drop=FALSE] <- Matrix(apply(coef[-1,,drop=FALSE], 2, do_group_threshold, object$groups), sparse=TRUE)
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

do_group_threshold <- function(x, groups){
    for(g in unique(groups)){
        g_ix <- (g == groups)
        x[g_ix] <- x[g_ix] * (abs(x[g_ix]) == max(abs(x[g_ix])))
    }
    x
}
