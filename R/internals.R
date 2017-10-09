eye <- function(n) diag(1, nrow=n, ncol=n)

is_scaled <- function(X){
    ((!is.null(attr(X, "scaled:scale",  exact=TRUE))) &
         (!is.null(attr(X, "scaled:center", exact=TRUE))))
}

unscale_matrix <- function(X,
                           scale=attr(X, "scaled:scale", TRUE),
                           center=attr(X, "scaled:center", TRUE)){
    n <- NROW(X)
    p <- NCOL(X)

    X * matrix(scale, n, p, byrow=TRUE) + matrix(center, n, p, byrow=TRUE)
}


capitalize_string <- function(x){
    paste0(toupper(substring(x, 1, 1)), substring(x, 2))
}

clamp <- function(x, range){
    pmin(pmax(x, min(range)), max(range))
}


lambda_interp <- function(x, old_lambda, new_lambda){
    new_lambda <- clamp(new_lambda, range(old_lambda))

    lb <- vapply(new_lambda, function(x) max(which(old_lambda <= x)), numeric(1))
    ub <- vapply(new_lambda, function(x) min(which(old_lambda >= x)), numeric(1))

    lb <- clamp(lb, c(1, length(old_lambda)))
    ub <- clamp(ub, c(1, length(old_lambda)))

    frac <- (new_lambda - old_lambda[lb]) / (old_lambda[ub] - old_lambda[lb])

    frac[lb == ub] <- 1


    frac * x[, lb, drop=FALSE] + (1-frac) * x[, ub, drop=FALSE]
}


#' @importFrom stats predict
inv_logit <- function(x) plogis(x)
