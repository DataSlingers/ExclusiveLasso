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
