#' ExclusiveLasso: Generalized Linear Models with the Exclusive Lasso Penalty
#'
#' Fitting generalized linear models with the exclusive lasso penalty using
#' the coordinate descent and inexact proximal gradient algorithms of
#' Campbell and Allen [1].
#'
#' @references
#' [1] Campbell, Frederick and Genevera I. Allen. "Within Group Variable Selection
#'     with the Exclusive Lasso". Electronic Journal of Statistics 11(2),
#'     pp.4220-4257. 2017. \url{https://doi.org/10.1214/17-EJS1317}
#'
#' @importFrom Rcpp evalCpp
#' @docType package
#' @name ExclusiveLasso-pkg
#' @useDynLib ExclusiveLasso
NULL
