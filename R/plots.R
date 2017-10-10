#' Visualizing Exclusive Lasso Regularization Paths
#'
#' @importMethodsFrom Matrix colSums
#' @importFrom RColorBrewer brewer.pal
#' @export
#' @param x An \code{ExclusiveLassoFit} object produced by \code{\link{exclusive_lasso}}
#' @param xvar Value to use on the x-axis (ordinate). \itemize{
#'    \item \code{norm}: The value of the composite \eqn{\ell_2/\ell_1}{l2/l1} norm
#'        "exclusive lasso" norm
#'    \item \code{lambda}: The log of the regularization parameter \eqn{\lambda}{lambda}.
#'    \item \code{l1}: The \eqn{\ell_1}{l1}-norm of the solution
#'    }
#' @param label Should variables be labeled? Set \code{label=NULL} to label only
#'    those variables which are non-zero at the sparsest end of
#'    the regularization path.
#' @param legend Should a legend of top variables (heuristically selcted) be displayed?
#'    Set \code{legend=NULL} to disable, else give the location of the legend.
#' @param ... Additional arguments passed to plotting functions
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
#' plot(exfit)
#' plot(exfit, legend=NULL, xvar="lambda")
plot.ExclusiveLassoFit <- function(x, xvar=c("norm", "lambda", "l1"),
                                   label=FALSE, legend="topright", ...){
    xvar <- match.arg(xvar)
    xvar_rev <- (xvar != "lambda")

    if(xvar == "norm"){
        xvar <- apply(x$coef, 2, el_norm, x$groups)
        xlab <- expression(sum(paste("|", beta[g], "|")[1]^2, g %in% G))
    } else if (xvar == "lambda"){
        xvar <- log(x$lambda)
        xlab <- expression(log(lambda))
    } else {
        xvar <- colSums(abs(x$coef))
        xlab <- expression(paste("|", beta, "|")[1])
    }

    COLORS <- brewer.pal(12, "Set3")

    plot(xvar, x$coef[1, ], type="n", xlab=xlab,
         ylim=range(x$coef),
         xlim=if(xvar_rev) rev(range(xvar)) else range(xvar),
         ylab=expression(hat(beta)))

    if(length(colnames(x$X))){
        do_legend <- !is.null(legend)
        if(is.null(label)){
            do_label <- TRUE
            do_label_all <- FALSE
        } else{
            do_label <- label
            do_label_all <- FALSE
        }
    } else {
        do_legend <- FALSE
        do_label <- FALSE
        do_label_all <- FALSE
    }

    legend_labels <- character()
    legend_colors <- character()

    for(i in 1:NCOL(x$X)){
        y <- x$coef[i, ]; col <- COLORS[(x$groups[i] %% 12) + 1]
        lines(xvar, y, col=col, ...)
        if(do_label || do_legend){
            if(do_label_all || (y[length(y)] != 0)){
                if(do_legend){
                    legend_labels <- c(legend_labels, labels[i]);
                    legend_colors <- c(legend_colors, col);
                }

                if(do_label){

                    text_x <- range(xvar) * c(0.95, 1.05)
                    text_x <- if(xvar_rev) min(text_x) else max(text_x)

                    text(text_x, y[length(y)] * runif(1, 0.95, 1.05),
                         labels[i], col=col, offset=3)
                }
            }
        }
    }

    if(do_legend){
        legend(legend, legend=legend_labels, col=legend_colors, bg="white", ...)
    }
}

el_norm <- function(x, g){
    sum(tapply(x, g, function(x) sum(abs(x)))^2)
}
