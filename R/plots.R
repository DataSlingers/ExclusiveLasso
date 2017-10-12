#' Visualizing Exclusive Lasso Regularization Paths
#'
#' @importMethodsFrom Matrix colSums
#' @importFrom RColorBrewer brewer.pal
#' @importFrom graphics plot lines text legend
#' @importFrom stats runif
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

    COLORS <- brewer.pal(9, "Set1")

    plot(xvar, x$coef[1, ], type="n", xlab=xlab,
         ylim=range(x$coef),
         xlim=if(xvar_rev) rev(range(xvar)) else range(xvar),
         ylab=expression(hat(beta)))

    if(length(colnames(x$X))){
        labels <- colnames(x$X)
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
        y <- x$coef[i, ]; col <- COLORS[(x$groups[i] %% 9) + 1]
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
        legend(legend, legend=legend_labels, col=legend_colors, lty=1, lwd=2, bg="white", ...)
    }
}

el_norm <- function(x, g){
    sum(tapply(x, g, function(x) sum(abs(x)))^2)
}

#' @param x An \code{ExclusiveLassoFit_cv} object as produced by
#'    \code{\link{cv.exclusive_lasso}}.
#' @param bar.width Width of error bars
#' @export
#' @rdname cv.exclusive_lasso
#' @importFrom graphics segments points abline
plot.ExclusiveLassoFit_cv <- function(x, bar.width=0.01, ...){

    log_lambda <- log(x$lambda)
    bar.width <- bar.width * diff(range(log_lambda))

    plot(log_lambda, x$cvm,
         ylim=range(c(x$cvup, x$cvlo)),
         xlab=expression(log(lambda)),
         ylab=toupper(x$name),
         type="n", ...)

    points(log_lambda, x$cvm, pch=20, col="red")

    segments(log_lambda, x$cvlo,
             log_lambda, x$cvup,
             col="darkgrey")
    segments(log_lambda - bar.width, x$cvlo,
             log_lambda + bar.width, x$cvlo,
             col="darkgrey")
    segments(log_lambda - bar.width, x$cvup,
             log_lambda + bar.width, x$cvup,
             col="darkgrey")

    abline(v=log(x$lambda.min), lty=3)
    abline(v=log(x$lambda.1se), lty=3)
}
