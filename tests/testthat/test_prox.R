context("Proximal operator works")

prox <- function(x, ...) {
    as.vector(ExclusiveLasso:::exclusive_lasso_prox(x, ...,
                                                    upper_bound = rep(Inf, length(x)),
                                                    lower_bound = rep(-Inf, length(x))))
}

norm_1 <- function(x) sum(abs(x))

check_kkt <- function(x, prox_x, g, lambda){
    ### Check the KKT conditions for the prox operator

    ## This comes from  the subgradient equation for the Exclusive Lasso
    ## and setting $X = I$
    ##
    ## -(x - prox_x) + lambda * z = 0
    ## => z = (x - prox_x) / lambda
    ##    where
    ##    z_{g, i} = sign(prox_x_{g, i}) |prox_x_g|_1 if prox_x_{g, i} \neq 0
    ##             \in [-|prox_x_g|_1, |prox_x_g|_1]  if z_{g, i} = 0
    ##
    ## so we check if prox_x == x - lambda z in the prox_x != 0 case
    ## and do a magnitude check in the prox_x == 0 case

    for(ix in seq_along(x)){
        if(prox_x[ix] != 0){
            expect_equal(prox_x[ix], x[ix] - lambda *  sign(x[ix]) * norm_1(prox_x[g == g[ix]]),
                        label=paste("Prox KKT Check: element", ix, "; non-zero subgradient check"))
        } else {
            expect_true(abs(x[ix]/lambda) <= norm_1(prox_x[g == g[ix]]),
                        label=paste("Prox KKT Check: element", ix, "; zero subgradient check"))
        }
    }
}

test_that("prox reduces to shrinkage (l2 prox) when no group structure", {
    set.seed(100)

    n <- 250
    x <- rnorm(n)
    lambda <- 1

    expect_equal(prox(x, seq.int(0, n-1), lambda, 1e-8), x / (1 + lambda))
})

test_that("prox with lambda=0 is a no-op", {
    set.seed(200)

    n <- 250
    x <- rnorm(n)
    lambda <- 0

    expect_equal(prox(x, seq.int(0, n-1), lambda, 1e-8), x)
})

test_that("prox separates over groups", {
    set.seed(300)

    n <- 20
    x <- rnorm(n)
    lambda <- 0.4

    g <- rep(c(0L, 1L), each=10L)

    expect_equal(prox(x, g, lambda, 1e-8),
                 c(prox(x[g == 0L], g[g == 0L], lambda, 1e-8),
                   prox(x[g == 1L], g[g == 1L] - 1L, lambda, 1e-8)))
})

test_that("prox doesn't change the sign of x",{
    set.seed(400)

    n <- 20
    x <- rnorm(n)
    lambda <- 1.2

    g <- rep.int(seq.int(0L, 4L), 4L)
    prox_x <- prox(x, g, lambda, 1e-8)

    expect_true(all((sign(prox_x) == sign(x)) | (sign(prox_x) == 0)))
})

test_that("prox passes KKT checks -- high lambda", {
    set.seed(500)
    n <- 20
    x <- rnorm(n)
    lambda <- 30

    g <- rep.int(seq.int(0L, 4L), 4L)

    prox_x <- prox(x, g, lambda, 1e-8)

    check_kkt(x, prox_x, g, lambda)
})

test_that("prox passes KKT checks -- low lambda", {
    set.seed(600)
    n <- 20
    x <- rexp(n) + 1
    lambda <- 0.05

    g <- rep.int(seq.int(0L, 4L), 4L)

    prox_x <- prox(x, g, lambda, 1e-8)

    expect_true(all(prox_x != 0))
    check_kkt(x, prox_x, g, lambda)
})

test_that("prox passes KKT checks -- moderate lambda", {
    set.seed(700)
    n <- 20
    x <- rt(n, 4)
    lambda <- 1.2

    g <- rep.int(seq.int(0L, 4L), 4L)

    prox_x <- prox(x, g, lambda, 1e-8)

    check_kkt(x, prox_x, g, lambda)
})

test_that("prox never is fully sparse for any group", {
    set.seed(800)
    n <- 20
    x <- rnorm(n)
    lambda <- 90000

    g <- rep.int(seq.int(0L, 4L), 4L)

    prox_x <- prox(x, g, lambda, 1e-8)

    expect_false(all(prox_x == 0))
    expect_true(all(unique(g) %in% g[prox_x != 0]))
})

test_that("negative group IDs are supported", {
    set.seed(900)

    n <- 200L
    G <- 50L
    x <- rnorm(n)
    g <- rep.int(seq.int(0L, G - 1), n / G)

    expect_equal(prox(x, g, 3), prox(x, -g, 3))
})
