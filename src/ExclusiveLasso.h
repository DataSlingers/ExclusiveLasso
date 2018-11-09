// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#include <limits>

#define EXLASSO_CHECK_USER_INTERRUPT_RATE 50
#define EXLASSO_CHECK_USER_INTERRUPT_RATE_GLM 10
#define EXLASSO_MAX_ITERATIONS_PROX 100
#define EXLASSO_MAX_ITERATIONS_PG 50000
#define EXLASSO_MAX_ITERATIONS_CD 10000
#define EXLASSO_MAX_ITERATIONS_GLM 500
#define EXLASSO_PREALLOCATION_FACTOR 10
#define EXLASSO_RESIZE_FACTOR 2
#define EXLASSO_FULL_LOOP_FACTOR 10
#define EXLASSO_FULL_LOOP_MIN 2
#define EXLASSO_GLM_FAMILY_GAUSSIAN 0
#define EXLASSO_GLM_FAMILY_LOGISTIC 1
#define EXLASSO_GLM_FAMILY_POISSON  2
#define EXLASSO_BACKTRACK_ALPHA 0.5
#define EXLASSO_BACKTRACK_BETA 0.8
#define EXLASSO_INF std::numeric_limits<double>::infinity()

// We only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
