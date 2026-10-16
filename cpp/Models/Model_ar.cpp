/* Source code of Model_ar class */

// The lag design matrix is rank-deficient for (near-)constant segments; solve
// then falls back to the minimum-norm least-squares solution, like numpy's
// lstsq on the Python side. This suppresses armadillo's per-call warning.
#define ARMA_WARN_LEVEL 0

#include "Model_ar.h"

#include <cmath>
#include <armadillo>
// IMPORTANT NOTICE:
// FOR THE MOMENT WE LEAVE THE FIT() METHODS UNUSED TO MAKE THE CODE AS SIMPLE AS POSSIBLE


Model_ar::Model_ar(const arma::Mat<double>& signal, const arma::Mat<double>& covar):
                                Model_Base(signal.n_rows, signal.n_cols)
                                ,y(signal.col(0))
                                ,covar(covar)
                                ,segment_length{0}
                                ,segment_cost{0}
                                {
    // Constructor body can be empty since all initialization is done in the initializer list.
};

// Cost of rows [start, end) = Python CostAR.error (costs/costautoregressive.py):
// residual sum of squares of the least-squares fit of the segment signal on
// the segment covariates (lagged values + intercept).
double Model_ar::compute_segment_cost(arma::uword start, arma::uword end) {
    sub_y = y.subvec(start, end-1);
    sub_covar = covar.rows(start, end-1);
    segment_solution = arma::solve(sub_covar, sub_y);
    residuals = sub_y -  sub_covar * segment_solution;
    segment_cost = arma::dot(residuals, residuals);
    return segment_cost;
}