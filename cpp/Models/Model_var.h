/*
Model_var is an implementation of the Model_Base that computes the multivariate
least-squares autoregressive cost over a segment: the residual sum of squares
of the segment signal regressed on its own lagged values (+ an optional
intercept), i.e. a piecewise vector autoregressive (VAR) model with Gaussian
noise. Derived from Model_Base class, it will have a compute_segment_cost()
member function to compute the cost over a segment.

It generalises Model_ar, whose univariate lagged covariates are prebuilt on the
Python side: here the lag design matrix is built in the constructor and
condensed into cumulative statistics, so a segment cost needs only O(1)
cross-product lookups and one least-squares solve.

The functional prefiltered implementation this class adapts fitted each segment
in two stages (a ridge-regularised prefilter fitted on the long-lag covariates
over the first burn_in rows, then OLS of the prefiltered signal on the main
lags); both stages and their ridge are dropped here for a plain single-stage
OLS on the main lags.
*/

#ifndef MODEL_VAR_H
#define MODEL_VAR_H

#include "Model_Base.h"

#include <armadillo>

/**
* @class Model_var
* @brief Model_var models signals that are piecewise vector autoregressive with Gaussian noise.
*
* The cost is the residual sum of squares of a least-squares fit of the
* segment signal on the segment lagged covariates.
*/
class Model_var : public Model_Base {
    public :
        Model_var(const arma::Mat<double>& data, arma::uword order, bool intercept);
        double compute_segment_cost(arma::uword start, arma::uword end);

    private :
        // member variables to compute cost in compute_segment_cost()
        // by being member variables, we avoid re-declaring them at each
        // compute_segment_cost call
        arma::uword n_regs;
        // Cumulative statistics: row t of each table is the sum of the t first
        // rows of the underlying per-row products, so a segment's
        // cross-products are read as row(end) - row(start).
        arma::Mat<double> cum_zz;     // lag covariates' Gram matrices
        arma::Mat<double> cum_yz;     // signal x lag covariates cross-products
        arma::Mat<double> cum_sum_y2; // row sums of squared signal values

        arma::Mat<double> zz;
        arma::Mat<double> yz;
        arma::Mat<double> segment_solution;
        double segment_cost;
    };


// Row t of the result is vectorise(X.row(t).t() * Y.row(t)) (column-major,
// matching arma::reshape's default layout): the flattened outer product of
// X and Y's t-th rows. Summed over rows and reshaped to
// (X.n_cols x Y.n_cols), it gives sum_t x_t y_t'.
inline arma::mat row_outer_products(const arma::mat& X, const arma::mat& Y) {
    arma::uword n_samples = X.n_rows;
    arma::mat out(n_samples, X.n_cols * Y.n_cols);
    for (arma::uword t = 0; t < n_samples; ++t) {
        out.row(t) = arma::vectorise(X.row(t).t() * Y.row(t)).t();
    }
    return out;
}

// out.row(0) is zero and out.row(t) is the sum of X.row(0..t-1) for
// t = 1..n_rows, so a segment's sum over [start, end) is
// out.row(end) - out.row(start) in O(1).
inline arma::mat cumulative_row_sums(const arma::mat& X) {
    arma::mat out(X.n_rows + 1, X.n_cols, arma::fill::zeros);
    for (arma::uword t = 0; t < X.n_rows; ++t) {
        out.row(t + 1) = out.row(t) + X.row(t);
    }
    return out;
}

// Row t is [Y.row(t - 1), ..., Y.row(t - n_lags)] (+ a trailing 1 if
// `intercept`) for t >= n_lags, and zero otherwise: the first n_lags rows
// have no lagged values yet.
inline arma::mat make_lag_block(const arma::mat& Y, arma::uword n_lags, bool intercept) {
    arma::uword n_samples = Y.n_rows;
    arma::uword n_dims = Y.n_cols;
    arma::uword n_cols = n_lags * n_dims + (intercept ? 1 : 0);

    arma::mat out(n_samples, n_cols, arma::fill::zeros);
    for (arma::uword t = n_lags; t < n_samples; ++t) {
        for (arma::uword j = 0; j < n_lags; ++j) {
            out.submat(t, j * n_dims, t, (j + 1) * n_dims - 1) = Y.row(t - 1 - j);
        }
        if (intercept) {
            out(t, n_cols - 1) = 1.0;
        }
    }
    return out;
}

// Sum of rows [start, end) of the (n_rows x n_cols) matrix whose
// cumulative_row_sums table is `cum`: reshaped difference of two cum rows.
inline arma::mat segment_sum(
    const arma::mat& cum,
    arma::uword start,
    arma::uword end,
    arma::uword n_rows,
    arma::uword n_cols) {
    return arma::reshape(cum.row(end) - cum.row(start), n_rows, n_cols);
}


#endif // MODEL_VAR_H
