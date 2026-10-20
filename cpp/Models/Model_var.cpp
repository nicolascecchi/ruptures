/* Source code of Model_var class */

// The lag Gram matrix is rank-deficient for (near-)constant or short
// segments; solve then falls back to the minimum-norm least-squares solution,
// like numpy's lstsq on the Python side. This suppresses armadillo's per-call
// warning.
#define ARMA_WARN_LEVEL 0

#include "Model_var.h"

#include <armadillo>
#include <stdexcept>
// IMPORTANT NOTICE:
// FOR THE MOMENT WE LEAVE THE FIT() METHODS UNUSED TO MAKE THE CODE AS SIMPLE AS POSSIBLE

// fast/allow_ugly skip the accuracy bookkeeping of a plain solve; force_sym
// exploits the symmetry of the lag Gram matrix zz = Z'Z.
const auto kSolveOpts =
    arma::solve_opts::fast + arma::solve_opts::allow_ugly + arma::solve_opts::force_sym;

Model_var::Model_var(const arma::Mat<double>& data, arma::uword order, bool intercept):
                                Model_Base(data.n_rows, data.n_cols)
                                ,n_regs{order * data.n_cols + (intercept ? 1 : 0)}
                                ,segment_cost{0}
                                {
    if (order == 0) {
        throw std::invalid_argument("order must be positive.");
    }
    // Lag design matrix from the raw signal, then condensed into cumulative
    // statistics: row t of each table is the sum of the t first rows of the
    // underlying per-row products, so a segment's cross-products are read in
    // O(1) as row(end) - row(start) and the lag block is never touched again.
    arma::Mat<double> lag_block = make_lag_block(data, order, intercept);
    cum_zz = cumulative_row_sums(row_outer_products(lag_block, lag_block));
    cum_yz = cumulative_row_sums(row_outer_products(data, lag_block));
    cum_sum_y2 = cumulative_row_sums(arma::sum(arma::square(data), 1));
};

// Cost of rows [start, end) = residual sum of squares of the least-squares fit
// of the segment signal on the segment lag covariates. Via the normal
// equations zz B' = yz', RSS = sum_t ||y_t||^2 - <B, sum_t y_t z_t'>, so the
// fit only enters the cost through this inner product.
double Model_var::compute_segment_cost(arma::uword start, arma::uword end) {
    zz = segment_sum(cum_zz, start, end, n_regs, n_regs);
    yz = segment_sum(cum_yz, start, end, ndims, n_regs);
    segment_solution = arma::solve(zz, yz.t(), kSolveOpts).t();
    const double segment_sum_y2 = cum_sum_y2(end, 0) - cum_sum_y2(start, 0);
    segment_cost = segment_sum_y2 - arma::dot(segment_solution, yz);
    return segment_cost;
}
