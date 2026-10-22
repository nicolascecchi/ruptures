
/* Source code of Model_constant_rank class */

#include "Model_constant_rank.h"

#include <cmath>
#include <armadillo>
// IMPORTANT NOTICE:
// FOR THE MOMENT WE LEAVE THE FIT() METHODS UNUSED TO MAKE THE CODE AS SIMPLE AS POSSIBLE
// NOTE: compute_centered_ranks (below) was modified to average the ranks of
// tied values, like scipy's rankdata does for Python's CostRank. Without
// this, tied columns (e.g. a constant signal) get an arbitrary permutation of
// the ranks and their cost becomes very negative instead of 0, so PELT
// returns spurious change points.


Model_constant_rank::Model_constant_rank(const arma::Mat<double>& data): 
                                Model_Base(data.n_rows, data.n_cols)
                                ,centered_ranks{compute_centered_ranks(data)}
                                ,cusum_ranks{compute_cusum_ranks(centered_ranks)}
                                ,inv_cov{compute_inv_cov(centered_ranks)}
                                ,segment_length{0}
                                ,dim_cost(ndims, arma::fill::zeros)
                                ,segment_cost{0}
                                ,segment_dim_mean(ndims, arma::fill::zeros)
                                {
    // Constructor body can be empty since all initialization is done in the initializer list.
};

// Effective cost in the interval (start, end] = y_{start+1}, ... , y_{end}, 
// for 1-indexed points, meaning that we think of data as 
// y = {y_1, y_2,..., y_nobs} and not {y_0, y_1,..., y_{nobs-1}}.
double Model_constant_rank::compute_segment_cost(arma::uword start, arma::uword end) {
    segment_length = end - start; // number of points len = t - (s+1) -1 = t - s
    segment_cost = 0;
    for (arma::uword d = 0; d < ndims; d++) {
        segment_dim_mean(d) = (cusum_ranks(end, d) - cusum_ranks(start, d)) / segment_length;
    }
    segment_cost = -(segment_length)* arma::as_scalar(segment_dim_mean.t() * inv_cov * segment_dim_mean);
    return segment_cost;
}

const arma::Mat<double> Model_constant_rank::compute_centered_ranks(const arma::Mat<double>& data) {
    arma::Mat<double> tmp_centered_ranks(data.n_rows, data.n_cols, arma::fill::zeros);
    // Compute ranks for each column; tied values take the average of their
    // ranks (scipy rankdata's default), which is the same as an ordinal rank
    // when values are all distinct.
    for (arma::uword d = 0; d < ndims; ++d) {
        arma::uvec sorted_indices = arma::sort_index(data.col(d));
        arma::uword i = 0;
        while (i < sorted_indices.n_elem) {
            // find the end j of the run of tied values
            arma::uword j = i;
            while (j + 1 < sorted_indices.n_elem &&
                   data(sorted_indices(j + 1), d) == data(sorted_indices(i), d)) {
                ++j;
            }
            // average of the 1-based ranks i+1, ..., j+1
            double average_rank = (i + j + 2.0) / 2.0;
            for (arma::uword k = i; k <= j; ++k) {
                tmp_centered_ranks(sorted_indices(k), d) = average_rank;
            }
            i = j + 1;
        }
    }
    // Centering ranks
    tmp_centered_ranks.for_each([this] (double& val) {val -= ((nobs + 1) / 2.0);}) ;
    return tmp_centered_ranks;
}

const arma::Mat<double> Model_constant_rank::compute_inv_cov(const arma::Mat<double>& centered_ranks) {
    auto cov = arma::cov(centered_ranks, 1); // norm_type = 1 computes cov dividing by N (biased estimator)
    auto inv_cov = arma::pinv(cov);       
    return inv_cov;
}

const arma::Mat<double> Model_constant_rank::compute_cusum_ranks(const arma::Mat<double>& centered_ranks) {
    arma::Mat<double> cusum_ranks(centered_ranks.n_rows + 1, centered_ranks.n_cols, arma::fill::zeros);
    for (arma::uword d = 0; d < centered_ranks.n_cols; ++d) {
        // place the cumulative sum of column d into rows 1..n (leave row 0 as zero)
        cusum_ranks.col(d).rows(1, centered_ranks.n_rows) = arma::cumsum(centered_ranks.col(d));
    }
    return cusum_ranks;
}

//Model_constant_rank::Model_constant_rank(arma::Mat<double> data_) : Cost_Base(std::move(data_)){};
// void Model_constant_rank::fit(arma::Mat<double>& data){
//     int ncols = data.n_cols;
//     
//     // https://arma.sourceforge.net/docs.html#cumsum
//     // index 0 sums the rows on each column
//     cumsum_y = arma::join_vert(
//         arma::zeros(1, ncols),
//         arma::cumsum(data, 0)
//     );
//     cumsum_squared_y = arma::join_vert(
//         arma::zeros(1, ncols),
//         arma::cumsum(arma::square(data), 0));
// };
