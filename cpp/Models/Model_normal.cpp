/* Source code of Model_normal class */

#include "Model_normal.h"

#include <armadillo>

Model_normal::Model_normal(const arma::Mat<double>& data):
                                Model_Base(data.n_rows, data.n_cols)
                                ,y{data}
                                ,segment_length{0}
                                ,cov(1, 1)
                                ,segment_cost{0}
                                {
    // Constructor body can be empty since all initialization is done in the initializer list.
};

// The cost replicates Python's CostNormal.error (costnormal.py): the
// Gaussian log-likelihood segment_length * log|det(cov)| with a 1e-6
// diagonal bias always added (add_small_diag=True behaviour). The covariance
// divisor follows Python branch by branch: the maximum-likelihood variance
// (ddof=0) in 1-D, and the unbiased divisor (ddof=1) of np.cov with several
// dimensions.
double Model_normal::compute_segment_cost(arma::uword start, arma::uword end) {
    segment_length = end - start;
    if (ndims == 1) {
        arma::Col<double> sub_y = y.col(0).rows(start, end - 1);
        double variance = arma::sum(arma::square(sub_y - arma::mean(sub_y))) / segment_length;
        segment_cost = segment_length * std::log(variance + 1e-6);
    } else {
        arma::Mat<double> sub_y = y.submat(start, 0, end - 1, ndims - 1);
        deviations = sub_y.each_row() - arma::mean(sub_y, 0);
        cov = deviations.t() * deviations / (segment_length - 1.0);
        cov.diag() += 1e-6;
        segment_cost = segment_length * arma::log_det(cov).real();
    }
    return segment_cost;
};