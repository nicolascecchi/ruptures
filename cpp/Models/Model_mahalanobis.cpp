/* Source code of Model_mahalanobis class */

#include "Model_mahalanobis.h"

#include <cmath>
#include <armadillo>
// IMPORTANT NOTICE:
// FOR THE MOMENT WE LEAVE THE FIT() METHODS UNUSED TO MAKE THE CODE AS SIMPLE AS POSSIBLE


Model_mahalanobis::Model_mahalanobis(const arma::Mat<double>& data, const arma::Mat<double>& metric_):
                                Model_Base(data.n_rows, data.n_cols)
                                ,y(data)
                                ,metric(metric_)
                                ,cumsum_quad{arma::join_vert(
                                    arma::zeros<arma::Mat<double>>(1, 1)
                                    ,arma::cumsum(arma::sum((y * metric) % y, 1))
                                )}
                                ,cumsum_y{arma::join_vert(
                                    arma::zeros<arma::Mat<double>>(1, ndims)
                                    ,arma::cumsum(data, 0))
                                }
                                ,segment_length{0}
                                ,segment_cost{0}
                                {
    // Constructor body can be empty since all initialization is done in the initializer list.
};

// Cost of rows [start, end) = Python CostMl.error (costml.py), i.e.
// sum of the Gram diagonal minus the Gram sum over the segment, on the Gram
// matrix y M y^T. Both sums are computed algebraically:
// diagonal -> cumulative sum of the quad forms y_i^T M y_i,
// full sum -> (column sums of the segment)^T M (column sums of the segment).
double Model_mahalanobis::compute_segment_cost(arma::uword start, arma::uword end) {
    segment_length = end - start;
    const double sum_quad = cumsum_quad(end) - cumsum_quad(start);
    sums = cumsum_y.row(end) - cumsum_y.row(start);
    segment_cost = sum_quad - arma::as_scalar(sums * metric * sums.t()) / segment_length;
    return segment_cost;
}