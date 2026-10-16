/* Source code of Model_l1 class */

#include "Model_l1.h"

#include <cmath>
#include <armadillo>
// IMPORTANT NOTICE:
// FOR THE MOMENT WE LEAVE THE FIT() METHODS UNUSED TO MAKE THE CODE AS SIMPLE AS POSSIBLE


Model_l1::Model_l1(const arma::Mat<double>& data):
                                Model_Base(data.n_rows, data.n_cols)
                                ,y(data)
                                ,segment_cost{0}
                                {
    // Constructor body can be empty since all initialization is done in the initializer list.
};

// Cost of rows [start, end) = Python CostL1.error (costs/costl1.py):
// sum of absolute deviations from the per-column median.
double Model_l1::compute_segment_cost(arma::uword start, arma::uword end) {
    sub_y = y.submat(start, 0, end-1, ndims-1);
    // per-column medians; even-length columns take the mean of the two
    // central values, like numpy
    med.set_size(ndims);
    for (arma::uword d = 0; d < ndims; ++d) {
        med(d) = arma::median(sub_y.col(d));
    }
    segment_cost = arma::accu(arma::abs(sub_y.each_row() - med));
    return segment_cost;
}