/* Source code of Model_clinear class */

#include "Model_clinear.h"

#include <cmath>
#include <armadillo>
// IMPORTANT NOTICE:
// FOR THE MOMENT WE LEAVE THE FIT() METHODS UNUSED TO MAKE THE CODE AS SIMPLE AS POSSIBLE


Model_clinear::Model_clinear(const arma::Mat<double>& data):
                                Model_Base(data.n_rows, data.n_cols)
                                ,y(data)
                                ,segment_length{0}
                                ,segment_cost{0}
                                {
    // Constructor body can be empty since all initialization is done in the initializer list.
};

// Cost of rows [start, end) = Python CostCLinear.error (costs/costclinear.py):
// the line is anchored on the point BEFORE the segment (y[start-1], so for
// start == 0 the evaluated segment is [1, end)); anchors are read in
// absolute coordinates.
double Model_clinear::compute_segment_cost(arma::uword start, arma::uword end) {
    if (start == 0) {
        start = 1;
    }
    segment_length = end - start;
    segment_times = arma::regspace<arma::Col<double>>(1.0, segment_length);
    const arma::Row<double> slope = (y.row(end-1) - y.row(start-1)) / (end - start);
    const arma::Row<double> intercept = y.row(start-1);
    // (segment_length x 1) * (1 x ndims) = (segment_length x ndims);
    // each_row adds the intercept row (no mat+rowvec operator in armadillo)
    segment_prediction = segment_times * slope;
    segment_prediction.each_row() += intercept;
    residuals = y.submat(start, 0, end-1, ndims-1) - segment_prediction;
    segment_cost = arma::accu(arma::square(residuals));
    return segment_cost;
}