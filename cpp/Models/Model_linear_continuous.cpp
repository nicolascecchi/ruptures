
/* Source code of Model_linear_continuous class */

#include "Model_linear_continuous.h"

#include <cmath>
#include <armadillo>
// IMPORTANT NOTICE:
// FOR THE MOMENT WE LEAVE THE FIT() METHODS UNUSED TO MAKE THE CODE AS SIMPLE AS POSSIBLE


Model_linear_continuous::Model_linear_continuous(const arma::Mat<double>& data): 
                                    Model_Base(data.n_rows, data.n_cols)
                                    ,segment_length{0}
                                    ,y(data)
                                    ,segment_cost{0}
                                {
    // Constructor body can be empty since all initialization is done in the initializer list.
};

// Effective cost in the interval (start, end] = y_{start+1}, ... , y_{end}, 
// for 1-indexed points, meaning that we think of data as 
// y = {y_1, y_2,..., y_nobs} and not {y_0, y_1,..., y_{nobs-1}}.
double Model_linear_continuous::compute_segment_cost(arma::uword start, arma::uword end) {
    sub_y = y.subvec(start, end-1);
    slope = (y(end-1) - y(start)) / (end - start);
    intercept = y(start);
    segment_prediction = intercept + slope * arma::regspace(0, end - start - 1);
    residuals = sub_y -  segment_prediction;
    segment_cost = arma::dot(residuals, residuals);
    return segment_cost;
}
