
/* Source code of Model_linear class */

#include "Model_linear.h"

#include <cmath>
#include <armadillo>
// IMPORTANT NOTICE:
// FOR THE MOMENT WE LEAVE THE FIT() METHODS UNUSED TO MAKE THE CODE AS SIMPLE AS POSSIBLE


Model_linear::Model_linear(const arma::Mat<double>& data): 
                                Model_Base(data.n_rows, data.n_cols)
                                ,segment_length{0}
                                ,y(data.col(0))
                                ,covariates(data.cols(1, data.n_cols-1))
                                ,segment_cost{0}
                                {
    // Constructor body can be empty since all initialization is done in the initializer list.
};

// Effective cost in the interval (start, end] = y_{start+1}, ... , y_{end}, 
// for 1-indexed points, meaning that we think of data as 
// y = {y_1, y_2,..., y_nobs} and not {y_0, y_1,..., y_{nobs-1}}.
double Model_linear::compute_segment_cost(arma::uword start, arma::uword end) {
    sub_y = y.subvec(start, end-1);
    sub_covariates = covariates.rows(start, end-1);
    segment_solution = arma::solve(sub_covariates, sub_y);
    residuals = sub_y -  sub_covariates * segment_solution;
    segment_cost = arma::dot(residuals, residuals);
    return segment_cost;
}
