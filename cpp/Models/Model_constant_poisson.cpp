
/* Source code of Model_constant_poisson class */

#include "Model_constant_poisson.h"

#include <cmath>
#include <armadillo>
// IMPORTANT NOTICE:
// FOR THE MOMENT WE LEAVE THE FIT() METHODS UNUSED TO MAKE THE CODE AS SIMPLE AS POSSIBLE


Model_constant_poisson::Model_constant_poisson(const arma::Mat<double>& data): 
                                Model_Base(data.n_rows, data.n_cols)
                                ,cumsum_y{arma::join_vert(
                                    arma::zeros(1, ndims)
                                    ,arma::cumsum(data, 0))
                                }
                                ,segment_length{0}
                                ,dim_cost(ndims, arma::fill::zeros)
                                ,segment_cost{0}
                                {
    // Constructor body can be empty since all initialization is done in the initializer list.
};

// Effective cost in the interval (start, end] = y_{start+1}, ... , y_{end}, 
// for 1-indexed points, meaning that we think of data as 
// y = {y_1, y_2,..., y_nobs} and not {y_0, y_1,..., y_{nobs-1}}.
double Model_constant_poisson::compute_segment_cost(arma::uword start, arma::uword end) {
    const double segment_length = end - start; // number of points len = t - (s+1) -1 = t - s
    double segment_cost = 0.0;
    for (arma::uword d = 0; d < ndims; ++d) {
        const double sum_y  = cumsum_y(end, d) - cumsum_y(start, d);
        // An all-zero dimension has cost 0 in the mathematical limit
        // (0 * log(0) is undefined), so the guard keeps the compiled cost
        // finite for valid count data.
        if (sum_y > 0.0) {
            segment_cost +=  sum_y - sum_y * std::log(sum_y / segment_length) ;
        }
    }
    return segment_cost;
}

