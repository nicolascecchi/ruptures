
/* Source code of Model_constant_binomial class */

#include "Model_constant_binomial.h"

#include <cmath>
#include <armadillo>
// IMPORTANT NOTICE:
// FOR THE MOMENT WE LEAVE THE FIT() METHODS UNUSED TO MAKE THE CODE AS SIMPLE AS POSSIBLE


Model_constant_binomial::Model_constant_binomial(const arma::Mat<double>& data
                                                , int m): 
                                Model_Base(data.n_rows, data.n_cols)
                                ,cumsum_y{arma::join_vert(
                                    arma::zeros(1, ndims)
                                    ,arma::cumsum(data, 0))
                                }
                                ,segment_length{0}
                                ,dim_cost(ndims, arma::fill::zeros)
                                ,segment_cost{0}
                                ,m{m}
                                {
    // Constructor body can be empty since all initialization is done in the initializer list.
};

// Effective cost in the interval (start, end] = y_{start+1}, ... , y_{end}, 
// for 1-indexed points, meaning that we think of data as 
// y = {y_1, y_2,..., y_nobs} and not {y_0, y_1,..., y_{nobs-1}}.
double Model_constant_binomial::compute_segment_cost(arma::uword start, arma::uword end) {
    const double segment_length = end - start; // number of points len = t - (s+1) -1 = t - s
    double segment_cost = 0.0;

    for (arma::uword d = 0; d < ndims; ++d) {
        const double sum_y  = cumsum_y(end, d) - cumsum_y(start, d);
        segment_phat = sum_y/(segment_length * m); 
        segment_cost +=   -(std::log(segment_phat)*sum_y + 
                            std::log(1-segment_phat)*(segment_length * m - sum_y));
    }
    return segment_cost;
}


arma::Mat<double> Model_constant_binomial::get_cumsum_y() const {return cumsum_y;};
