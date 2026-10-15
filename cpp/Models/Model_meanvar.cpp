
/* Source code of Model_meanvar class */

#include "Model_meanvar.h"

#include <cmath>
#include <armadillo>
// IMPORTANT NOTICE:
// FOR THE MOMENT WE LEAVE THE FIT() METHODS UNUSED TO MAKE THE CODE AS SIMPLE AS POSSIBLE


Model_meanvar::Model_meanvar(const arma::Mat<double>& data): 
                                Model_Base(data.n_rows, data.n_cols)
                                ,y(data)
                                ,segment_length{0}
                                ,segment_cost{0}
                                ,eps(1e-6)
                                {
    // Constructor body can be empty since all initialization is done in the initializer list.
};

// Effective cost in the interval (start, end] = y_{start+1}, ... , y_{end}, 
// for 1-indexed points, meaning that we think of data as 
// y = {y_1, y_2,..., y_nobs} and not {y_0, y_1,..., y_{nobs-1}}.
double Model_meanvar::compute_segment_cost(arma::uword start, arma::uword end) {
    sub_y = y.subvec(start, end-1);
    segment_length = end - start;
    
    if (ndims == 1){
        cov = arma::var(sub_y) + eps;
    }
    else{
        cov = arma::cov(sub_y);
        cov = cov.for_each( [this](double& val) { val += this->eps; } );
    }
    val = std::log(std::abs(arma::det(cov)));
    segment_cost = segment_length * val;
    return segment_cost;
}
