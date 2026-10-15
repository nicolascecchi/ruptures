
/* Source code of Model_constant_normal class */

#include "Model_constant_normal.h"

#include <cmath>
#include <armadillo>
// IMPORTANT NOTICE:
// FOR THE MOMENT WE LEAVE THE FIT() METHODS UNUSED TO MAKE THE CODE AS SIMPLE AS POSSIBLE


Model_constant_normal::Model_constant_normal(const arma::Mat<double>& data): 
                                Model_Base(data.n_rows, data.n_cols)
                                ,cumsum_y{arma::join_vert(
                                    arma::zeros(1, ndims)
                                    ,arma::cumsum(data, 0))
                                }
                                ,cumsum_squared_y{arma::join_vert(
                                                    arma::zeros(1, ndims),
                                                    arma::cumsum(arma::square(data), 0))}
                                ,segment_length{0}
                                ,dim_cost(ndims, arma::fill::zeros)
                                ,segment_cost{0}
                                {
    // Constructor body can be empty since all initialization is done in the initializer list.
};

// Effective cost in the interval (start, end] = y_{start+1}, ... , y_{end}, 
// for s  = 0 ... t-1
// meaning that we think of data as 
// y = {y_1, y_2,..., y_nobs} and not {y_0, y_1,..., y_{nobs-1}}.
double Model_constant_normal::compute_segment_cost(arma::uword start, arma::uword end) {
    const double segment_length = end - start; // number of points len = t - (s+1) -1 = t - s
    double segment_cost = 0.0;
    for (arma::uword d = 0; d < ndims; ++d) {
        const double sum_y  = cumsum_y(end, d) - cumsum_y(start, d);
        const double sum_y2 = cumsum_squared_y(end, d) - cumsum_squared_y(start, d);
        segment_cost += sum_y2 - (sum_y * sum_y) / segment_length;
    }
    return segment_cost;
}


arma::Mat<double> Model_constant_normal::get_cumsum_y() const {return cumsum_y;};
arma::Mat<double> Model_constant_normal::get_cumsum_squared_y() const {return cumsum_squared_y;};

//Model_constant_normal::Model_constant_normal(arma::Mat<double> data_) : Cost_Base(std::move(data_)){};
// void Model_constant_normal::fit(arma::Mat<double>& data){
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
