/* Source code of Model_cosine class */

#include "Model_cosine.h"

#include <cmath>
#include <limits>
#include <armadillo>
// IMPORTANT NOTICE:
// FOR THE MOMENT WE LEAVE THE FIT() METHODS UNUSED TO MAKE THE CODE AS SIMPLE AS POSSIBLE


Model_cosine::Model_cosine(const arma::Mat<double>& data):
                                Model_Base(data.n_rows, data.n_cols)
                                ,y(data)
                                ,unit(data.n_rows, data.n_cols)
                                ,unit_prefix(nobs + 1, data.n_cols)
                                ,zero_prefix(nobs + 1)
                                ,segment_length{0}
                                ,segment_cost{0}
                                {
    // Rows scaled to unit norm, and their cumulative sums, which is all the
    // cost needs (see the class comment in the header). The Python
    // CostCosine.gram is squareform(1 - cosine distance), whose diagonal
    // squareform sets to 0 -- hence the "- L" in the identity, which the
    // 1 - ||sum||^2 / L form below already accounts for.
    arma::Col<double> norms = arma::sqrt(arma::sum(arma::square(y), 1));
    unit = y.each_col() / norms;
    unit_prefix.zeros();
    zero_prefix.zeros();
    for (arma::uword i = 0; i < nobs; ++i) {
        zero_prefix(i + 1) = zero_prefix(i);
        if (norms(i) == 0.0) {
            // A zero-norm row makes every cosine in its row and column NaN in
            // Python, so only the segments that contain it may be NaN. Keep the
            // cumulative sums finite by zeroing the row here, and put the NaN
            // back, segment by segment, in compute_segment_cost.
            unit.row(i).zeros();
            zero_prefix(i + 1) += 1.0;
        }
        unit_prefix.row(i + 1) = unit_prefix.row(i) + unit.row(i);
    }
};

// Cost of rows [start, end) = Python CostCosine.error (costcosine.py):
// sum of the Gram diagonal minus the Gram sum over the segment, evaluated as
// 1 - ||sum of the unit rows||^2 / L, which is the same number.
double Model_cosine::compute_segment_cost(arma::uword start, arma::uword end) {
    segment_length = end - start;
    if (zero_prefix(end) != zero_prefix(start)) {
        // zero-norm row inside the segment: NaN, like the Python cost
        segment_cost = std::numeric_limits<double>::quiet_NaN();
        return segment_cost;
    }
    arma::rowvec segment_sum = unit_prefix.row(end) - unit_prefix.row(start);
    segment_cost = 1.0 - arma::dot(segment_sum, segment_sum) / segment_length;
    return segment_cost;
}