/* Source code of Model_rbf class */

#include "Model_rbf.h"

#include <cmath>
#include <armadillo>
// IMPORTANT NOTICE:
// FOR THE MOMENT WE LEAVE THE FIT() METHODS UNUSED TO MAKE THE CODE AS SIMPLE AS POSSIBLE


Model_rbf::Model_rbf(const arma::Mat<double>& data, double gamma):
                                Model_Base(data.n_rows, data.n_cols)
                                ,y(data)
                                ,gamma{gamma}
                                ,segment_length{0}
                                ,segment_cost{0}
                                {
    // Gram matrix cached like the Python CostRbf.gram property (costrbf.py):
    // exp(-clip(gamma * squared euclidean distance, 1e-2, 1e2)).
    // Squared pairwise distances via ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j;
    // near-zero entries are clamped anyway, so the round-off is harmless.
    arma::Col<double> sq = arma::sum(arma::square(y), 1);
    arma::Mat<double> sqdist = sq * arma::ones<arma::Mat<double>>(1, nobs)
                                + arma::ones<arma::Mat<double>>(nobs, 1) * sq.t()
                                - 2 * y * y.t();
    // clip gamma * d^2 as Python does (costrbf.py clips the condensed pdist
    // vector, so the diagonal is never clipped: exp(0) = 1)
    arma::Mat<double> clipped = arma::clamp(gamma * sqdist, 1e-2, 1e2);
    clipped.diag().zeros();
    gram = arma::exp(-clipped);
};

// Cost of rows [start, end) = Python CostRbf.error (costrbf.py):
// sum of the Gram diagonal minus the Gram sum over the segment.
double Model_rbf::compute_segment_cost(arma::uword start, arma::uword end) {
    segment_length = end - start;
    sub_gram = gram.submat(start, start, end-1, end-1);
    segment_cost = arma::accu(sub_gram.diag()) - arma::accu(sub_gram) / segment_length;
    return segment_cost;
}