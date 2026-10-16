/* Source code of Model_cosine class */

#include "Model_cosine.h"

#include <cmath>
#include <armadillo>
// IMPORTANT NOTICE:
// FOR THE MOMENT WE LEAVE THE FIT() METHODS UNUSED TO MAKE THE CODE AS SIMPLE AS POSSIBLE


Model_cosine::Model_cosine(const arma::Mat<double>& data):
                                Model_Base(data.n_rows, data.n_cols)
                                ,y(data)
                                ,segment_length{0}
                                ,segment_cost{0}
                                {
    // Gram matrix cached like the Python CostCosine.gram property
    // (costcosine.py): squareform(1 - cosine distance). squareform sets the
    // diagonal to 0, so the Gram diagonal is 0, not 1.
    arma::Col<double> norms = arma::sqrt(arma::sum(arma::square(y), 1));
    gram = (y * y.t()) / (norms * norms.t());
    gram.diag().zeros();
};

// Cost of rows [start, end) = Python CostCosine.error (costcosine.py):
// sum of the Gram diagonal minus the Gram sum over the segment.
double Model_cosine::compute_segment_cost(arma::uword start, arma::uword end) {
    segment_length = end - start;
    sub_gram = gram.submat(start, start, end-1, end-1);
    segment_cost = arma::accu(sub_gram.diag()) - arma::accu(sub_gram) / segment_length;
    return segment_cost;
}