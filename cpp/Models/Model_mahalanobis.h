/*
Model_mahalanobis is an implementation of the Model_Base that computes a Mahalanobis-type kernelized cost (Python CostMl, costs/costml.py).
Derived from Model_Base class, it will have an compute_segment_cost() member function to compute the cost over a segment.

The metric argument is the metric matrix prepared by CostMl.fit() (inverse of
the covariance when no custom metric was given): the numpy construction lives
in Python, the cost here. The cost sums the Gram diagonal and the Gram sum
algebraically, so no O(nobs^2) Gram matrix is needed.
*/

#ifndef MODEL_MAHALANOBIS_H
#define MODEL_MAHALANOBIS_H

#include "Model_Base.h"

#include <armadillo>

/**
* @class Model_mahalanobis
* @brief Model_mahalanobis models signals with a Mahalanobis-type kernel cost.
*
*/
class Model_mahalanobis : public Model_Base {
    public :
        Model_mahalanobis(const arma::Mat<double>& data, const arma::Mat<double>& metric);
        double compute_segment_cost(arma::uword start, arma::uword end);

    private :
        // member variables to compute cost in compute_segment_cost()
        // by being member variables, we avoid re-declaring them at each
        // compute_segment_cost call
        // caches carry a leading zero row: cumsum_cache(end) - cumsum_cache(start)
        // is then the sum over rows [start, end), same as Model_constant_normal.
        arma::Mat<double> y;
        const arma::Mat<double> metric;
        const arma::Mat<double> cumsum_quad;  // cumulative sum of y_i^T M y_i (quad forms)
        const arma::Mat<double> cumsum_y;     // cumulative sums of the data

        arma::Row<double> sums;
        int segment_length;
        double segment_cost;
    };
#endif // MODEL_MAHALANOBIS_H