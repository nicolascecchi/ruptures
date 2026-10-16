/*
Model_ar is an implementation of the Model_Base that computes the least-squares autoregressive cost over a segment (Python CostAR, costs/costautoregressive.py).
Derived from Model_Base class, it will have an compute_segment_cost() member function to compute the cost over a segment.

The signal and covar constructors' arguments are exactly the products of
CostAR.fit() (the signal with its first `order` points replaced by the one at
index `order`, and the lagged covariates with intercept column): the numpy
stride-trick construction lives in Python, the least-squares cost here.
*/

#ifndef MODEL_AR_H
#define MODEL_AR_H

#include "Model_Base.h"

#include <armadillo>

/**
* @class Model_ar
* @brief Model_ar models signals that are piecewise autoregressive with Gaussian noise.
*
* The cost is the residual sum of squares of a least-squares fit of the
* segment signal on the segment covariates.
*
*/
class Model_ar : public Model_Base {
    public :
        Model_ar(const arma::Mat<double>& signal, const arma::Mat<double>& covar);
        double compute_segment_cost(arma::uword start, arma::uword end);

    private :
        // member variables to compute cost in compute_segment_cost()
        // by being member variables, we avoid re-declaring them at each
        // compute_segment_cost call
        arma::Col<double> y;
        arma::Mat<double> covar;

        int segment_length;
        arma::Col<double> sub_y;
        arma::Mat<double> sub_covar;
        arma::Col<double> residuals;
        arma::Col<double> segment_solution;
        double segment_cost;
    };
#endif // MODEL_AR_H