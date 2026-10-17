/*
Model_normal is an implementation of the Model_Base that computes the Python
CostNormal cost (costs/costnormal.py): the Gaussian log-likelihood cost
segment_length * log|det(cov)|, where cov is the segment's covariance.
Python's v1.1.5 behaviour (add_small_diag=True, PR#198) always adds a 1e-6
diagonal bias so constant segments stay positive definite.
*/

#ifndef MODEL_NORMAL_H
#define MODEL_NORMAL_H

#include "Model_Base.h"

#include <armadillo>

/**
* @class Model_normal
* @brief Model_normal models signals whose segments are Gaussian.
*
*/
class Model_normal : public Model_Base {
    public :
        Model_normal(const arma::Mat<double>& data);
        double compute_segment_cost(arma::uword start, arma::uword end);

    private :
        // member variables to compute cost in compute_segment_cost()
        // by being member variables, we avoid re-declaring them at each
        // compute_segment_cost call
        arma::Mat<double> y;

        double segment_length;
        arma::Mat<double> deviations;
        arma::Mat<double> cov;
        double segment_cost;
    };
#endif // MODEL_NORMAL_H