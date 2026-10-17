/*
Model_rbf is an implementation of the Model_Base that computes a kernelized cost with an RBF (Gaussian) kernel (Python CostRbf, costs/costrbf.py).
Derived from Model_Base class, it will have an compute_segment_cost() member function to compute the cost over a segment.
*/

#ifndef MODEL_RBF_H
#define MODEL_RBF_H

#include "Model_Base.h"

#include <armadillo>

/**
* @class Model_rbf
* @brief Model_rbf models signals with a Gaussian-kernel (RBF) cost.
*
* The gamma parameter is the RBF width (Python CostRbf fits the median
* heuristic itself, at Python level, and passes it here). The Gram matrix
* exp(-clip(gamma * squared distance, 1e-2, 1e2)) is computed once, like the
* cached Python CostRbf.gram; memory is O(nobs^2), same as Python.
*
*/
class Model_rbf : public Model_Base {
    public :
        Model_rbf(const arma::Mat<double>& data, double gamma);
        double compute_segment_cost(arma::uword start, arma::uword end);

    private :
        // member variables to compute cost in compute_segment_cost()
        // by being member variables, we avoid re-declaring them at each
        // compute_segment_cost call
        arma::Mat<double> y;
        double gamma;
        arma::Mat<double> gram;

        double segment_length;
        double segment_cost;
    };
#endif // MODEL_RBF_H