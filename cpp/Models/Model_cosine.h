/*
Model_cosine is an implementation of the Model_Base that computes a kernelized cost with the cosine similarity kernel (Python CostCosine, costs/costcosine.py).
Derived from Model_Base class, it will have an compute_segment_cost() member function to compute the cost over a segment.
*/

#ifndef MODEL_COSINE_H
#define MODEL_COSINE_H

#include "Model_Base.h"

#include <armadillo>

/**
* @class Model_cosine
* @brief Model_cosine models signals with a cosine-similarity kernel cost.
*
* The Gram matrix dot(y_i, y_j) / (|y_i| |y_j|) is computed once, like the
* cached Python CostCosine.gram; memory is O(nobs^2), same as Python. Rows
* with zero norm give NaN entries, like scipy's pdist cosine distances.
*
*/
class Model_cosine : public Model_Base {
    public :
        Model_cosine(const arma::Mat<double>& data);
        double compute_segment_cost(arma::uword start, arma::uword end);

    private :
        // member variables to compute cost in compute_segment_cost()
        // by being member variables, we avoid re-declaring them at each
        // compute_segment_cost call
        arma::Mat<double> y;
        arma::Mat<double> gram;

        arma::Mat<double> sub_gram;
        double segment_length;
        double segment_cost;
    };
#endif // MODEL_COSINE_H