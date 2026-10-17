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
* No Gram matrix is needed. The cosine Gram has a zero diagonal and exact
* cosines off it, so the sum of its entries over a segment is
* ||sum of the unit rows||^2 - L, which makes the cost 1 - ||sum||^2 / L: the
* same number as the Python CostCosine's diagonal-minus-block-sum, at O(nobs)
* memory and O(n_features) per call instead of O(nobs^2) and O(L^2).
* Rows with zero norm give NaN in Python, for the segments that contain them;
* zero_prefix keeps that behaviour instead of letting NaN spread through the
* cumulative sums.
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
        arma::Mat<double> unit;         /**< y with unit-norm rows */
        arma::Mat<double> unit_prefix;  /**< unit_prefix.row(i) = sum of unit rows before i */
        arma::Col<double> zero_prefix;  /**< zero_prefix(i) = number of zero-norm rows before i */

        double segment_length;
        double segment_cost;
    };
#endif // MODEL_COSINE_H