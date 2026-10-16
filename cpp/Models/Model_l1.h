/*
Model_l1 is an implementation of the Model_Base that computes the least absolute deviation cost over a segment (Python CostL1, costs/costl1.py).
Derived from Model_Base class, it will have an compute_segment_cost() member function to compute the cost over a segment.
*/

#ifndef MODEL_L1_H
#define MODEL_L1_H

#include "Model_Base.h"

#include <armadillo>

/**
* @class Model_l1
* @brief Model_l1 models signals that are piecewise constant, with the median fitted on each segment.
*
* The cost is the sum of absolute deviations from the per-dimension median.
*
*/
class Model_l1 : public Model_Base {
    public :
        Model_l1(const arma::Mat<double>& data);
        double compute_segment_cost(arma::uword start, arma::uword end);

    private :
        // member variables to compute cost in compute_segment_cost()
        // by being member variables, we avoid re-declaring them at each
        // compute_segment_cost call
        arma::Mat<double> y;

        arma::Mat<double> sub_y;
        arma::Row<double> med;
        double segment_cost;
    };
#endif // MODEL_L1_H