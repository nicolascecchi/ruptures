/*
Model_clinear is an implementation of the Model_Base that computes the Python CostCLinear cost (costs/costclinear.py): a piecewise linear approximation with a continuity constraint.
Derived from Model_Base class, it will have an compute_segment_cost() member function to compute the cost over a segment.

NOTE: this is NOT the same segmentation model as Model_linear_continuous,
which follows the biblio-these convention of anchoring the line on the
segment endpoints. Model_clinear replicates Python: the line is anchored on
the point BEFORE the segment, and anchors are read at absolute positions.
*/

#ifndef MODEL_CLINEAR_H
#define MODEL_CLINEAR_H

#include "Model_Base.h"

#include <armadillo>

/**
* @class Model_clinear
* @brief Model_clinear models signals that are piecewise linear with a continuity constraint.
*
*/
class Model_clinear : public Model_Base {
    public :
        Model_clinear(const arma::Mat<double>& data);
        double compute_segment_cost(arma::uword start, arma::uword end);

    private :
        // member variables to compute cost in compute_segment_cost()
        // by being member variables, we avoid re-declaring them at each
        // compute_segment_cost call
        arma::Mat<double> y;

        int segment_length;
        arma::Col<double> segment_times;
        arma::Mat<double> segment_prediction;
        arma::Mat<double> residuals;
        double segment_cost;
    };
#endif // MODEL_CLINEAR_H