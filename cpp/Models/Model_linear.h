/*
Model_linear is an implementation of the Model_Base that computes the L2 cost when prediction changes linearly over a segment. 
Derived from Model_Base class, it will have an compute_segment_cost() member function to compute the cost over a segment.
*/

#ifndef MODEL_LINEAR_H
#define MODEL_LINEAR_H

#include "Model_Base.h"

#include <armadillo>

/** 
* @class Model_linear 
* @brief Model_linear models signals that are piecewise linear with Gaussian noise and identical variance.
*
* The model maximizes the likelihood of the data, which is the same as minimizing 
* the sum of squared errors (L2 cost) between the data and the piecewise linear.
* 
*/
class Model_linear : public Model_Base {
    public :
        Model_linear(const arma::Mat<double>& data); 
        //Model_constant_rank(arma::Mat<double> data_) : Model_Base(std::move(data_)) {};
        double compute_segment_cost(arma::uword start, arma::uword end);
        //void fit(arma::Mat<double>& data);

    private :
        // member variables to compute cost in compute_segment_cost() 
        // by being member variables, we avoid re-declaring them at each 
        // compute_segment_cost call
        arma::Col<double> y;
        arma::Mat<double> covariates;
        
        
        int segment_length;
        arma::Col<double> sub_y;
        arma::Mat<double> sub_covariates;
        arma::Col<double> residuals;
        double segment_cost;
        arma::Col<double> segment_solution;
    };
#endif // MODEL_LINEAR_H
