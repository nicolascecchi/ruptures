/*
Model_linear is an implementation of the Model_Base that computes the L2 cost when prediction changes linearly over a segment. 
Derived from Model_Base class, it will have an compute_segment_cost() member function to compute the cost over a segment.
*/

#ifndef MODEL_LINEAR_CONTINUOUS_H
#define MODEL_LINEAR_CONTINUOUS_H

#include "Model_Base.h"

#include <armadillo>

/** 
* @class Model_linear_continuous
* @brief Model_linear_continuous models signals that are piecewise linear with Gaussian noise and identical variance.
*
* The model maximizes the likelihood of the data, which is the same as minimizing 
* the sum of squared errors (L2 cost) between the data and the piecewise linear.
* 
*/
class Model_linear_continuous : public Model_Base {
    public :
    /** @brief Constructor for the Model_linear_continuous class.
     *  @param data The input data matrix.
     * 
     * 
     */
        Model_linear_continuous(const arma::Mat<double>& data); 
        //Model_constant_rank(arma::Mat<double> data_) : Model_Base(std::move(data_)) {};
        double compute_segment_cost(arma::uword start, arma::uword end);
        //void fit(arma::Mat<double>& data);

    private :
        // member variables to compute cost in compute_segment_cost() 
        // by being member variables, we avoid re-declaring them at each 
        // compute_segment_cost call
        arma::Col<double> y;
        
        
        int segment_length;
        double slope;
        double intercept;
        double segment_cost;
        arma::Col<double> sub_y;
        arma::Col<double> residuals;
        arma::Col<double> segment_prediction;
    };
#endif // MODEL_LINEAR_CONTINUOUS_H
