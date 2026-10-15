/*
Model_linear is an implementation of the Model_Base that computes the L2 cost when prediction changes linearly over a segment. 
Derived from Model_Base class, it will have an compute_segment_cost() member function to compute the cost over a segment.
*/

#ifndef MODEL_MEANVAR_H
#define MODEL_MEANVAR_H

#include "Model_Base.h"

#include <armadillo>

/** 
* @class Model_meanvar
* @brief Model_meanvar models signals that are piecewise constant with Gaussian noise and identical variance.
*
*/
class Model_meanvar : public Model_Base {
    public :
    /** @brief Constructor for the Model_meanvar class.
     *  @param data The input data matrix.
     * 
     * 
     */
        Model_meanvar(const arma::Mat<double>& data); 
        //Model_constant_rank(arma::Mat<double> data_) : Model_Base(std::move(data_)) {};
        double compute_segment_cost(arma::uword start, arma::uword end);
        //void fit(arma::Mat<double>& data);

    private :
        // member variables to compute cost in compute_segment_cost() 
        // by being member variables, we avoid re-declaring them at each 
        // compute_segment_cost call
        arma::Col<double> y;
        
        int segment_length;
        double segment_cost;
        double eps;
        double val;
        arma::Col<double> sub_y;
        arma::Mat<double> cov;
    };
#endif // MODEL_MEANVAR_H
