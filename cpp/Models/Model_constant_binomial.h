/*
Model_constant_binomial is an implementation of the Model_Base that computes the L2 cost when prediction changes in mean over a segment. 
Derived from Model_Base class, it will have an compute_segment_cost() member function to compute the cost over a segment.
*/

#ifndef MODEL_CONSTANT_BINOMIAL_H
#define MODEL_CONSTANT_BINOMIAL_H

#include "Model_Base.h"

#include <armadillo>

/* @brief Model_constant_binomial models signals that are piecewise constant in mean with binomial noise and identical variance.
*
* The model maximizes the likelihood of the data, which is the same as minimizing 
* the sum of squared errors (L2 cost) between the data and the piecewise constant mean.
* 
*   
*
*/
class Model_constant_binomial : public Model_Base {
    public :
        Model_constant_binomial(const arma::Mat<double>& data
                                , int m=1); 
        //Model_constant_binomial(arma::Mat<double> data_) : Model_Base(std::move(data_)) {};
        double compute_segment_cost(arma::uword start, arma::uword end);
        //void fit(arma::Mat<double>& data);
        arma::Mat<double> get_cumsum_y() const;
    
    private :
        // member variables to compute cost in compute_segment_cost() 
        // by being member variables, we avoid re-declaring them at each 
        // compute_segment_cost call
 
        arma::Mat<double> cumsum_y;    
        arma::Row<double> dim_cost;
        double segment_cost;
        double segment_phat;
        int segment_length;
        int m; // number of trials in the binomial distribution
};
#endif // MODEL_CONSTANT_BINOMIAL_H
