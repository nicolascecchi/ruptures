/*
Model_constant_normal is an implementation of the Model_Base that computes de L2 cost when prediction changes in mean over a segment. 
Derived from Model_Base class, it will have an compute_segement_cost() member function to compute the cost over a segment.
*/

#ifndef MODEL_CONSTANT_NORMAL_H
#define MODEL_CONSTANT_NORMAL_H

#include "Model_Base.h"

#include <armadillo>

/* @brief Model_constant_normal models signals that are piecewise constant in mean with Gaussian noise and identical variance.
*
* The model maximizes the likelihood of the data, which is the same as minimizing 
* the sum of squared errors (L2 cost) between the data and the piecewise constant mean.
* 
*   
*
*/
class Model_constant_normal : public Model_Base {
    public :
        Model_constant_normal(const arma::Mat<double>& data); 
        //Model_constant_normal(arma::Mat<double> data_) : Model_Base(std::move(data_)) {};
        double compute_segment_cost(arma::uword start, arma::uword end);
        //void fit(arma::Mat<double>& data);
        arma::Mat<double> get_cumsum_y() const;
        arma::Mat<double> get_cumsum_squared_y() const;
    
    private :
        arma::Mat<double> cumsum_y;
        arma::Mat<double> cumsum_squared_y;
        
        // member variables to compute cost in compute_segment_cost() 
        // by being member variables, we avoid re-declaring them at each 
        // compute_segment_cost call
        int segment_length;
        arma::Row<double> dim_cost;
        double segment_cost;
};
#endif // MODEL_CONSTANT_NORMAL_H
