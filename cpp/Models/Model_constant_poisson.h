/*
Model_constant_normal is an implementation of the Model_Base that computes de L2 cost when prediction changes in mean over a segment. 
Derived from Model_Base class, it will have an compute_segement_cost() member function to compute the cost over a segment.
*/

#ifndef MODEL_CONSTANT_POISSON_H
#define MODEL_CONSTANT_POISSON_H

#include "Model_Base.h"

#include <armadillo>

/** 
* @class Model_constant_poisson 
* @brief Model_constant_poisson models signals that are piecewise constant in mean with Poisson noise and identical variance.
*
* The model computes the negative log-likelihood of the data under a Poisson model. 
*/
class Model_constant_poisson : public Model_Base {
    public :
        Model_constant_poisson(const arma::Mat<double>& data); 

        /** @brief Computes the cost of a segment.
         *  @param start The starting index of the segment.
         *  @param end The ending index of the segment.
         *  @return The computed cost.
         */
        double compute_segment_cost(arma::uword start, arma::uword end);
        arma::Mat<double> get_cumsum_y() const {return cumsum_y;};
    
    private :
        arma::Mat<double> cumsum_y ;
        
        // member variables to compute cost in compute_segment_cost() 
        // by being member variables, we avoid re-declaring them at each 
        // compute_segment_cost call
        int segment_length;
        arma::Row<double> dim_cost;
        double segment_cost;
};

#endif // MODEL_CONSTANT_POISSON_H
