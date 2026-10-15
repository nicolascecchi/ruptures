/*
Model_constant_rank is an implementation of the Model_Base that computes de L2 cost when prediction changes in rank over a segment. 
Derived from Model_Base class, it will have an compute_segement_cost() member function to compute the cost over a segment.
*/

#ifndef MODEL_CONSTANT_RANK_H
#define MODEL_CONSTANT_RANK_H

#include "Model_Base.h"

#include <armadillo>

/** 
* @class Model_constant_rank
* @brief Model_constant_rank models signals that are piecewise constant in rank with Gaussian noise and identical variance.
*
* The model maximizes the likelihood of the data, which is the same as minimizing 
* the sum of squared errors (L2 cost) between the data and the piecewise constant rank.
* 
*/
class Model_constant_rank : public Model_Base {
    public :
        Model_constant_rank(const arma::Mat<double>& data); 
        //Model_constant_rank(arma::Mat<double> data_) : Model_Base(std::move(data_)) {};
        double compute_segment_cost(arma::uword start, arma::uword end);
        //void fit(arma::Mat<double>& data);

        // GETERS
        arma::Mat<double> get_centered_ranks() const { return centered_ranks; }
        arma::Mat<double> get_cusum_ranks() const { return cusum_ranks; }
        arma::Mat<double> get_inv_cov() const { return inv_cov; }
    private :
        const arma::Mat<double> centered_ranks;
        const arma::Mat<double> cusum_ranks;
        const arma::Mat<double> inv_cov;
        
        // member variables to compute cost in compute_segment_cost() 
        // by being member variables, we avoid re-declaring them at each 
        // compute_segment_cost call
        int segment_length;
        arma::Row<double> dim_cost;
        double segment_cost;
        arma::vec segment_dim_mean;

        const arma::Mat<double> compute_centered_ranks(const arma::Mat<double>& data);
        const arma::Mat<double> compute_inv_cov(const arma::Mat<double>& centered_ranks);
        const arma::Mat<double> compute_cusum_ranks(const arma::Mat<double>& centered_ranks);
    };
#endif // MODEL_CONSTANT_RANK_H
