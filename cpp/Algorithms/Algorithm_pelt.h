#ifndef ALGORITHM_PELT_H
#define ALGORITHM_PELT_H

#include "Algorithm_Base.h"
#include "../Models/Model_Base.h"

#include <armadillo>

/*
Extended with the min_size and jump segmentation parameters, following the
same grid semantics as ruptures' pure Python Pelt: candidate change points are the multiples of
`jump` that are at least `min_size`, plus the last observation. The defaults
(1, 1) keep the original single-argument predict(double) behaviour.
*/

class Algorithm_pelt : public Algorithm_Base {
    public:
        Algorithm_pelt(Model_Base& model_, arma::uword min_size = 1, arma::uword jump = 1);

        // Precomputes in the Model class any necessary structures for fast cost computation
        //  TODO ADD OVERLOADS FOR THE FUNCTION FIT AS NEEDED FOR
        //  DIFFERENT CASES OF THE CONSTRUCTION OF THE ALGORITHM_CLASS
        //  WITH / WITHOUT DATA AND MODELS ASSIGNED

        // Predicts change-points with given penalty
        void predict(double beta) override;
        void predict(arma::uword K) override;
        void backtrack() override;

    private:
        // Prunes the admissible starts at t using the candidate costs predict()
        // has just computed for those same starts, in admissible_points order.
        void prune(arma::uword t, const arma::dvec& candidate_costs);
        arma::uword min_size; /**< smallest admissible segment length */
        arma::uword jump;     /**< candidate change points are multiples of jump */
        //double get_cost_at_t(int t) const { return dp_cost[t]; }
        //arma::uvec get_changepoints() const { return changepoints; }

};

#endif // ALGORITHM_PELT_H
