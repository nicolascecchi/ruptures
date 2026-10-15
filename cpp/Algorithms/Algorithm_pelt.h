#ifndef ALGORITHM_PELT_H
#define ALGORITHM_PELT_H

#include "Algorithm_Base.h"
#include "../Models/Model_Base.h"

#include <armadillo>

class Algorithm_pelt : public Algorithm_Base {
    public:
        Algorithm_pelt(Model_Base& model_);

        // Precomputes in the Model class any necessary structures for fast cost computation
        //  TODO ADD OVERLOADS FOR THE FUNCTION FIT AS NEEDED FOR
        //  DIFFERENT CASES OF THE CONSTRUCTION OF THE ALGORITHM_CLASS
        //  WITH / WITHOUT DATA AND MODELS ASSIGNED

        // Predicts change-points with given penalty
        void predict(double beta) override;
        void predict(arma::uword K) override;
        void backtrack() override;

    private:        
        void prune(arma::uword t);
        //double get_cost_at_t(int t) const { return dp_cost[t]; }
        //arma::uvec get_changepoints() const { return changepoints; }

};

#endif // ALGORITHM_PELT_H
