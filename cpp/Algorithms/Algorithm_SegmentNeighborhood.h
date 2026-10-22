#ifndef ALGORITHM_SEGMENTNEIGHBORHOOD_H
#define ALGORITHM_SEGMENTNEIGHBORHOOD_H

#include "Algorithm_Base.h"
#include "../Models/Model_Base.h"

#include <armadillo>

/*
Adapted from an upstream C++ codebase, here extended with the
min_size and jump segmentation parameters, following the same grid semantics
as ruptures' pure Python Dynp: candidate change points are the multiples of
`jump` and every segment is at least `min_size` points long. The defaults
(1, 1) keep the original single-argument predict(uword) behaviour.

NOTE: the upstream class and file are Algorithm_SegmentNeighbourhood (British
spelling), renamed here to the American spelling. This file is an adaptation,
not a verbatim copy: re-syncing it from upstream would drop the min_size/jump
extension, which upstream's own copy does not have.
*/

class Algorithm_SegmentNeighborhood : public Algorithm_Base {
    public:
        Algorithm_SegmentNeighborhood(Model_Base& model_, arma::uword min_size = 1, arma::uword jump = 1);

        // Precomputes in the Model class any necessary structures for fast cost computation
        //  TODO ADD OVERLOADS FOR THE FUNCTION FIT AS NEEDED FOR
        //  DIFFERENT CASES OF THE CONSTRUCTION OF THE ALGORITHM_CLASS
        //  WITH / WITHOUT DATA AND MODELS ASSIGNED

        // Predicts change-points with a fixed number of changes K
        // (predict(double) is not a valid call: SegmentNeighborhood has no penalty).
        void predict(double beta) override;
        void predict(arma::uword K) override;
        void backtrack() override;

    private:
        // Fills dp_cost(t, k) with the optimal cost of segmenting {y_1,...,y_t}
        // with exactly k change points, and dp_prev_time(t, k) with the last
        // change point of that optimal segmentation (see scan()).
        void scan(arma::uword t, arma::uword k);
        arma::uword min_size; /**< smallest admissible segment length */
        arma::uword jump;     /**< candidate change points are multiples of jump */
        //double get_cost_at_t(int t) const { return dp_cost[t]; }
        //arma::uvec get_changepoints() const { return changepoints; }

};

#endif // ALGORITHM_SEGMENTNEIGHBORHOOD_H