#include "Algorithm_Base.h"
#include "Algorithm_SegmentNeighborhood.h"

#include "../Models/Model_Base.h"

#include <armadillo>
#include <limits>
#include <stdexcept>

constexpr double INF = std::numeric_limits<double>::infinity();

/**
 * @brief Constructs a segment neighborhood algorithm bound to a model for changepoints detection.
 *
 * The provided @p model_ supplies segment cost evaluations reused across predict() to segment the
 * signal. @p min_size is the smallest admissible segment length and @p jump the step between
 * candidate change points, following ruptures' Python Dynp semantics.
 *
 * @param[in] model_ Reference to the underlying model that computes segment costs.
 * @param[in] min_size_ Minimum segment length (default 1: no constraint).
 * @param[in] jump_ Step between candidate change points (default 1: every index).
 */
Algorithm_SegmentNeighborhood::Algorithm_SegmentNeighborhood(Model_Base& model_, arma::uword min_size_, arma::uword jump_)
	: Algorithm_Base{model_}
    , min_size{min_size_}
    , jump{jump_}
    {
    if (min_size < 1 || jump < 1){
        throw std::invalid_argument("SegmentNeighborhood expects min_size >= 1 and jump >= 1");
    }
};


/**
 * @brief Runtime error. The segment neighborhood algorithm takes the integer
 * number of changepoints K, not a float penalty.
 *
 * @param beta Unused penalty argument.
 */
void Algorithm_SegmentNeighborhood::predict(double beta){
    throw std::invalid_argument("Segment neighborhood algorithm requires the integer number of changepoints K (predict(K)), not a float penalty beta.");
};


/**
 * @brief Scans the candidate last change points and fills @c dp_cost(t, k) and @c dp_prev_time(t, k).
 *
 * @c dp_cost(t, k) is the optimal cost of segmenting {y_1,...,y_t} with exactly k change points
 * and @c dp_prev_time(t, k) the last change point of that optimal segmentation. The candidates s
 * are the multiples of jump strictly below t (Python's Dynp.seg scans k % jump == 0 in
 * range(start, end)). A candidate is admissible when the left sub-problem [0, s) can host
 * k - 1 change points — mirroring ruptures' sanity_check — and the right segment [s, t) holds
 * at least min_size points. The first minimizing candidate wins (strict <, ascending s),
 * mirroring Python min()'s first-wins tie-breaking.
 *
 * @param[in] t End of the region being segmented.
 * @param[in] k Number of change points required in [0, t).
 */
void Algorithm_SegmentNeighborhood::scan(arma::uword t, arma::uword k){
    double CURR_MIN{INF};
    double segment_cost{0};
    double candidate_cost{0};
    arma::uword best_s{0};
    arma::uword s{0};
    for (s = 0; s < t; s += jump){
        // Feasibility of the left sub-problem [0, s) with k - 1 change points,
        // mirroring Python's sanity_check(n_samples=s, n_bkps=k-1, jump, min_size)
        // (src/ruptures/utils/utils.py): enough grid points for k - 1 change
        // points, and enough room for k - 1 minimum-length grid-quantized
        // segments plus the leftmost one.
        if (k - 1 > s / jump){
            continue;
        }
        if ((k - 1) * ((min_size + jump - 1) / jump) * jump + min_size > s){
            continue;
        }
        // The right segment [s, t) must hold at least min_size points
        // (Python: end - bkp >= min_size).
        if (t - s < min_size){
            continue;
        }
        // An infeasible left sub-problem cannot start a feasible segmentation:
        // skip it without evaluating the segment cost (Python only ever visits
        // feasible splits, since its seg() recursion succeeds for every
        // candidate passing the checks above).
        if (dp_cost(s, k - 1) == INF){
            continue;
        }
        segment_cost = model.compute_segment_cost(s, t);
        candidate_cost = dp_cost(s, k - 1) + segment_cost;
        if (candidate_cost < CURR_MIN){
            CURR_MIN = candidate_cost;
            best_s = s;
        }
    }
    dp_cost(t, k) = CURR_MIN;
    dp_prev_time(t, k) = best_s;
};


/**
 * @brief Runs the segment neighborhood recurrence to compute optimal changepoints with number of changes @p K.
 *
 * Iterates over the number of change points (outer) and the candidate last segment ends (the
 * multiples of @p jump, plus the last observation — as in ruptures' Python Dynp), and, at each
 * (k, t), finds the minimal cost segmentation ending at @p t with exactly k changes by scanning
 * the admissible previous change points (see scan()).
 *
 * @param[in] K Number of changepoints to detect.
 *
 * @note The matrices storing partial results are reshaped here, since their column count
 *       depends on @p K.
 * @note Updates costs and best predecessors in place; the admissible-set buffers inherited
 *       from Algorithm_Base are left untouched (the segment neighborhood never prunes).
 * @note Reinitializes internal buffers unconditionally at each call.
 * @see scan()
 * @see backtrack()
 */
void Algorithm_SegmentNeighborhood::predict(arma::uword K){
    if (K < 1) {
        throw std::invalid_argument("Number of changepoints K must be a strictly positive integer (K>=1).");
    }
    if (K >= model.nobs) {
        throw std::invalid_argument("Number of changepoints K must satisfy K < nobs.");
    }
    if (min_size > model.nobs){
        // Unreachable through Dynp (sanity_check pre-validates); the guard
        // keeps direct ruptures._core users from underflowing the grid.
        throw std::invalid_argument("min_size cannot exceed the number of observations");
    }
    // Column k stores the optimal cost with exactly k changepoints, for k in {0, ..., K}.
    // The buffers are reinitialized unconditionally: their column count depends on K, so
    // Algorithm_pelt's "already populated" check does not apply.
    const arma::uword n_cols = K + 1;
    dp_cost.reshape(model.nobs + 1, n_cols);
    dp_prev_time.reshape(model.nobs + 1, n_cols);
    dp_cost.fill(INF);
    dp_prev_time.zeros();
    changepoints = arma::ucolvec{};

    // Base case: zero changepoints. Like Python's seg(start, end, 0), the single
    // segment carries no min_size constraint.
    dp_cost(0, 0) = 0.0;
    for (arma::uword t = 1; t < model.nobs + 1; ++t) {
        dp_cost(t, 0) = model.compute_segment_cost(0, t);
        dp_prev_time(t, 0) = 0;
    }

    // DP recurrence for k >= 1 changepoints. t ranges over the candidate last
    // segment ends, as in ruptures' Python implementation (Dynp.seg): the
    // multiples of jump, plus the last observation, which is always evaluated
    // even when it is off grid (not a multiple of jump).
    for (arma::uword k = 1; k < n_cols; ++k) {
        for (arma::uword t = jump; t < model.nobs; t += jump) {
            scan(t, k);
        }
        scan(model.nobs, k);
    }

    if (dp_cost(model.nobs, K) == INF){
        // Unreachable through Dynp (sanity_check pre-validates): Python's seg()
        // would die on its "no admissible last breakpoints" assert instead.
        throw std::invalid_argument("No feasible segmentation exists for the given (K, min_size, jump) parameters.");
    }

    backtrack(); // update changepoints vector inplace
};


/**
 * @brief Reconstructs changepoints by backtracking optimal predecessors.
 *
 * Starts from the last observation and follows @c dp_prev_time to prepend each changepoint into
 * a temporary vector before storing it in @c changepoints. The resulting vector is the ascending
 * list of the K interior change points followed by the last observation (index 0 excluded) —
 * the same convention as Algorithm_pelt::backtrack, whose output the Python layer relies on.
 *
 * @see predict()
 */
void Algorithm_SegmentNeighborhood::backtrack(){
    arma::uword t = model.nobs;
    arma::uvec changepoints_temp;

    // Include the endpoint of the last segment.
    changepoints_temp.insert_rows(0, arma::uvec{t});

    // Last valid column index is K (n_cols - 1).
    arma::uword k = dp_cost.n_cols - 1;
    while (k > 0){
        t = dp_prev_time(t, k);
        changepoints_temp.insert_rows(0, arma::uvec{t});
        k -= 1; // move to the previous column corresponding to one less segment
    }
    changepoints = changepoints_temp;
};