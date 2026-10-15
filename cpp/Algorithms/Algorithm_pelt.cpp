#include "Algorithm_Base.h"
#include "Algorithm_pelt.h"

#include "../Models/Model_Base.h"

#include <armadillo>
#include <limits>
#include <iostream>
#include <cmath>
#include <stdexcept>

constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double EPSILON = std::numeric_limits<double>::epsilon();

/**
 * @brief Constructs a PELT algorithm bound to a model for changepoints detection.
 *
 * The provided @p model_ supplies segment cost evaluations that the algorithm
 * reuses throughout predict() without additional copies. @p min_size is the
 * smallest admissible segment length and @p jump the step between candidate
 * change points, following ruptures' Python Pelt semantics.
 *
 * @param[in] model_ Reference to the underlying model that computes segment costs.
 * @param[in] min_size_ Minimum segment length (default 1: no constraint).
 * @param[in] jump_ Step between candidate change points (default 1: every index).
 */
Algorithm_pelt::Algorithm_pelt(Model_Base& model_, arma::uword min_size_, arma::uword jump_)
	: Algorithm_Base{model_}
    , min_size{min_size_}
    , jump{jump_}
    {
    if (min_size < 1 || jump < 1){
        throw std::invalid_argument("PELT expects min_size >= 1 and jump >= 1");
    }
};


/**
 * @brief Prunes the list of admissible change-point candidates using the PELT algorithm.
 * 
 * This method implements the pruning step of the PELT (Pruned Exact Linear Time) algorithm.
 * It removes suboptimal change-points from the admissible set based on an inequality-based pruning rule. 
 * Only change-points that could potentially lead to an optimal segmentation are retained.
 * 
 * The pruning criterion compares the cost of a candidate change-point plus the segment cost
 * against the best solution found for current time. Change-points that cannot improve future
 * segmentations are discarded.
 * 
 * @param[in] t The current time index at which pruning is performed. Indicates how many points have been observed so far.
 * 
 * @note This method modifies the internal state:
 *       - Updates @c admissible_points so that the first @c new_len entries contain the remaining admissible change-points after pruning
 *       - Stores the new count of admissible points in @c admissible_size_at_t at the next change-point to scan (t + jump, the last observation when t + jump overshoots it)
 * 
 * @warning The method assumes that @c admissible_size_at_t(t) correctly reflects
 *          the number of valid candidates after seeing @c t points.
 * 
 * @see dp_cost
 * @see admissible_points
 * @see admissible_size_at_t
 * @see model.compute_segment_cost()
 */
void Algorithm_pelt::prune(arma::uword t){
    arma::uword new_len {0}; // remove declaration from the function.
    arma::uword idx;

    // We explicitly don't want to go over all the admissible_points vector
    // but only from 0 up to the number of admissibles at time t, which is stored in admissible_size_at_t(t)
    // Going all over the vector would be making nonsense computations.
    for (arma::uword i = 0; i < admissible_size_at_t(t); i++){
        idx = admissible_points(i);
        auto cas = dp_cost(idx);
        auto seg_cost = model.compute_segment_cost(idx, t);
        auto cat = dp_cost(t);
        if (cas + seg_cost <= cat + EPSILON){
            admissible_points(new_len) = idx;
            new_len += 1;
        }
    }
    // the filtered starts are the initial admissible set of the next change
    // point to scan (the last observation, when t + jump overshoots it).
    // New starts are appended before each scan, in predict().
    arma::uword next = t + jump;
    if (next > model.nobs){
        next = model.nobs;
    }
    admissible_size_at_t(next) = new_len;
};

/**
 * @brief Runs the PELT segmentation to compute optimal changepoints with penalty @p beta.
 *
 * This method implements the main loop of the PELT algorithm to find optimal changepoints in the data.
 * Loops over the candidate change points (the multiples of jump that are at least min_size, plus the last observation) and, at each candidate, computes the optimal cost of segmentation up to that point by considering all admissible previous change-points.
 *
 * @param[in] beta Penalty added per changepoint to control segmentation complexity.
 *
 * @note The method updates the internal state of the algorithm:
 *       - @c dp_cost is updated with the optimal cost of segmentation up to t observed data points {y_1,...,y_t}.
 *       - @c dp_prev_time is updated to store the 0-based index of the best last change, having seen t points.
 *       - @c admissible_points and @c admissible_size_at_t are updated in place by the prune() method to maintain the set of admissible changepoints for future iterations.
 *       - @c changepoints vector is populated by calling backtrack() method.
 * @note Resets internal buffers when a prior fit already populated them.
 *
 * @see prune()
 * @see backtrack()
 */
void Algorithm_pelt::predict(double beta){
    if (min_size > model.nobs){
        // Unreachable through Pelt (sanity_check pre-validates); the guard
        // keeps direct ruptures._core users from underflowing the grid.
        throw std::invalid_argument("min_size cannot exceed the number of observations");
    }
    // the algorithm was already fit before
    // reinitialize all the vectors to be sure to start from a clean state
    if (dp_cost(0)!=0){ 
        dp_cost = arma::dmat(model.nobs + 1, 1, arma::fill::zeros);
        dp_prev_time = arma::umat(model.nobs + 1, 1, arma::fill::zeros);
        changepoints = arma::ucolvec({});
        admissible_points = arma::umat(model.nobs + 1, 1, arma::fill::zeros);
        admissible_size_at_t = arma::umat(model.nobs + 2, 1, arma::fill::zeros);
    }
    // Set variables to controlled initial states.
    double CURR_MIN{INF};
    double segment_cost{0}; 
    double candidate_cost{0};
    dp_cost(0) = -beta;
    arma::uword best_s{0};
    arma::uword s{0};
    // t ranges over the candidate change points, as in ruptures' Python
    // implementation (Pelt._seg): the multiples of jump that are at least
    // min_size, plus the last observation, which is always evaluated even
    // when it is off the grid (not a multiple of jump).
    arma::uword t = jump * ((min_size + jump - 1) / jump);
    while (true){
        if (t > model.nobs){
            t = model.nobs; // off-grid last observation
        }
        // Newest admissible start, Python's new_adm_pt: the largest grid
        // point (multiple of jump) not exceeding t - min_size.
        admissible_size_at_t(t) += 1;
        admissible_points(admissible_size_at_t(t) - 1) = jump * ((t - min_size) / jump);
        // s is the index of the last (1-indexed) point of the previous segment
        // s = 0 means that the last segment starts at the beginning of the data, aka no changes,
        // otherwise we have at least one changepoint.
        // candidate = dp_cost[s] + segment_cost(s,t) + beta means
        // candidate = cost for {y_1,...,y_s} + cost for {y_{s+1},...,y_t} + penalty for the change at s
        // Starts s with 0 < s < min_size define no valid previous segmentation,
        // so they are skipped (Python: KeyError on partitions[s]).
        CURR_MIN = INF;
        for (arma::uword i = 0; i < admissible_size_at_t(t); i++){
            s = admissible_points(i);
            if (s > 0 && s < min_size){
                continue;
            }
            segment_cost = model.compute_segment_cost(s, t);
            candidate_cost = (dp_cost(s) + segment_cost + beta);
            if (candidate_cost < CURR_MIN){
                CURR_MIN = candidate_cost;
                best_s = s;
            }
        }
        // Use segment_cost and beta to update optimal partitioning out of previous time loops
        dp_cost(t) = CURR_MIN;
        dp_prev_time(t) = best_s;
        prune(t); // updates admissible_points vector inplace
        if (t == model.nobs){
            break;
        }
        t += jump;
    }
    backtrack(); // updates changepoints vector inplace
};
/**
 * @brief Runs the PELT segmentation to compute optimal changepoints with penalty @p K. 
 * 
 * This method is an overload of predict() that takes an integer but treats it as a float for the penalty.
 * 
 * @param[in] K  Penalty added per changepoint to control segmentation complexity.
 * 
 * @note The method updates the internal state of the algorithm:
 *       - @c dp_cost is updated with the optimal cost of segmentation up to t observed data points {y_1,...,y_t}.
 *       - @c dp_prev_time is updated to store the 0-based index of the best last change, having seen t points.
 *       - @c admissible_points and @c admissible_size_at_t are updated in place by the prune() method to maintain the set of admissible changepoints for future iterations.
 *       - @c changepoints vector is populated by calling backtrack() method.
 * @note Resets internal buffers when a prior fit already populated them.
 *
 * @see prune()
 * @see backtrack()
 * 
 */
void Algorithm_pelt::predict(arma::uword K){
    double beta = static_cast<double>(K);
    predict(beta);
};

/**
 * @brief Reconstructs changepoints by backtracking optimal predecessors.
 *
 * Starts from the last observation and follows @c dp_prev_time to prepend each
 * changepoint into a temporary vector, then stores it in @c changepoints.
 *pred
 * @note Assumes @c dp_prev_time and @c model.nobs were populated by predict().
 *
 * @see predict()
 */
void Algorithm_pelt::backtrack(){
    arma::uword t = model.nobs;
    arma::ucolvec changepoints_temp;
    while (t > 0){
        changepoints_temp.insert_rows(0, arma::uvec{t});
        t = dp_prev_time(t);
    }
    changepoints = changepoints_temp;   
};
