#include "Algorithm_Base.h"
#include "Algorithm_pelt.h"

#include "../Models/Model_Base.h"

#include <armadillo>
#include <limits>
#include <iostream>
#include <cmath>

constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double EPSILON = std::numeric_limits<double>::epsilon();

/**
 * @brief Constructs a PELT algorithm bound to a model for changepoints detection.
 *
 * The provided @p model_ supplies segment cost evaluations that the algorithm
 * reuses throughout predict() without additional copies.
 *
 * @param[in] model_ Reference to the underlying model that computes segment costs.
 */
Algorithm_pelt::Algorithm_pelt(Model_Base& model_)
	: Algorithm_Base{model_}
    {
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
 *       - Updates @c admissible_size_at_t(t+1) with the new count of admissible points
 *       - Adds the current time point @c t as a candidate for future pruning (admissible_points(new_len) = t)
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
    // add the current point as a candidate for future pruning
    // and make other necessary updates.
    admissible_points(new_len) = t;     
    new_len += 1;
    admissible_size_at_t(t + 1) = new_len;
};

/**
 * @brief Runs the PELT segmentation to compute optimal changepoints with penalty @p beta.
 *
 * This method implements the main loop of the PELT algorithm to find optimal changepoints in the data.
 * Loops over all the observations and, at each time point, computes the optimal cost of segmentation up to that point by considering all admissible previous changepoints.
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
    admissible_size_at_t(1) = 1; // only point 0 is admissible

    // t counts the number of points observed
    // it goes from 1 point to nobs (inclusive)
    for (arma::uword t=1; t<model.nobs+1; t++){
        // s is the index of the last (1-indexed) point of the previous segment
        // s = 0 means that the last segment starts at the beginning of the data, aka no changes,
        // otherwise we have at least one changepoint.
        // candidate = dp_cost[s] + segment_cost(s,t) + beta means
        // candidate = cost for {y_1,...,y_s} + cost for {y_{s+1},...,y_t} + penalty for the change at s
        CURR_MIN = INF;
        for (arma::uword i = 0; i < admissible_size_at_t(t); i++){
            s = admissible_points(i);
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
