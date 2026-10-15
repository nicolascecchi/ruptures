/*
Algorithm_Base is the base class for algorithms. 
Different algorithms (e.g., Optimal Partitioning, PELT, etc.) will be derived from this base class.
Each algorithm will use a Cost_Base derived class to compute costs over segments and find the optimal
partition associated to the underlying model of the data. 

Naming convention: Algorithm_[AlgorithmName] 
eg. Algorithm_VanillaOP, Algorithm_PELT, etc.
*/
#ifndef ALGORITHM_BASE_H
#define ALGORITHM_BASE_H

#include "../Models/Model_Base.h"

#include <memory>
#include <armadillo>

class Algorithm_Base {
    public:
        Algorithm_Base(Model_Base& model_): 
            model(model_)
            ,dp_cost{arma::dmat(model.nobs + 1, 1, arma::fill::zeros)}
            ,dp_prev_time{arma::umat(model.nobs + 1, 1, arma::fill::zeros)}
            ,changepoints{}
            ,admissible_points{arma::umat(model.nobs + 1, 1, arma::fill::zeros)}
            ,admissible_size_at_t{arma::umat(model.nobs + 2, 1, arma::fill::zeros)}{
        };
        // Since we own a unique_ptr, we need to delete copy constructor and copy assignment operator
        Algorithm_Base(const Algorithm_Base&) = delete;
        Algorithm_Base& operator=(const Algorithm_Base&) = delete;
        // Default destructor, needed for correct cleaning of derived classes
        virtual ~Algorithm_Base() = default;
        virtual void predict(double beta) = 0;
        virtual void predict(arma::uword K) = 0;
        virtual void backtrack() = 0;

        arma::dmat get_cost_history() const {return dp_cost;} /**< Returns the cost history matrix */
        arma::umat get_best_prev_idx() const {return dp_prev_time;} /**< Returns the matrix of best previous indices for backtracking */
        arma::ucolvec get_changepoints() const {return changepoints;} /**< Returns the vector of detected change-points */
        arma::umat get_admissible_points() const {
            const arma::uword n_admissible = admissible_size_at_t(model.nobs);
            if (n_admissible == 0) {
                return arma::umat{};
            }
            return admissible_points.rows(0, n_admissible - 1);
        } /**< Returns the matrix of admissible points after pruning, up to the size at time nobs */
        arma::uword get_admissible_size_at_t(arma::uword t) const {return admissible_size_at_t(t);} /**< Returns the size of admissible points at time t */
        arma::umat get_admissible_size_history() const {return admissible_size_at_t;} /**< Returns the history of admissible sizes at each time t */
        
        // Accessors
        //double get_cost(arma::uword t) const { return dp_cost(t); }
    protected:        
        Model_Base& model; /**< Reference to the model used by this algorithm */
        arma::dmat dp_cost{}; /**< Matrix to store the optimal cost up to each time t */
        arma::umat dp_prev_time{}; /**< Matrix to store the index of the last change */
        arma::ucolvec changepoints{}; /**< Vector to store the detected change-points */
        arma::umat admissible_points{}; /**< Matrix to store admissible changepoints after pruning */
        arma::umat admissible_size_at_t{}; /**< Size of admissible_points at time t */
};    

using AlgPtr = Algorithm_Base*; //std::unique_ptr<Algorithm_Base>;

#endif // ALGORITHM_BASE_H
