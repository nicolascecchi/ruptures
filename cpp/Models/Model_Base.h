/*
Base class for model classes. 
Multiple Model classes will be derived from this Model_Base class. 
The Model_Classes will be passed to a Model class that will use this Model_Class to compute
the costs of segments and optimize a cost function. 

Naming convention for derived classes will be: Model_[Model]_[Loss].
Example: 
Model_constant_normal is a derived class that computes the L2 cost when the prediction changes in mean over a segment.
*/
#ifndef Model_Base_H
#define Model_Base_H

#include <armadillo>
#include <memory> // For CostPtr

class Model_Base{
    public:
        ////////////////////////////////////////////////////////////////
        // Constructors, Destructors, etc
        ////////////////////////////////////////////////////////////////
        Model_Base()=delete;
        Model_Base(arma::uword nobs_, arma::uword ndims_):
            nobs{nobs_}
            ,ndims{ndims_}
        {
            if (nobs < ndims) {
                std::cerr << "Error: nobs (" << nobs << ") is smaller than ndims (" << ndims << "), likely a mistake." << std::endl;
            }
        };
        // Virtual destructor to ensure that derived classes are properly cleaned up.
        virtual ~Model_Base() = default; 

        ////////////////////////////////////////////////////////////////
        // member functions 
        ////////////////////////////////////////////////////////////////

        // Pure virtual method to compute the cost of a segment 
        // from start (included) to end (excluded).
        // Pure virtual means that each derived class must implement this method.
        virtual double compute_segment_cost(arma::uword start, arma::uword end) = 0;
        // Overloaded version that is only valid for SplineOP.
        virtual double compute_segment_cost( size_t s
                                            , size_t t
                                            , const arma::Row<double>& p_s
                                            , const arma::Row<double>& p_t
                                            , const arma::Row<double>& v_s) {
                                            throw std::logic_error("This model does not support the 5-argument compute_segment_cost.");
                                        }
        virtual arma::Mat<double> get_data(){
            throw std::logic_error("This model does not store data.");
        }
        ////////////////////////////////////////////////////////////////
        // Member variables
        ////////////////////////////////////////////////////////////////
        arma::uword nobs;
        arma::uword ndims;
        ////////////////////////////////////////////////////////////////
        // Member functions
        ////////////////////////////////////////////////////////////////
        double sum_of_costs(std::vector<int>& bkps){
            double total_cost = 0.0;
            arma::uword start = 0;
            for (arma::uword bkp : bkps) {
                total_cost += compute_segment_cost(start, bkp);
                start = bkp;
            }
            return total_cost;
        };
        arma::uword get_nobs() {return nobs;}
        arma::uword get_ndims() {return ndims;}
};

using ModelPtr = Model_Base*; //std::unique_ptr<Model_Base>;

#endif // Model_Base_H
