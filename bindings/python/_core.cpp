/* pybind11 bindings for the compiled ``ruptures._core`` extension.
 *
 * The C++ sources (PELT algorithm and the cost model classes) are imported
 * from the `biblio-these` / `miniruptures` project; this file only
 * assembles the Python-facing bindings, extracting the relevant sections of
 * the upstream `py_bind_algorithms.cpp` and `py_bind_models.cpp` files.
 * Algorithm_pelt is further extended with min_size/jump parameters following
 * the grid semantics of ruptures' pure Python Pelt; the model classes are
 * copied verbatim, except the ones implemented here (l1, rbf, cosine, ar,
 * mahalanobis, clinear) for parity with ruptures' Python cost functions. Model_var
 * is the OOP adaptation of an imported functional prefiltered VAR cost
 * (ridge prefiltering dropped), with cumulative statistics per segment cost.
 * Upstream's Model_Factory.h and Model_splineop.h (and its
 * Functions/Faulhaber dependency, which Model_splineop.h includes) are not
 * synced yet: the factory includes Model_splineop.h, so both are left for a
 * dedicated splineop pass.
 *
 * The Python-visible names (``models.ConstantNormal``, ``algorithms.PELT``,
 * ...) follow the upstream binding names; the C++ class names are left
 * unchanged.
 *
 * NOTE: <carma> must be included before <armadillo> so that carma can install
 * the NumPy allocator hooks into Armadillo (see carma docs).
 */
#include <carma>

#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "Algorithms/Algorithm_Base.h"
#include "Algorithms/Algorithm_pelt.h"
#include "Models/Model_Base.h"
#include "Models/Model_constant_normal.h"
#include "Models/Model_constant_poisson.h"
#include "Models/Model_constant_exponential.h"
#include "Models/Model_constant_binomial.h"
#include "Models/Model_constant_rank.h"
#include "Models/Model_linear.h"
#include "Models/Model_l1.h"
#include "Models/Model_rbf.h"
#include "Models/Model_cosine.h"
#include "Models/Model_ar.h"
#include "Models/Model_var.h"
#include "Models/Model_mahalanobis.h"
#include "Models/Model_clinear.h"
#include "Models/Model_normal.h"

namespace py = pybind11;

void init_models(py::module_ &models) {
    py::class_<Model_Base>(models, "Base"
    ,"Base class for cost models.")
        .def("get_nobs", &Model_Base::get_nobs
        ,"Get the number of observations"
        )
        .def("get_ndims", &Model_Base::get_ndims
        ,"Get the number of dimensions")
        .def("sum_of_costs", &Model_Base::sum_of_costs
        ,"Compute the sum of costs of a segmentation defined by its breakpoints"
        , py::arg("bkps")
    );

    // Bind Model_constant_normal class
    py::class_<Model_constant_normal, Model_Base>(models
        , "ConstantNormal"
        , "Model for detecting changes in mean with Gaussian noise (L2 cost).")
        .def(py::init<const arma::Mat<double>&>()
            ,"Constructor that takes a 2D numpy array as input data"
            ,py::arg("data")
        )
        .def("compute_segment_cost", &Model_constant_normal::compute_segment_cost
            , "Compute the cost of a segment defined by start and end indices"
            , py::arg("start"), py::arg("end")
        )
        .def("get_cumsum_y", &Model_constant_normal::get_cumsum_y
            , "Get the cumulative sum of the data"
        )
        .def("get_cumsum_squared_y", &Model_constant_normal::get_cumsum_squared_y
            , "Get the cumulative sum of squares of the data"
    );

    // Bind Model_constant_poisson class
    py::class_<Model_constant_poisson, Model_Base>(models
        , "ConstantPoisson"
        , "Model for detecting changes in mean with Poisson noise.")
        .def(py::init<const arma::Mat<double>&>()
            ,"Constructor that takes a 2D numpy array as input data"
            ,py::arg("data")
        )
        .def("compute_segment_cost", &Model_constant_poisson::compute_segment_cost
            , "Compute the cost of a segment defined by start and end indices"
            , py::arg("start"), py::arg("end")
        )
        .def("get_cumsum_y", &Model_constant_poisson::get_cumsum_y
            , "Get the cumulative sum of the data"
    );

    // Bind Model_constant_exponential class
    py::class_<Model_constant_exponential, Model_Base>(models
        , "ConstantExponential"
        , "Model for detecting changes in mean with Exponential noise.")
        .def(py::init<const arma::Mat<double>&>()
            ,"Constructor that takes a 2D numpy array as input data"
            ,py::arg("data")
        )
        .def("compute_segment_cost", &Model_constant_exponential::compute_segment_cost
            , "Compute the cost of a segment defined by start and end indices"
            , py::arg("start"), py::arg("end")
        )
        .def("get_cumsum_y", &Model_constant_exponential::get_cumsum_y
            , "Get the cumulative sum of the data"
    );

    // Bind Model_constant_binomial class
    py::class_<Model_constant_binomial, Model_Base>(models
        , "ConstantBinomial"
        , "Model for detecting changes in mean with Binomial noise.")
        .def(py::init([](const arma::Mat<double>& data, int m) {
                if (m == 1) {
                    PyErr_WarnEx(PyExc_UserWarning, "Binomial model requires m > 1", 1);
                }
                return std::make_unique<Model_constant_binomial>(data, m);
            })
            ,"Constructor that takes a 2D numpy array and the number of "
            "trials m of the binomial"
            ,py::arg("data"), py::arg("m") = 1
        )
        .def("compute_segment_cost", &Model_constant_binomial::compute_segment_cost
            , "Compute the cost of a segment defined by start and end indices"
            , py::arg("start"), py::arg("end")
        )
        .def("get_cumsum_y", &Model_constant_binomial::get_cumsum_y
            , "Get the cumulative sum of the data"
    );

    // Bind Model_constant_rank class
    py::class_<Model_constant_rank, Model_Base>(models
        , "ConstantRank"
        , "Model for detecting changes in rank with rank-based noise.")
        .def(py::init<const arma::Mat<double>&>()
            ,"Constructor that takes a 2D numpy array as input data"
            ,py::arg("data")
        )
        .def("compute_segment_cost", &Model_constant_rank::compute_segment_cost
            , "Compute the cost of a segment defined by start and end indices"
            , py::arg("start"), py::arg("end")
        )
        .def("get_centered_ranks", &Model_constant_rank::get_centered_ranks
            , "Get the centered ranks of the data"
        )
        .def("get_cusum_ranks", &Model_constant_rank::get_cusum_ranks
            , "Get the cumulative sum of ranks of the data"
        )
        .def("get_inv_cov", &Model_constant_rank::get_inv_cov
            , "Get the inverse covariance matrix of the ranks"
    );

    // Bind Model_linear class
    py::class_<Model_linear, Model_Base>(models
        , "CostLinear"
        , "Model for detecting changes in linear regression coefficients with Gaussian noise.")
        .def(py::init<const arma::Mat<double>&>()
            ,"Constructor that takes a 2D numpy array as input data"
            ,py::arg("data")
        )
        .def("compute_segment_cost", &Model_linear::compute_segment_cost
            , "Compute the cost of a segment defined by start and end indices"
            , py::arg("start"), py::arg("end")
    );

    // Bind Model_l1 class
    py::class_<Model_l1, Model_Base>(models
        , "CostL1"
        , "Model for detecting changes with the least absolute deviation cost.")
        .def(py::init<const arma::Mat<double>&>()
            ,"Constructor that takes a 2D numpy array as input data"
            ,py::arg("data")
        )
        .def("compute_segment_cost", &Model_l1::compute_segment_cost
            , "Compute the cost of a segment defined by start and end indices"
            , py::arg("start"), py::arg("end")
    );

    // Bind Model_rbf class
    py::class_<Model_rbf, Model_Base>(models
        , "CostRbf"
        , "Model for detecting changes with a Gaussian-kernel (RBF) cost.")
        .def(py::init<const arma::Mat<double>&, double>()
            ,"Constructor that takes a 2D numpy array and the RBF width "
            "gamma (resolving the median heuristic is left to Python's "
            "CostRbf.fit())."
            ,py::arg("data"), py::arg("gamma")
        )
        .def("compute_segment_cost", &Model_rbf::compute_segment_cost
            , "Compute the cost of a segment defined by start and end indices"
            , py::arg("start"), py::arg("end")
    );

    // Bind Model_cosine class
    py::class_<Model_cosine, Model_Base>(models
        , "CostCosine"
        , "Model for detecting changes with a cosine-similarity kernel cost.")
        .def(py::init<const arma::Mat<double>&>()
            ,"Constructor that takes a 2D numpy array as input data"
            ,py::arg("data")
        )
        .def("compute_segment_cost", &Model_cosine::compute_segment_cost
            , "Compute the cost of a segment defined by start and end indices"
            , py::arg("start"), py::arg("end")
    );

    // Bind Model_ar class
    py::class_<Model_ar, Model_Base>(models
        , "CostAR"
        , "Model for detecting changes in autoregressive coefficients.")
        .def(py::init<const arma::Mat<double>&, const arma::Mat<double>&>()
            ,"Constructor that takes the signal and the lagged covariate "
            "matrix (with intercept) prepared by Python's CostAR.fit()."
            ,py::arg("signal"), py::arg("covar")
        )
        .def("compute_segment_cost", &Model_ar::compute_segment_cost
            , "Compute the cost of a segment defined by start and end indices"
            , py::arg("start"), py::arg("end")
    );

    // Bind Model_var class
    py::class_<Model_var, Model_Base>(models
        , "CostVar"
        , "Model for detecting changes in vector autoregressive coefficients "
        "(multivariate generalisation of CostAR).")
        .def(py::init<const arma::Mat<double>&, arma::uword, bool>()
            ,"Constructor that takes a 2D numpy array, the autoregressive "
            "order and whether to fit an intercept; the lagged covariates are "
            "built in C++ (the first order rows have no lags and are zero)."
            ,py::arg("data"), py::arg("order"), py::arg("intercept") = true
        )
        .def("compute_segment_cost", &Model_var::compute_segment_cost
            , "Compute the cost of a segment defined by start and end indices"
            , py::arg("start"), py::arg("end")
    );

    // Bind Model_mahalanobis class
    py::class_<Model_mahalanobis, Model_Base>(models
        , "CostMl"
        , "Model for detecting changes with a Mahalanobis-type kernel cost.")
        .def(py::init<const arma::Mat<double>&, const arma::Mat<double>&>()
            ,"Constructor that takes a 2D numpy array and the metric matrix "
            "prepared by Python's CostMl.fit()."
            ,py::arg("data"), py::arg("metric")
        )
        .def("compute_segment_cost", &Model_mahalanobis::compute_segment_cost
            , "Compute the cost of a segment defined by start and end indices"
            , py::arg("start"), py::arg("end")
    );

    // Bind Model_clinear class
    py::class_<Model_clinear, Model_Base>(models
        , "CostCLinear"
        , "Model for detecting changes with Python CostCLinear's "
        "continuity-constrained linear cost.")
        .def(py::init<const arma::Mat<double>&>()
            ,"Constructor that takes a 2D numpy array as input data"
            ,py::arg("data")
        )
        .def("compute_segment_cost", &Model_clinear::compute_segment_cost
            , "Compute the cost of a segment defined by start and end indices"
            , py::arg("start"), py::arg("end")
    );

    // Bind Model_normal class
    py::class_<Model_normal, Model_Base>(models
        , "CostNormal"
        , "Model for detecting changes in the distribution of a Gaussian "
        "signal: the segment log-likelihood cost, with CostNormal's v1.1.5 "
        "diagonal bias (costs/costnormal.py, add_small_diag=True).")
        .def(py::init<const arma::Mat<double>&>()
            ,"Constructor that takes a 2D numpy array as input data"
            ,py::arg("data")
        )
        .def("compute_segment_cost", &Model_normal::compute_segment_cost
            , "Compute the cost of a segment defined by start and end indices"
            , py::arg("start"), py::arg("end")
    );
}

void init_algorithms(py::module_ &algorithms) {
    // Bind Algorithm_Base
    py::class_<Algorithm_Base>(algorithms, "Base"
    ,"Base class for changepoint detection algorithms.")
        .def("get_changepoints", &Algorithm_Base::get_changepoints
            , "Get the detected change-points"
        )
        .def("get_admissible_size_history", &Algorithm_Base::get_admissible_size_history
            , "Get the history of admissible size at each time step"
        )
        .def("get_cost_history", &Algorithm_Base::get_cost_history
            , "Get the history of costs at each time step"
        )
        .def("get_admissible_points", &Algorithm_Base::get_admissible_points
            , "Get the admissible points vector"
        )
        .def("get_admissible_size_at_t", &Algorithm_Base::get_admissible_size_at_t
            , "Get the size of admissible points at a specific time step"
        )
        .def("get_best_prev_idx", &Algorithm_Base::get_best_prev_idx
            , "Get the best previous index vector for backtracking"
    );
    // Bind Algorithm_pelt class
    py::class_<Algorithm_pelt, Algorithm_Base>(algorithms, "PELT"
    ,"PELT algorithm for changepoint detection with inequality based pruning.")
        .def(py::init<Model_Base&, arma::uword, arma::uword>()
            ,"Constructor that takes a reference to a Model_Base object, "
            "the minimum segment length (min_size) and the step between "
            "candidate change points (jump); defaults reproduce the "
            "single-argument behaviour."
            , py::arg("model"), py::arg("min_size") = 1, py::arg("jump") = 1
            , py::keep_alive<1, 2>()
        )
        .def("predict", py::overload_cast<double>(&Algorithm_pelt::predict)
            , "Predict changepoints with a given penalty value."
            , py::arg("beta")
        )
        .def("backtrack", &Algorithm_pelt::backtrack
            , "Backtrack to find the optimal change-points."
    );
}

PYBIND11_MODULE(_core, m) {
    m.doc() =
        "Compiled C++ core for ruptures (PELT and cost model classes), "
        "imported from the biblio-these / miniruptures project sources.";

    py::module_ models = m.def_submodule("models", "Cost models.");
    py::module_ algorithms = m.def_submodule("algorithms", "Detection algorithms.");

    init_models(models);
    init_algorithms(algorithms);
}