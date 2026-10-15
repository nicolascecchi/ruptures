/* pybind11 bindings for the compiled ``ruptures._core`` extension.
 *
 * The C++ sources (PELT algorithm and constant-normal cost model) are imported
 * verbatim from the `biblio-these` / `miniruptures` project; this file only
 * assembles the Python-facing bindings, extracting the relevant sections of
 * the upstream `py_bind_algorithms.cpp` and `py_bind_models.cpp` files.
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
        .def(py::init<Model_Base&>()
            ,"Constructor that takes a reference to a Model_Base object."
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
        "Compiled C++ core for ruptures (PELT + constant-normal cost), "
        "imported from the biblio-these / miniruptures project sources.";

    py::module_ models = m.def_submodule("models", "Cost models.");
    py::module_ algorithms = m.def_submodule("algorithms", "Detection algorithms.");

    init_models(models);
    init_algorithms(algorithms);
}