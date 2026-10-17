import os

from setuptools import Extension, setup

HERE = os.path.abspath(os.path.dirname(__file__))

cython_ext_modules = [
    Extension(
        "ruptures.detection._detection.ekcpd",
        sources=[
            "src/ruptures/detection/_detection/ekcpd.pyx",
            "src/ruptures/detection/_detection/ekcpd_computation.c",
            "src/ruptures/detection/_detection/ekcpd_pelt_computation.c",
            "src/ruptures/detection/_detection/kernels.c",
        ],
    ),
    Extension(
        "ruptures.utils._utils.convert_path_matrix",
        sources=[
            "src/ruptures/utils/_utils/convert_path_matrix.pyx",
            "src/ruptures/utils/_utils/convert_path_matrix_c.c",
        ],
    ),
]

# Header-only dependencies of the compiled backend, provided by the git
# submodules under extern/ (see .gitmodules). pybind11 and the NumPy headers
# come from their installed packages.
CPP_INCLUDE_DIRS = [
    os.path.join(HERE, "cpp"),
    os.path.join(HERE, "extern", "armadillo", "include"),
    os.path.join(HERE, "extern", "carma", "include"),
]


def pybind_ext_modules():
    """Compiled change point detection backend, if its headers are available.

    The backend is an optional accelerator: ``ruptures.Pelt`` falls back to the
    pure Python implementation when it is not built. Its dependencies live in
    git submodules, so a checkout that has not run ``git submodule update
    --init`` cannot build it.
    """
    missing = [d for d in CPP_INCLUDE_DIRS if not os.path.isdir(d)]
    if missing:
        print(
            "warning: skipping ruptures._core, missing include directories "
            f"{missing}. Run 'git submodule update --init' to build it."
        )
        return []

    import numpy
    from pybind11.setup_helpers import Pybind11Extension

    return [
        Pybind11Extension(
            "ruptures._core",
            sources=[
                "bindings/python/_core.cpp",
                "cpp/Algorithms/Algorithm_pelt.cpp",
                "cpp/Models/Model_constant_normal.cpp",
                "cpp/Models/Model_constant_poisson.cpp",
                "cpp/Models/Model_constant_exponential.cpp",
                "cpp/Models/Model_constant_binomial.cpp",
                "cpp/Models/Model_constant_rank.cpp",
                "cpp/Models/Model_linear.cpp",
                "cpp/Models/Model_linear_continuous.cpp",
                "cpp/Models/Model_l1.cpp",
                "cpp/Models/Model_rbf.cpp",
                "cpp/Models/Model_cosine.cpp",
                "cpp/Models/Model_ar.cpp",
                "cpp/Models/Model_mahalanobis.cpp",
                "cpp/Models/Model_clinear.cpp",
                "cpp/Models/Model_normal.cpp",
            ],
            # carma bridges NumPy and Armadillo, so the NumPy headers are needed.
            include_dirs=CPP_INCLUDE_DIRS + [numpy.get_include()],
            cxx_std=17,
            # BLAS/LAPACK are linked directly, mirroring upstream
            # miniruptures' find_package(BLAS)/find_package(LAPACK): the
            # constant-normal model needs only cumsums, but the rank model
            # (svd/pinv) and the linear models (solve) call into LAPACK.
            # The Armadillo wrapper stays unused (the header-only submodules
            # provide the C++ API).
            libraries=["lapack", "blas"],
            define_macros=[
                ("ARMA_DONT_USE_WRAPPER", None),
            ],
        )
    ]


if __name__ == "__main__":
    from Cython.Build import cythonize

    # cythonize() rebuilds Extension objects, so the pybind11 extension is kept
    # out of its input and only concatenated to the result.
    ext_modules = cythonize(cython_ext_modules, language_level="3")

    cmdclass = {}
    pybind_exts = pybind_ext_modules()
    if pybind_exts:
        from pybind11.setup_helpers import build_ext

        ext_modules += pybind_exts
        cmdclass["build_ext"] = build_ext

    setup(ext_modules=ext_modules, cmdclass=cmdclass)
