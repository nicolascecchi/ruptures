cmake_minimum_required(VERSION 3.23)
project(ruptures LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_PYTHON_BINDINGS "Build the Python bindings (_core module, Cython extensions, version.py)" ON)
option(RUPTURES_REQUIRE_CORE "Fail the configure when the C++ core headers are missing instead of skipping them" OFF)

# Python package location: repository checkout (python/) vs flattened sdist (.)
# produced by scikit-build-core's sdist.force-include (see python/pyproject.toml).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/pyproject.toml")
    set(RUPTURES_PYTHON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python")
else()
    set(RUPTURES_PYTHON_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
message(STATUS "ruptures: Python package directory: ${RUPTURES_PYTHON_DIR}")

# --- Optional C++ core (git submodules under extern/) -------------------------
# The core is an optional accelerator: ruptures falls back to the pure Python
# implementations when it is not built. Its dependencies live in git submodules,
# so a checkout that has not run `git submodule update --init` cannot build it.
set(RUPTURES_MISSING_CORE "")
foreach(_dir
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/armadillo/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/carma/include"
)
    if(NOT EXISTS "${_dir}")
        list(APPEND RUPTURES_MISSING_CORE "${_dir}")
    endif()
endforeach()

if(RUPTURES_MISSING_CORE)
    if(RUPTURES_REQUIRE_CORE)
        message(FATAL_ERROR
            "ruptures: missing core directories: ${RUPTURES_MISSING_CORE}. "
            "Run 'git submodule update --init' (or build from the sdist)."
        )
    else()
        message(WARNING
            "ruptures: skipping the C++ core (ruptures._core will not be built); "
            "missing directories: ${RUPTURES_MISSING_CORE}. "
            "Run 'git submodule update --init' to build it."
        )
    endif()
endif()

add_library(carma INTERFACE)
target_include_directories(carma INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/carma/include"
)

if(NOT RUPTURES_MISSING_CORE)
    # BLAS/LAPACK are linked directly: the constant-normal model needs only
    # cumsums, but the rank model (svd/pinv) and the linear models (solve) call
    # into LAPACK. The Armadillo wrapper stays unused (the header-only submodules
    # provide the C++ API).
    if(APPLE)
        # FindBLAS is not reliable at picking up Accelerate otherwise.
        set(BLA_VENDOR "Apple")
    endif()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)

    add_library(Models STATIC)
    target_sources(Models
        PRIVATE
            cpp/Models/Model_ar.cpp
            cpp/Models/Model_clinear.cpp
            cpp/Models/Model_constant_binomial.cpp
            cpp/Models/Model_constant_exponential.cpp
            cpp/Models/Model_constant_normal.cpp
            cpp/Models/Model_constant_poisson.cpp
            cpp/Models/Model_constant_rank.cpp
            cpp/Models/Model_cosine.cpp
            cpp/Models/Model_l1.cpp
            cpp/Models/Model_linear.cpp
            cpp/Models/Model_mahalanobis.cpp
            cpp/Models/Model_normal.cpp
            cpp/Models/Model_rbf.cpp
            cpp/Models/Model_var.cpp
        PUBLIC FILE_SET HEADERS
            BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
            FILES
                cpp/Models/Model_Base.h
                cpp/Models/Model_ar.h
                cpp/Models/Model_clinear.h
                cpp/Models/Model_constant_binomial.h
                cpp/Models/Model_constant_exponential.h
                cpp/Models/Model_constant_normal.h
                cpp/Models/Model_constant_poisson.h
                cpp/Models/Model_constant_rank.h
                cpp/Models/Model_cosine.h
                cpp/Models/Model_l1.h
                cpp/Models/Model_linear.h
                cpp/Models/Model_mahalanobis.h
                cpp/Models/Model_normal.h
                cpp/Models/Model_rbf.h
                cpp/Models/Model_var.h
    )
    target_include_directories(Models PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/extern/armadillo/include"
    )
    target_compile_definitions(Models PUBLIC ARMA_DONT_USE_WRAPPER)

    add_library(Algorithms STATIC)
    target_sources(Algorithms
        PRIVATE
            cpp/Algorithms/Algorithm_pelt.cpp
            cpp/Algorithms/Algorithm_SegmentNeighborhood.cpp
        PUBLIC FILE_SET HEADERS
            BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
            FILES
                cpp/Algorithms/Algorithm_Base.h
                cpp/Algorithms/Algorithm_pelt.h
                cpp/Algorithms/Algorithm_SegmentNeighborhood.h
    )
    target_include_directories(Algorithms PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/extern/armadillo/include"
    )
    target_compile_definitions(Algorithms PUBLIC ARMA_DONT_USE_WRAPPER)
endif()

# --- Python bindings ----------------------------------------------------------
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
    message(STATUS "ruptures: Python interpreter: ${Python3_EXECUTABLE}")

    set(PYBIND11_FINDPYTHON OFF)
    find_package(pybind11 CONFIG REQUIRED)

    # version.py is generated at build time from the PEP 517 version that
    # scikit-build-core provides (no edits to the Python sources).
    # SKBUILD_PROJECT_VERSION is capped to 4 components for CMake validity;
    # SKBUILD_PROJECT_VERSION_FULL is the exact packaged version string.
    set(RUPTURES_VERSION "${SKBUILD_PROJECT_VERSION_FULL}")
    if("${RUPTURES_VERSION}" STREQUAL "")
        set(RUPTURES_VERSION "0.0.0+unknown")
    endif()
    configure_file(
        "${RUPTURES_PYTHON_DIR}/version.py.in"
        "${CMAKE_CURRENT_BINARY_DIR}/version.py"
        @ONLY
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/version.py"
            DESTINATION ruptures)

    # --- Cython extensions (always built: hard runtime imports) ---------------
    # The .pxd sits next to the .pyx (Cython finds it there); the src/ include
    # path resolves the fully-qualified self-cimports used by the .pyx files.
    set(RUPTURES_SRC_DIR "${RUPTURES_PYTHON_DIR}/src")

    set(_DETECTION_DIR "${RUPTURES_PYTHON_DIR}/src/ruptures/detection/_detection")
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ekcpd.c"
        COMMAND "${Python3_EXECUTABLE}" -m cython -3
            -I "${RUPTURES_SRC_DIR}"
            --output-file "${CMAKE_CURRENT_BINARY_DIR}/ekcpd.c"
            "${_DETECTION_DIR}/ekcpd.pyx"
        DEPENDS "${_DETECTION_DIR}/ekcpd.pyx" "${_DETECTION_DIR}/ekcpd.pxd"
        COMMENT "Cythonizing ekcpd.pyx"
        VERBATIM
    )
    Python3_add_library(ekcpd MODULE WITH_SOABI
        "${CMAKE_CURRENT_BINARY_DIR}/ekcpd.c"
        "${_DETECTION_DIR}/ekcpd_computation.c"
        "${_DETECTION_DIR}/ekcpd_pelt_computation.c"
        "${_DETECTION_DIR}/kernels.c"
    )
    target_include_directories(ekcpd PRIVATE "${_DETECTION_DIR}")
    install(TARGETS ekcpd DESTINATION ruptures/detection/_detection)

    set(_UTILS_DIR "${RUPTURES_PYTHON_DIR}/src/ruptures/utils/_utils")
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/convert_path_matrix.c"
        COMMAND "${Python3_EXECUTABLE}" -m cython -3
            -I "${RUPTURES_SRC_DIR}"
            --output-file "${CMAKE_CURRENT_BINARY_DIR}/convert_path_matrix.c"
            "${_UTILS_DIR}/convert_path_matrix.pyx"
        DEPENDS "${_UTILS_DIR}/convert_path_matrix.pyx" "${_UTILS_DIR}/convert_path_matrix.pxd"
        COMMENT "Cythonizing convert_path_matrix.pyx"
        VERBATIM
    )
    Python3_add_library(convert_path_matrix MODULE WITH_SOABI
        "${CMAKE_CURRENT_BINARY_DIR}/convert_path_matrix.c"
        "${_UTILS_DIR}/convert_path_matrix_c.c"
    )
    target_include_directories(convert_path_matrix PRIVATE "${_UTILS_DIR}")
    install(TARGETS convert_path_matrix DESTINATION ruptures/utils/_utils)

    # --- Optional compiled backend (_core) ------------------------------------
    if(NOT RUPTURES_MISSING_CORE)
        pybind11_add_module(_core "${RUPTURES_PYTHON_DIR}/bindings/_core.cpp")
        # carma bridges NumPy and Armadillo, so the NumPy headers are needed.
        target_include_directories(_core PRIVATE "${Python3_NumPy_INCLUDE_DIRS}")
        target_link_libraries(_core PRIVATE
            Models
            Algorithms
            carma
            ${BLAS_LIBRARIES}
            ${LAPACK_LIBRARIES}
        )
        install(TARGETS _core DESTINATION ruptures)
    endif()
endif()