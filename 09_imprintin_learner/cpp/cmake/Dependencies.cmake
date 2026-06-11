# Dependencies.cmake
#
# Strategy for every third-party dependency:
#   1. Try to use a copy already installed on the host (pkg-config / find_package).
#   2. If it is missing, fetch and build it from source via FetchContent.
#
# Each dependency is exposed under a stable target name so the rest of the
# build never has to care which path was taken.

include(FetchContent)

# ---------------------------------------------------------------------------
# OpenBLAS  ->  il::blas
# ---------------------------------------------------------------------------
add_library(il_blas INTERFACE)
add_library(il::blas ALIAS il_blas)

find_package(PkgConfig QUIET)
set(_il_openblas_system FALSE)
if(PkgConfig_FOUND)
    pkg_check_modules(OpenBLAS IMPORTED_TARGET openblas)
    if(OpenBLAS_FOUND)
        set(_il_openblas_system TRUE)
    endif()
endif()

if(_il_openblas_system)
    message(STATUS "OpenBLAS: using host install (v${OpenBLAS_VERSION})")
    target_link_libraries(il_blas INTERFACE PkgConfig::OpenBLAS)
else()
    message(STATUS "OpenBLAS: not found on host -> building via FetchContent")
    # Keep the source build lean: BLAS only, no Fortran toolchain required.
    set(NOFORTRAN ON CACHE BOOL "" FORCE)
    set(BUILD_WITHOUT_LAPACK ON CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        OpenBLAS
        GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
        GIT_TAG        v0.3.33
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(OpenBLAS)
    target_link_libraries(il_blas INTERFACE openblas)
    # cblas.h lives at the OpenBLAS source root; generated config in the build tree.
    target_include_directories(il_blas SYSTEM INTERFACE
        "${openblas_SOURCE_DIR}"
        "${openblas_BINARY_DIR}"
        "${openblas_BINARY_DIR}/generated")
endif()

# ---------------------------------------------------------------------------
# GoogleTest  ->  GTest::gtest / GTest::gtest_main
# ---------------------------------------------------------------------------
if(IL_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY    https://github.com/google/googletest.git
        GIT_TAG           v1.17.0
        GIT_SHALLOW       TRUE
        FIND_PACKAGE_ARGS NAMES GTest)
    # If we end up building it, don't install gtest with our project.
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# ---------------------------------------------------------------------------
# Google Benchmark  ->  benchmark::benchmark / benchmark::benchmark_main
# ---------------------------------------------------------------------------
if(IL_BUILD_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING     OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL     OFF CACHE BOOL "" FORCE)
    # GCC/Clang trunk can emit new warnings; don't let -Werror break the fetch build.
    set(BENCHMARK_ENABLE_WERROR      OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY    https://github.com/google/benchmark.git
        GIT_TAG           v1.9.4
        GIT_SHALLOW       TRUE
        FIND_PACKAGE_ARGS NAMES benchmark)
    FetchContent_MakeAvailable(benchmark)
endif()
