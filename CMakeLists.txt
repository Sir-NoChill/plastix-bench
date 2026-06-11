# plastix-bench — cross-framework benchmark suite for Plastix.
#
# Layout:
#   <bench>/{pytorch,plastix,cpp}/...
#   common/plastix/common.hpp            shared headers + helpers (Plastix impls)
#   common/cpp/{common.hpp, mlp.hpp}     used by C++ / OpenBLAS impls.
#   common/pytorch/common.py             used by Python impls.
#
# The per-benchmark wiring (target naming, include dirs, output paths,
# symlink fallback) lives in cmake/PlastixBench.cmake — this file is just an
# inventory of benches plus dependency discovery (Plastix + OpenBLAS).

cmake_minimum_required(VERSION 3.24)
project(plastix_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------------------------------------------------------------
# Plastix dependency.
#
# Preferred: an installed Plastix discovered via find_package (point
# CMAKE_PREFIX_PATH at its install prefix). Falls back to building Plastix
# from source — a local checkout via -DPLASTIX_SOURCE_DIR=/path/to/plastix,
# otherwise fetched from upstream. Either way the imported target is
# `plastix::plastix`.
# ----------------------------------------------------------------------------
find_package(plastix CONFIG QUIET)
if(NOT plastix_FOUND)
    include(FetchContent)
    # Plastix's own tests/examples/benchmarks are irrelevant here; build only
    # the library so the fetch stays cheap.
    set(PLASTIX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(PLASTIX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(PLASTIX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    if(PLASTIX_SOURCE_DIR)
        message(STATUS "plastix-bench: building Plastix from ${PLASTIX_SOURCE_DIR}")
        FetchContent_Declare(plastix SOURCE_DIR "${PLASTIX_SOURCE_DIR}")
    else()
        message(STATUS "plastix-bench: fetching Plastix from upstream git")
        FetchContent_Declare(plastix
            GIT_REPOSITORY https://github.com/dhanrajhira/plastix.git
            GIT_TAG master)
    endif()
    FetchContent_MakeAvailable(plastix)
endif()

# Make the shared bench helpers available; they expect PLASTIX_BENCH_ROOT to
# point at this directory.
set(PLASTIX_BENCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(PlastixBench)

# ----------------------------------------------------------------------------
# OpenBLAS lookup (shared across all `cpp/` implementations)
# ----------------------------------------------------------------------------
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(OPENBLAS QUIET IMPORTED_TARGET openblas)
endif()
if(NOT OPENBLAS_FOUND)
    find_package(BLAS REQUIRED)
    find_path(OPENBLAS_INCLUDE_DIR cblas.h
              HINTS /usr/include/openblas /usr/local/include/openblas
                    /opt/homebrew/opt/openblas/include
              REQUIRED)
endif()

# ----------------------------------------------------------------------------
# Discover benchmarks
# ----------------------------------------------------------------------------
#
# A benchmark may ship any subset of {pytorch, plastix, cpp}. We don't list
# them explicitly: glob every per-benchmark directory and probe for each
# implementation's source file. Adding a new benchmark = creating its
# directory.

file(GLOB _BENCHMARK_DIRS
     LIST_DIRECTORIES true
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     ${CMAKE_CURRENT_SOURCE_DIR}/*)

foreach(BENCH ${_BENCHMARK_DIRS})
    set(_bench_dir ${CMAKE_CURRENT_SOURCE_DIR}/${BENCH})
    if(NOT IS_DIRECTORY ${_bench_dir})
        continue()
    endif()
    # Skip shared / metadata directories: the leading-underscore convention
    # plus the shared `common/` and `cmake/` trees.
    string(SUBSTRING ${BENCH} 0 1 _LEAD)
    if(_LEAD STREQUAL "_" OR BENCH STREQUAL "common" OR BENCH STREQUAL "cmake")
        continue()
    endif()

    file(GLOB _plastix_srcs ${_bench_dir}/plastix/*.cpp)
    if(_plastix_srcs)
        list(GET _plastix_srcs 0 _src)
        plastix_add_plastix_bench(${BENCH} ${_src})
    endif()

    # The imprinting-learner bench reuses the standalone `il::agent` library
    # that ships under cpp/{include,src,third_party}. The .cpp glob would
    # otherwise miss it (no top-level cpp/*.cpp). We special-case it here so
    # the rest of the suite stays a one-liner.
    if(BENCH STREQUAL "09_imprintin_learner")
        if(NOT TARGET il::agent)
            # The sub-project's tests / examples / benchmarks each FetchContent
            # their own deps (gtest, benchmark) and define another harness; we
            # only need the library, so turn them off before recursing.
            set(IL_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
            set(IL_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
            set(IL_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
            add_subdirectory(${_bench_dir}/cpp _il_subproject EXCLUDE_FROM_ALL)
        endif()
        set(_il_src ${_bench_dir}/cpp/09_imprintin_learner.cpp)
        if(EXISTS ${_il_src})
            set(_target cpp_${BENCH})
            add_executable(${_target} ${_il_src})
            target_include_directories(${_target} PRIVATE
                ${PLASTIX_BENCH_COMMON}
                ${_bench_dir}/cpp
                ${_bench_dir}/cpp/examples)
            target_link_libraries(${_target} PRIVATE il::agent)
            target_compile_options(${_target} PRIVATE -O3)
            plastix_bench_binary_path(${_target} ${BENCH} cpp)
        endif()
        continue()
    endif()

    file(GLOB _cpp_srcs ${_bench_dir}/cpp/*.cpp)
    if(_cpp_srcs)
        list(GET _cpp_srcs 0 _src)
        plastix_add_cpp_bench(${BENCH} ${_src})
    endif()
endforeach()
