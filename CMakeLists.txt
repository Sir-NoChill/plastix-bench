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
# CUDA toggle.
#
# The plastix/ benchmark implementations include the Plastix headers, which
# under a CUDA-enabled Plastix pull in CUB device code and the <<<>>> dispatch
# path — those only compile through nvcc. PLASTIX_BENCH_ENABLE_CUDA wires that
# up: it forces the from-source Plastix dependency to build with CUDA and turns
# on the CUDA language here so the plastix/ TUs are compiled by nvcc (see
# plastix_bench_enable_cuda_on_target in cmake/PlastixBench.cmake).
#
# Default OFF: the suite builds host-only, links no CUDA runtime, and runs with
# no NVIDIA driver/toolkit present — exactly as before. Only flip this on to
# benchmark the GPU path. The cpp/ and pytorch/ impls are unaffected either way.
# ----------------------------------------------------------------------------
option(PLASTIX_BENCH_ENABLE_CUDA
       "Build the plastix/ impls through nvcc against a CUDA-enabled Plastix" OFF)

if(PLASTIX_BENCH_ENABLE_CUDA)
    # Make the from-source Plastix (FetchContent, below) build its CUDA path.
    # Harmless for an already-installed Plastix found via find_package.
    set(PLASTIX_ENABLE_CUDA ON CACHE BOOL "" FORCE)

    # Locate the toolkit first (honours -DCUDAToolkit_ROOT, as Plastix's own
    # build does) so we can point the CUDA language at nvcc even when it is not
    # on PATH. Also provides CUDA::cudart in this scope for robust cross-scope
    # linking (Plastix links it transitively through plastix::plastix).
    find_package(CUDAToolkit REQUIRED)
    if(NOT CMAKE_CUDA_COMPILER)
        if(NOT CUDAToolkit_NVCC_EXECUTABLE)
            message(FATAL_ERROR
                "PLASTIX_BENCH_ENABLE_CUDA=ON but no CUDA compiler (nvcc) was "
                "found. Pass -DCUDAToolkit_ROOT=/path/to/cuda, or leave "
                "PLASTIX_BENCH_ENABLE_CUDA=OFF for a host-only build.")
        endif()
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
    endif()
    # Match the arches Plastix compiles for (Ampere+; 32-bit float atomics).
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING
            "CUDA arches to compile the bench plastix/ impls for" FORCE)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(PLASTIX_BENCH_HAVE_CUDA ON)
endif()

# ----------------------------------------------------------------------------
# Per-bench cuda/ implementations (cuBLAS, on-device). These need only a CUDA
# toolkit — NOT a CUDA-enabled Plastix — so gate them on whether CUDA is
# *installed*, independent of PLASTIX_BENCH_ENABLE_CUDA. When no toolkit is
# present the cuda/ targets are silently skipped (host build is unaffected).
# ----------------------------------------------------------------------------
if(NOT PLASTIX_BENCH_HAVE_CUDA)
    # Locate the toolkit (honours -DCUDAToolkit_ROOT; FindCUDAToolkit also probes
    # /usr/local/cuda). nvcc is often not on PATH, so point the CUDA language at
    # the discovered nvcc before enabling it.
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        if(NOT CMAKE_CUDA_COMPILER AND CUDAToolkit_NVCC_EXECUTABLE)
            set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
        endif()
        include(CheckLanguage)
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            set(PLASTIX_BENCH_HAVE_CUDA ON)
            if(NOT CMAKE_CUDA_ARCHITECTURES)
                set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
            endif()
            set(CMAKE_CUDA_STANDARD 20)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
            message(STATUS "plastix-bench: CUDA toolkit detected (${CUDAToolkit_NVCC_EXECUTABLE}) — building cuda/ impls")
        endif()
    endif()
    if(NOT PLASTIX_BENCH_HAVE_CUDA)
        message(STATUS "plastix-bench: no CUDA toolkit — skipping cuda/ impls")
    endif()
endif()

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
    find_package(BLAS)
    find_path(OPENBLAS_INCLUDE_DIR cblas.h
              HINTS /usr/include/openblas /usr/local/include/openblas
                    /opt/homebrew/opt/openblas/include)
endif()

# The cpp/ implementations are the only ones that need OpenBLAS. The plastix/
# and pytorch/ impls build without it, so treat BLAS as optional and just skip
# the cpp/ targets when it is missing rather than failing configuration.
if(OPENBLAS_FOUND OR (BLAS_FOUND AND OPENBLAS_INCLUDE_DIR))
    set(PLASTIX_BENCH_HAVE_BLAS ON)
else()
    set(PLASTIX_BENCH_HAVE_BLAS OFF)
    message(WARNING
        "plastix-bench: OpenBLAS / cblas.h not found — skipping all cpp/ "
        "(OpenBLAS) implementations. The plastix/ and pytorch/ impls still "
        "build.")
endif()

# ----------------------------------------------------------------------------
# LAPACKE lookup — needed ONLY by 07_esn_mackey_class/cpp, which solves the
# ESN read-out via LAPACKE_dgesv. OpenBLAS ships the BLAS (cblas.h) but not
# always the LAPACKE C interface (lapacke.h + LAPACKE_* symbols); on Debian
# that lives in liblapacke-dev. Treat it as optional and skip just the 07 cpp
# target when absent — every other cpp/ impl needs only cblas. Install
# liblapacke-dev and re-configure to pull 07's cpp impl back in.
# ----------------------------------------------------------------------------
if(PkgConfig_FOUND)
    pkg_check_modules(LAPACKE QUIET IMPORTED_TARGET lapacke)
endif()
if(NOT LAPACKE_FOUND)
    find_path(LAPACKE_INCLUDE_DIR lapacke.h
              HINTS /usr/include /usr/include/openblas /usr/local/include)
    find_library(LAPACKE_LIBRARY NAMES lapacke)
endif()
if(LAPACKE_FOUND OR (LAPACKE_INCLUDE_DIR AND LAPACKE_LIBRARY))
    set(PLASTIX_BENCH_HAVE_LAPACKE ON)
else()
    set(PLASTIX_BENCH_HAVE_LAPACKE OFF)
    message(WARNING
        "plastix-bench: LAPACKE (lapacke.h) not found — skipping the "
        "07_esn_mackey_class cpp/ impl only (it needs LAPACKE_dgesv). Install "
        "liblapacke-dev and re-configure to include it. All other cpp/ impls "
        "build normally.")
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
        if(NOT PLASTIX_BENCH_HAVE_BLAS)
            continue()
        endif()
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

        # Pure-CUDA impl (cuBLAS, on-device). Only built when CUDA is enabled;
        # compiled by nvcc and linked against cuBLAS. Output mirrors the layout
        # the sentinel/orchestrator expect: <build>/<bench>/cuda/run_benchmark.
        if(PLASTIX_BENCH_HAVE_CUDA)
            file(GLOB _cuda_srcs ${_bench_dir}/cuda/*.cu)
            if(_cuda_srcs)
                list(GET _cuda_srcs 0 _cusrc)
                set(_ctarget cuda_${BENCH})
                add_executable(${_ctarget} ${_cusrc})
                set_source_files_properties(${_cusrc} PROPERTIES LANGUAGE CUDA)
                target_include_directories(${_ctarget} PRIVATE
                    ${PLASTIX_BENCH_COMMON}
                    ${_bench_dir}/cpp/examples)
                target_link_libraries(${_ctarget} PRIVATE CUDA::cublas CUDA::cudart)
                target_compile_options(${_ctarget} PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
                    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
                set_target_properties(${_ctarget} PROPERTIES
                    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
                plastix_bench_binary_path(${_ctarget} ${BENCH} cuda)
            endif()
        endif()
        continue()
    endif()

    # 07's cpp impl is the lone LAPACKE consumer; skip it when LAPACKE is
    # absent rather than failing the whole build (cblas-only impls still build).
    if(BENCH STREQUAL "07_esn_mackey_class" AND NOT PLASTIX_BENCH_HAVE_LAPACKE)
        continue()
    endif()

    file(GLOB _cpp_srcs ${_bench_dir}/cpp/*.cpp)
    if(_cpp_srcs AND PLASTIX_BENCH_HAVE_BLAS)
        list(GET _cpp_srcs 0 _src)
        plastix_add_cpp_bench(${BENCH} ${_src})
        # Link LAPACKE into the lone target that needs it.
        if(BENCH STREQUAL "07_esn_mackey_class")
            if(LAPACKE_FOUND)
                target_link_libraries(cpp_${BENCH} PRIVATE PkgConfig::LAPACKE)
            else()
                target_include_directories(cpp_${BENCH} PRIVATE ${LAPACKE_INCLUDE_DIR})
                target_link_libraries(cpp_${BENCH} PRIVATE ${LAPACKE_LIBRARY})
            endif()
        endif()
    endif()

    # Pure-CUDA impl (cuBLAS), generic across benches. Only when CUDA is on.
    if(PLASTIX_BENCH_HAVE_CUDA)
        file(GLOB _cuda_srcs ${_bench_dir}/cuda/*.cu)
        if(_cuda_srcs)
            list(GET _cuda_srcs 0 _cusrc)
            set(_ctarget cuda_${BENCH})
            add_executable(${_ctarget} ${_cusrc})
            set_source_files_properties(${_cusrc} PROPERTIES LANGUAGE CUDA)
            target_include_directories(${_ctarget} PRIVATE
                ${PLASTIX_BENCH_COMMON} ${_bench_dir}/cpp ${_bench_dir}/cpp/examples)
            # cuBLAS for the GEMMs; cuSOLVER for impls with a dense linear solve
            # (07 ESN ridge read-out). cusolver links harmlessly into the rest.
            target_link_libraries(${_ctarget} PRIVATE
                CUDA::cublas CUDA::cudart CUDA::cusolver)
            target_compile_options(${_ctarget} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
                $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
            set_target_properties(${_ctarget} PROPERTIES
                CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
            plastix_bench_binary_path(${_ctarget} ${BENCH} cuda)
        endif()
    endif()
endforeach()
