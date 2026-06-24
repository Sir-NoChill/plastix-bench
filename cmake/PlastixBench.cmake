# PlastixBench — CMake helpers for the plastix-bench suite.
#
# Two functions are exposed:
#
#   plastix_add_plastix_bench(BENCH SRC)
#     Creates a target named `plastix_<BENCH>`, output binary named
#     `run_benchmark`, deposited at:
#         <build>/<BENCH>/plastix/run_benchmark
#     Linked against the Plastix library (imported target plastix::plastix)
#     and, when CUDA is enabled in the Plastix build, the CUDA toolkit.
#
#   plastix_add_cpp_bench(BENCH SRC)
#     Creates `cpp_<BENCH>`, deposited at:
#         <build>/<BENCH>/cpp/run_benchmark
#     Linked against OpenBLAS (detected via PkgConfig or FindBLAS by the
#     caller — the helper just consumes the variables that lookup left
#     behind).
#
# Each helper also calls `plastix_bench_binary_path` which materialises a
# stable, build-config-independent path at
#     <build>/<BENCH>/<IMPL>/run_benchmark
# (the orchestrator + sentinel wrappers find binaries through that path,
# mirroring the source layout <BENCH>/<IMPL>/). On multi-config generators
# the helper stamps a symlink so the path stays stable.
#
# Shared include dirs (`common/plastix/`, `common/cpp/`) are added by the
# helper rather than by each caller — that keeps the top-level CMakeLists to
# one line per benchmark.

if(NOT DEFINED PLASTIX_BENCH_ROOT)
    message(FATAL_ERROR
        "PlastixBench: PLASTIX_BENCH_ROOT must be set before including this "
        "file (point it at the suite root directory)")
endif()

# The shared utilities live under common/{cpp,plastix,pytorch}. Putting the
# common/ root on the include path lets each framework include its own header
# explicitly, e.g. `#include "cpp/common.hpp"` or `#include "plastix/common.hpp"`.
set(PLASTIX_BENCH_COMMON ${PLASTIX_BENCH_ROOT}/common)


# ----------------------------------------------------------------------------
# Helper: ensure `<build>/<bench>/<impl>/run_benchmark` exists regardless of
# generator behaviour. With single-config generators (Ninja, Unix Makefiles)
# RUNTIME_OUTPUT_DIRECTORY already lands there; with multi-config generators
# we add a post-build symlink to keep the path stable.
# ----------------------------------------------------------------------------
function(plastix_bench_binary_path TARGET BENCH IMPL)
    set(_dest ${CMAKE_BINARY_DIR}/${BENCH}/${IMPL}/run_benchmark)
    set_target_properties(${TARGET} PROPERTIES
        OUTPUT_NAME run_benchmark
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${BENCH}/${IMPL}
    )
    get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_is_multi_config)
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                    $<TARGET_FILE:${TARGET}> ${_dest}
            VERBATIM
        )
    endif()
endfunction()


# ----------------------------------------------------------------------------
# Helper: compile a plastix/ bench target through nvcc.
#
# Mirrors Plastix's own plastix_enable_cuda_on_target. When the suite is built
# with PLASTIX_BENCH_ENABLE_CUDA=ON the plastix/ TUs pull in CUB device code +
# <<<>>> through the Plastix headers (guarded there by PLASTIX_HAS_CUDA), so
# their .cpp sources must be handed to nvcc. A no-op when CUDA is off, so the
# host build never touches the CUDA toolchain.
# ----------------------------------------------------------------------------
function(plastix_bench_enable_cuda_on_target target)
    if(NOT PLASTIX_BENCH_ENABLE_CUDA)
        return()
    endif()
    get_target_property(_srcs ${target} SOURCES)
    foreach(_src ${_srcs})
        get_filename_component(_ext ${_src} EXT)
        if(_ext STREQUAL ".cpp")
            set_source_files_properties(${_src} TARGET_DIRECTORY ${target}
                                        PROPERTIES LANGUAGE CUDA)
        endif()
    endforeach()
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endfunction()


# ----------------------------------------------------------------------------
# Plastix-framework benchmark.
# ----------------------------------------------------------------------------
function(plastix_add_plastix_bench BENCH SRC)
    set(_target plastix_${BENCH})
    add_executable(${_target} ${SRC})
    target_include_directories(${_target} PRIVATE
        ${PLASTIX_BENCH_COMMON}
        ${PLASTIX_BENCH_ROOT}/${BENCH}/plastix)
    target_link_libraries(${_target} PRIVATE plastix::plastix)
    target_compile_options(${_target} PRIVATE -Wall -Wextra -Wpedantic)
    # Under PLASTIX_BENCH_ENABLE_CUDA, compile this TU with nvcc (no-op for the
    # host build).
    plastix_bench_enable_cuda_on_target(${_target})
    plastix_bench_binary_path(${_target} ${BENCH} plastix)
endfunction()


# ----------------------------------------------------------------------------
# Raw / OpenBLAS benchmark.
#
# The caller is expected to have already located OpenBLAS (PkgConfig or
# FindBLAS) and to expose either:
#   - OPENBLAS_FOUND + PkgConfig::OPENBLAS imported target, or
#   - OPENBLAS_INCLUDE_DIR + BLAS_LIBRARIES
# We probe both and link accordingly.
# ----------------------------------------------------------------------------
function(plastix_add_cpp_bench BENCH SRC)
    set(_target cpp_${BENCH})
    add_executable(${_target} ${SRC})
    target_include_directories(${_target} PRIVATE
        ${PLASTIX_BENCH_COMMON}
        ${PLASTIX_BENCH_ROOT}/${BENCH}/cpp)
    if(OPENBLAS_FOUND)
        target_link_libraries(${_target} PRIVATE PkgConfig::OPENBLAS)
    else()
        if(OPENBLAS_INCLUDE_DIR)
            target_include_directories(${_target} PRIVATE
                                       ${OPENBLAS_INCLUDE_DIR})
        endif()
        target_link_libraries(${_target} PRIVATE ${BLAS_LIBRARIES})
    endif()
    target_compile_options(${_target} PRIVATE -Wall -Wextra -Wpedantic -O3)
    plastix_bench_binary_path(${_target} ${BENCH} cpp)
endfunction()
