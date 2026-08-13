cmake_minimum_required(VERSION 3.28)

project(imprinting_learner
    VERSION 0.1.0
    DESCRIPTION "C++26 linear algebra playground backed by OpenBLAS"
    LANGUAGES CXX)

# --- Global C++ standard ---------------------------------------------------
set(CMAKE_CXX_STANDARD 23)            # C++23 -- only C++23 feature used is std::print (src/main.cpp)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release for single-config generators.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(IL_BUILD_TESTS      "Build unit tests"  ON)
option(IL_BUILD_BENCHMARKS "Build benchmarks"  ON)
option(IL_BUILD_EXAMPLES   "Build example harnesses" ON)
option(IL_PER_GENERATION_SAMPLING
    "Sample generation params once per step instead of per neuron" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# --- Dependencies ----------------------------------------------------------
# All third-party deps resolve through cmake/Dependencies.cmake, which prefers
# a host install (pkg-config / find_package) and falls back to FetchContent
# when the dependency is missing on the build machine.
include(Dependencies)

# --- Header-only core (domain types & concepts) ----------------------------
# Owns the public include root and the C++26 requirement; everything else
# consumes it. Header-only, so no compiled translation units.
add_library(il_core INTERFACE)
add_library(il::core ALIAS il_core)

target_include_directories(il_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(il_core INTERFACE cxx_std_23)

# --- Linear algebra library ------------------------------------------------
add_library(il_linalg src/linalg.cpp)
add_library(il::linalg ALIAS il_linalg)

target_link_libraries(il_linalg PUBLIC il::core il::blas)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(il_linalg PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- Vendored third-party code ---------------------------------------------
# SwiftTD: the temporal-difference update mechanism (target il::swifttd).
add_subdirectory(third_party/swifttd)

# --- Feature arena (struct-of-arrays agent state) --------------------------
add_library(il_features src/feature_arena.cpp)
add_library(il::features ALIAS il_features)

target_link_libraries(il_features PUBLIC il::core)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(il_features PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- Agent (feature arena + SwiftTD update mechanism) ----------------------
add_library(il_agent src/imprinting_learner.cpp)
add_library(il::agent ALIAS il_agent)

target_link_libraries(il_agent PUBLIC il::features il::swifttd)

if(IL_PER_GENERATION_SAMPLING)
    # Flip the compile-time generation-sampling toggle for il_agent and everything
    # that links it (harness, tests), keeping the inline constexpr consistent.
    target_compile_definitions(il_agent PUBLIC IL_SAMPLE_GENERATION_PARAMS_PER_NEURON=0)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(il_agent PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- Demo executable -------------------------------------------------------
add_executable(imprinting_demo src/main.cpp)
target_link_libraries(imprinting_demo PRIVATE il::linalg)

# --- Tests -----------------------------------------------------------------
if(IL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# --- Benchmarks ------------------------------------------------------------
if(IL_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# --- Examples --------------------------------------------------------------
# il-based audio-prediction harness (header-only dataset reader, no extra deps).
if(IL_BUILD_EXAMPLES)
    add_executable(il_audio_pred examples/il_audio_pred.cpp)
    target_link_libraries(il_audio_pred PRIVATE il::agent)
    target_include_directories(il_audio_pred PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(il_audio_pred PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(IL_BUILD_TESTS)
        add_executable(il_audio_pred_test examples/il_audio_pred_test.cpp)
        target_link_libraries(il_audio_pred_test
            PRIVATE il::agent GTest::gtest GTest::gtest_main)
        target_include_directories(il_audio_pred_test
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
        include(GoogleTest)
        gtest_discover_tests(il_audio_pred_test)
    endif()
endif()

message(STATUS "imprinting_learner ${PROJECT_VERSION} | "
    "C++${CMAKE_CXX_STANDARD} | tests=${IL_BUILD_TESTS} bench=${IL_BUILD_BENCHMARKS} "
    "examples=${IL_BUILD_EXAMPLES}")
