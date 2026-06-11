# Vendored SwiftTD C++ core (MIT, Khurram Javed et al.).
# Sources copied from https://github.com/khurramjaved96/SwiftTD — see LICENSE
# and CITATION in this directory. Built as a static library; compiled with the
# upstream's own conventions (no project -Werror/-Wextra applied to vendored code).

add_library(swifttd STATIC src/SwiftTD.cpp)
add_library(il::swifttd ALIAS swifttd)

target_include_directories(swifttd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# Upstream targets C++17; the code is forward-compatible with the project's C++26.
target_compile_features(swifttd PUBLIC cxx_std_17)
set_target_properties(swifttd PROPERTIES POSITION_INDEPENDENT_CODE ON)
