add_executable(il_tests
    linalg_test.cpp
    feature_concept_test.cpp
    feature_arena_test.cpp
    imprinting_learner_test.cpp
    swifttd_smoke_test.cpp)

target_link_libraries(il_tests PRIVATE
    il::core
    il::linalg
    il::features
    il::agent
    il::swifttd
    GTest::gtest
    GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(il_tests)
