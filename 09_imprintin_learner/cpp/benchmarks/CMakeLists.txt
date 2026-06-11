add_executable(il_benchmarks linalg_benchmark.cpp)

target_link_libraries(il_benchmarks PRIVATE
    il::linalg
    benchmark::benchmark
    benchmark::benchmark_main)
