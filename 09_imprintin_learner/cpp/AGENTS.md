# AGENTS.md

Agent onboarding for **imprinting-learner-c** — a C++26 linear-algebra playground
backed by OpenBLAS, with GoogleTest unit tests and Google Benchmark micro-benchmarks.

See [.agents/architecture.md](.agents/architecture.md) for the rationale behind these choices.

## Toolchain

- **Standard:** C++26 (`CMAKE_CXX_STANDARD 26`, extensions off). Needs a recent
  compiler — verified with GCC 16.1 and Clang 22.
- **Build system:** CMake (>= 3.28) + Ninja, driven through `CMakePresets.json`.

## Common commands

```sh
# Configure (presets: default | debug | clang)
cmake --preset default

# Build everything
cmake --build build/default

# Run unit tests
ctest --preset default                 # or: ctest --test-dir build/default --output-on-failure

# Run the demo and benchmarks
./build/default/imprinting_demo
./build/default/benchmarks/il_benchmarks --benchmark_min_time=0.1s

# Run the imprinting learner on the audio-prediction benchmark (0 = all steps).
# Sweep hyperparameters without recompiling via IL_* env vars.
./build/default/il_audio_pred examples/output/dataset.bin 0 il_predictions.csv
IL_ETA=2.0 IL_ALPHA=0.005 ./build/default/il_audio_pred examples/output/dataset.bin 10000

# Overlay predictions on the dataset visuals (spectrogram + reward events + features)
examples/venv/bin/python examples/plot_il.py --predictions il_predictions.csv \
    --data-dir examples/output --time-range 0 180 --output examples/output/plots/il_overlay.png
# Full-hour overview: --time-range 0 3600 --no-spectrogram

# Reproducible env via Nix
nix develop          # dev shell with the full toolchain + deps
nix build            # builds + runs ctest as the derivation check
nix flake check --no-build
```

`compile_commands.json` is symlinked at the repo root for clangd.

## Layout

```
CMakeLists.txt            top-level targets, options (IL_BUILD_TESTS/BENCHMARKS)
cmake/Dependencies.cmake  all third-party deps resolve here
include/imprinting/       public headers: feature.hpp, feature_arena.hpp,
                          imprinting_learner.hpp, linalg.hpp
src/                      library impl (linalg.cpp, feature_arena.cpp,
                          imprinting_learner.cpp) + demo (main.cpp)
tests/                    GoogleTest suite
benchmarks/               Google Benchmark suite (microbenchmarks)
third_party/swifttd/      vendored SwiftTD core (target il::swifttd) + LICENSE/CITATION
examples/                 audio-prediction benchmark: il_audio_pred.cpp (our harness)
                          + dataset.hpp (APBD reader) + prepare-cpp.py (dataset gen)
                          + plot_il.py (prediction overlay) + sweep_il.py (run+plot
                          parameter sweeps) + run.sh (one-shot build→generate→run)
flake.nix / flake.lock    Nix dev shell + package output
```

## Conventions

- Namespace everything under `il`. Public headers live in `include/imprinting/`.
- Matrices are **row-major, dense, double**; hand storage straight to CBLAS.
  Use `blasint` (from `<cblas.h>`) for CBLAS dimension/stride args, not `int`.
- Library code compiles with `-Wall -Wextra -Wpedantic`. Keep it warning-clean.
- Minimal comments — explain *why*, not *what*. No doc blocks on obvious code.
- Tests use `gtest_discover_tests`; benchmarks link `benchmark::benchmark_main`
  (no hand-written `main`).

## Adding a third-party dependency

Wire it through `cmake/Dependencies.cmake`, never inline in a target file.
Resolve **host-first, FetchContent fallback**, and expose a stable target name:

```cmake
FetchContent_Declare(
    foo
    GIT_REPOSITORY https://github.com/org/foo.git
    GIT_TAG        vX.Y.Z
    GIT_SHALLOW    TRUE
    FIND_PACKAGE_ARGS NAMES Foo)   # tries find_package(Foo) before fetching
FetchContent_MakeAvailable(foo)
```

Current targets: `il::core` (header-only domain concepts/types), `il::features`
(SOA feature arena), `il::agent` (arena + SwiftTD = the learner), `il::linalg`,
`il::blas` (OpenBLAS), `il::swifttd` (vendored TD learner),
`GTest::gtest[_main]`, `benchmark::benchmark[_main]`.

Vendored code lives under `third_party/` (not FetchContent) — currently SwiftTD,
copied from upstream with provenance headers and an unmodified algorithm. Preserve
its `LICENSE`/`CITATION` and avoid editing the algorithm in place. Local edits vs.
upstream (Step/Predict arithmetic untouched): include path namespaced to
`<swifttd/SwiftTD.h>`; `SwiftTDBinaryFeatures` weights are a `std::span` bindable to
external storage (`bindWeights`) for in-place updates, plus `weights()`/`betas()` accessors.

## Domain model

`il::core` holds header-only domain abstractions, independent of linalg. The first
is the `il::Feature` concept (`feature.hpp`): a learnable unit exposing a weight, an
eligibility trace (`std::span<float>`), a per-feature step size, and lifecycle
(`Status`: Tenure / TenureTrack / Idle) + role (`FeatureType`: Observation / Pattern
/ Memory / Output) metadata. Eligibility traces + per-feature step sizes point at a
TD/RL-style ("imprinting") learner.

The `Feature` concept also requires `getActivation() -> bool` (true == 1 for the
prediction sum). Feature state is stored struct-of-arrays in `il::features`'
`FeatureArena` (`feature_arena.hpp`): common columns (weight, step_size, trace, status,
type, activation, type_slot) indexed by a global feature id, plus compact per-type
stores. `FeatureRef{&arena, id}` is a proxy that models `il::Feature`, so the SOA layout
stays type-checked (`static_assert(Feature<FeatureRef>)`). The arena does the **forward
pass**: `step(input)` recomputes activations (observations from input; patterns via a
k/n fraction over fixed-width int16 membership; memory via delay/window counters) and
returns the GVF prediction `Σ weightᵢ·activationᵢ`. See `.agents/architecture.md` for
the activation semantics and design choices.

The **update mechanism** (backward pass) is SwiftTD (`il::swifttd`, `<swifttd/SwiftTD.h>`):
TD learning with per-feature step-size adaptation and a bound on the learning rate. Three
variants — `SwiftTDNonSparse`, `SwiftTDBinaryFeatures`, `SwiftTD` (index/value pairs).

`il::agent`'s `ImprintingLearner` ties the two together. `step(input, reward)`: bind
SwiftTD's weight span to the arena's column → arena forward pass → collect active feature
ids → `SwiftTDBinaryFeatures::Step` (binary fits the 0/1 activations) writes the arena's
weights **in place** (no copy-back) → apply the **tenure policy** (hysteresis on `|weight|`):
promote above `tenure_track_threshold` / `tenure_threshold`, demote below those ×
`demotion_factor` (< 1, so demotion is harder to trigger than promotion). Observation
features are always `Tenure`; generated pattern/memory features start `Idle`. All knobs live
in one `HyperParams` struct (capacity, trace_dim, thresholds, SwiftTD params) — use it
rather than ad-hoc test values.

Feature **generation** (`generateFeatures`, off unless `k_pattern`/`k_memory` > 0) spawns
random-sampled pattern/memory features from the previous step's active-tenured pool, bounded by
a τ-vs-`eta` budget, immediately triggering each. Feature **removal** (`removeFeatures`, off
unless `epsilon_z > 0`) swap-pops `Idle`, decayed (`z < e^β·εᶻ`), unreferenced features across
the arena *and* SwiftTD's vectors in lockstep; observations are never removed. **Ghosts are
avoided by deferral**: a feature a survivor still references is kept until its dependents go
first, so no dangling refs form (see `.agents/architecture.md` for the trade-off vs. the thesis's
eager-removal-with-ghosts). Adapted per-feature step sizes still live inside SwiftTD;
`getStepSize()` returns the initial `alpha`.
