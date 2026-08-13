# plastix-bench

Cross-framework benchmark suite for [Plastix](https://github.com/dhanrajhira/plastix).
Nine benchmarks — `01_static_etth1`, `02_idempotent_imp`, `03_bursty_elec2`,
`04_continuous_small_appliances`, `05_continuous_large_mackey_glass`,
`06_ccwc_ncp`, `07_esn_mackey_class`, `08_snn_shd`, `09_imprintin_learner` —
each shipping up to three implementations of the same algorithm (`pytorch/`,
`plastix/`, `cpp/`) under a single sentinel-and-CSV protocol so the
trajectories are directly comparable.

## Dependencies

- **CMake ≥ 3.24**, a C++20 compiler, and **OpenBLAS** (for the `cpp/`
  implementations). On Arch: `pacman -S openblas`.
- **Plastix** — discovered via `find_package(plastix)` or built from source
  (see below).
- **Python ≥ 3.10** with `uv` for the orchestrator and the `pytorch/`
  implementations. Dependencies are declared in `pyproject.toml`; `uv run`
  provisions them automatically.

## Setup — datasets and soundfonts

Datasets and soundfonts are **not committed to git** (see `.gitignore`). Fetch
them with the setup pipeline, which downloads each artifact through the same
path the benchmarks use and is idempotent (re-runs skip what already exists):

```bash
uv run python setup.py            # CSVs, MNIST, SHD (+.plxbin cache), soundfont
uv run python setup.py --help     # stage list and options
```

Stages (`--stages a,b,...` to select, `--skip a,b` to exclude):

| Stage | Benches | What it fetches |
|---|---|---|
| `csvs` | 01, 03, 04 | ETTh1 / elec2 / appliances CSVs |
| `mnist` | 06 | sequential MNIST (torchvision) |
| `shd` | 08 | Spiking Heidelberg Digits (tonic) + the `.plxbin` cache the C++ bench reads |
| `soundfont` | 09 | `FluidR3_GM.sf2` (~140 MB) for the audio-prediction dataset |
| `audio` | 09 | synthesise `09_imprintin_learner/.../output/dataset.bin` |

By default everything **except `audio`** runs (the SHD download is several GB).
The `audio` stage needs a **system FluidSynth** (the Python bindings come from
the `09_imprintin_learner/cpp/examples/` sub-project, which the stage runs via
`uv`); invoke it explicitly:

```bash
# Arch: sudo pacman -S fluidsynth   |   Debian: sudo apt install fluidsynth
uv run python setup.py --stages soundfont,audio
```

If FluidSynth is missing, the `audio` stage prints install instructions and
skips rather than failing.

## Building the C++ binaries

The suite consumes Plastix as an external dependency. The recommended path is
to install Plastix and point CMake at the install prefix:

```bash
# 1. Install Plastix somewhere (from a Plastix checkout):
cmake -S /path/to/plastix -B /tmp/plastix-build \
      -DPLASTIX_BUILD_TESTS=OFF -DPLASTIX_BUILD_EXAMPLES=OFF \
      -DPLASTIX_BUILD_BENCHMARKS=OFF \
      -DCMAKE_INSTALL_PREFIX=/path/to/plastix-install
cmake --build /tmp/plastix-build -j
cmake --install /tmp/plastix-build

# 2. Configure + build the benchmark suite against it:
cmake -S . -B build-host -DCMAKE_PREFIX_PATH=/path/to/plastix-install
cmake --build build-host -j
```

If `find_package(plastix)` finds nothing, the configure step falls back to
building Plastix from source. Point it at a local checkout to avoid a network
fetch:

```bash
cmake -S . -B build-host -DPLASTIX_SOURCE_DIR=/path/to/plastix
```

…otherwise it fetches Plastix from upstream git. Either way the suite links the
imported target `plastix::plastix`.

C++ binaries land at `build-host/<bench>/<impl>/run_benchmark`, mirroring the
source layout. The matching `cpp_wrapper.py` sentinel `execvp`s them.

## Running the benchmarks

Run everything through the orchestrator from the repo root:

```bash
# Run everything (full datasets) and aggregate into _results/runs.csv:
uv run python orchestrator.py run

# Quick smoke pass (each bench takes its own --quick path):
uv run python orchestrator.py run --quick

# Filter to one bench + pass extras through with `--`:
uv run python orchestrator.py run --bench 09_imprintin_learner \
    -- --max-steps 4000 --log-every 1000

# Enumerate discovered (bench, impl) pairs:
uv run python orchestrator.py list

# Render the standard plot set:
uv run python orchestrator.py plot
uv run python orchestrator.py plot --kinds phase_stats,memory
```

Outputs land under `_results/` (per-run summary CSVs + `runs.csv` aggregate)
and `_plots/` (PNGs). See the `orchestrator.py` module docstring for the full
plot-kind list (`accuracy`, `overlay`, `walltime`, `memory`, `phases`,
`phase_stats`).

## Paper summary tables

`just tables` (run automatically at the end of every `run-*` pass) aggregates the
archived runs into two paper-ready CSVs under `_results/`, one row per benchmark
keyed by a short acronym (`bench_meta.ACRONYM`):

- `phase_table.csv` — per-framework runtime **fractions** by phase
  (`fwd/bwd/upd/prune/grow/uncat`, summing to 100%).
- `memory_table.csv` — per-framework **memory** breakdown in MiB
  (`dataset/weights/overhead/scratch/max`).

Both pull `plastix/pytorch/cpp` from `runs_cpu.csv` and `cuda` from `runs_gpu.csv`
and append the per-bench structural characteristics from `bench_meta.py`. Run them
standalone with `just phase-table` / `just memory-table`. The memory table's
`<fw>_vram` column is peak GPU VRAM (polled per-process via nvidia-smi; 0 on CPU).

`just tables-gpu` (auto-run after `just run-gpu`) writes `{phase,memory}_table_gpu.csv`
— the same tables but with **every** framework sourced from the GPU pass, so GPU
timings + VRAM show for all of them (not just `cuda`).

## Paper figure data

`just figures-data` (`figures_data.py`) projects the same archives into the CSV
schema the paper's pgfplots figures actually consume — per-work-unit wall seconds
per phase, and MiB per RSS bucket — under
`_results/figures/{segmented_bar_perf,memory_occupancy}/data/`.

This is a different schema from the summary tables above: the figures plot
*absolute* seconds per unit of work (the table plots 0-100% fractions), and they
carry a sixth `uncat` phase so the decomposition is loss-less.

**Benchmark set.** The figures draw the five benchmarks in
`figures_data.CANON_BENCHES` (Sparse, Bursty, Cont-S, Cont-L, Imprint) — the ones
with complete four-framework coverage and working phase timers in the current
archive. `docs/figure_bench_selection.md` records why each of the others (Dense,
LTC-sine, ESN, SNN, XL-NN) is excluded, and what the retained five actually show
— including the fact that the CPU throughput claim does **not** reproduce.
Changing the length of `CANON_BENCHES` no longer requires editing the `.tex`: the
figures size themselves from `\sbpnb`, which the standalone wrappers set.

**Two views.** Each figure is emitted twice:

- `*.csv` — **CPU view**: `plastix`/`pytorch`/`cpp` on CPU + the `cuda` backend.
- `*_gpu.csv` — **GPU view**: every column from the GPU pass, so the CUDA backend
  is compared against GPU-resident PyTorch and JAX. There is no cpp GPU impl, so
  the third bar slot carries JAX (under the reused `cpp_*` column prefix, so one
  `figure.tex` renders either view). Build it with the `standalone_gpu.tex`
  wrapper in each figure's `combined/` directory.

**Data root.** `--results-dir` selects the tree to read (archives *and* the
per-bench `summary.csv` files that the work-axis normalization falls back on).
It defaults to `./_results`, so if you untar a results bundle somewhere else,
pass it explicitly:

```sh
just figures-data -- --results-dir ../_results \
                     --figures-dir ../plastix-paper/figures
```

## Profiling (nsys / nvprof)

Profile a bench's **GPU** binary under an NVIDIA profiler; reports land in
`_profiles/`:

```bash
just profile 06_ccwc_ncp                 # auto (nsys, else nvprof)
just profile 06_ccwc_ncp --tool both     # nsys + nvprof (--tool is a profile.py flag)
just profile 06_ccwc_ncp -- --max-steps 200   # `--` forwards flags to the binary
just profile-scaling                     # the 11_scaling_imprint CUDA plastix binary
```

Auto-selects **nsys** (Nsight Systems → `.nsys-rep` timeline + kernel/API summary
CSVs) and falls back to **nvprof** (legacy → per-kernel CSV). Needs the CUDA build
(`build-cuda`) and a GPU; a missing profiler is reported and skipped. See
`profile.py`.

## Scaling sweep

`just scaling` runs the `11_scaling_imprint` bench (imprinting-style net) from
~10k to ~10M neurons, Plastix vs PyTorch, one subprocess per size for a clean
per-size peak RSS, and writes the long-format `_results/scaling_table.csv`
(`neurons, framework, wall_per_step_ns, peak_rss_mb, status`). It records each
framework's OOM/timeout ceiling and is opt-in (not part of `run-*`).
`just scaling -- --max-neurons 100000` runs a quick subset.

## Optional frameworks: JAX + bio-inspired baselines

Heavy/niche frameworks are optional extras (`uv sync --extra jax|snn|nengo|evosax`),
kept out of the default env:

- **JAX** ports the **whole suite (01-10)** (`<bench>/jax/`, on `common/jax/`);
  a `jax_*` column in the tables. Static-topology benches jit cleanly; the
  dynamic ones expose XLA's recompile-on-shape-change wall (bench 09 ~160×) —
  see **`docs/jax_expressibility.md`**.
- **SNN frameworks** — spiking ports of **01-05** in snnTorch (`<bench>/snn/`) and
  **01-02** in Norse (`<bench>/norse/`), first-class impls with `snn_*`/`norse_*`
  table columns and a `--timesteps` knob. Finding: all five are expressible
  (eager torch + stateless-per-forward LIF makes runtime grow/prune natural) — see
  **`docs/snn_expressibility.md`**.
- **Baseline frameworks** (TensorNEAT, an SNN stack, Nengo, evosax) are
  paradigm-different comparison baselines under `baselines/`; `just baselines`
  runs them into the separate `_results/baselines_table.csv` (uninstalled ones
  report `status=skipped`). See `baselines/README.md`.

## Layout convention

```
<bench>/<impl>/run_benchmark.py     sentinel (Python trainer or thin exec)
<bench>/<impl>/<bench>.cpp          C++ source (for cpp/plastix impls)
common/                             shared utilities, on the C++ include path
  cpp/{common.hpp, mlp.hpp}         raw-C++ helpers: CliArgs, SummaryWriter,
                                    PhaseTimer  ->  #include "cpp/common.hpp"
  plastix/common.hpp                Plastix-side equivalents (CliArgs, edge-set
                                    utilities, PhaseTimer) -> "plastix/common.hpp"
  pytorch/common.py                 Python helpers: StructuralLog, PhaseTimer,
                                    output paths
  pytorch/cpp_wrapper.py            sentinel that execs the compiled binary
  pytorch/data/                     datasets fetched by setup.py (git-ignored)
cmake/PlastixBench.cmake            per-bench target wiring + binary paths
```

C++ implementations include their framework's shared header by its `common/`
sub-path — `#include "cpp/common.hpp"` (and `"cpp/mlp.hpp"`) for the OpenBLAS
impls, `#include "plastix/common.hpp"` for the Plastix impls — since the helper
puts `common/` on the include path. Python sentinels add `common/pytorch` to
`sys.path`.

A bench is discovered automatically: drop `run_benchmark.py` in
`<bench>/<impl>/`, and the orchestrator's glob picks it up. C++ binaries are
wired by `cmake/PlastixBench.cmake`, glob-picked from `<bench>/<impl>/`; no
CMake edits are needed unless the bench needs an unusual dependency (see the
`09_imprintin_learner` special-case in `CMakeLists.txt` for the `il::agent`
library link).

## PhaseTimer and the summary CSV schema

Every training inner loop wraps its `forward` / `loss` / `backward` /
`update` / `prune` / `grow` / `reset` phases with a `PhaseTimer` (both languages
ship one — see `common/{pytorch,cpp,plastix}/common.{py,hpp}`). The timer uses
Welford's online algorithm so the per-run summary CSV carries **mean + std**
for each phase plus `step_count`, `step_ns_mean`, and `other_ns_mean` (slack
that the phase marks don't cover). The orchestrator also polls
`/proc/<descendant>/status` every 50 ms and reports both peak and
arithmetic-mean `VmRSS` across the descendant set.

`runs.csv` carries one row per (bench, impl, tag), with these phase columns
hoisted up so plots can read everything from a single file:

```
bench, impl, tag, wall_seconds, metric, metric_kind,
peak_rss_kb, mean_rss_kb, mem_samples,
step_count, step_ns_mean,
forward_ns_mean,    forward_ns_std,
loss_ns_mean,       loss_ns_std,
backward_ns_mean,   backward_ns_std,
update_ns_mean,     update_ns_std,
prune_ns_mean,      prune_ns_std,
grow_ns_mean,       grow_ns_std,
reset_ns_mean,      reset_ns_std,
other_ns_mean,
```

**`step_count` is a sampling count, not training work.** It tallies how many
`PhaseTimer.StepDone()` ticks the inner loop produced — that is the SGD
granularity, not the amount of training. The same bench can show different
`step_count`s across impls when one runs per-sample SGD and another runs
minibatched SGD, even though both completed the same number of epochs or
rounds. When comparing implementations of one bench, the unit of training work
is the bench's natural axis — `epochs`, `rounds`/`round_`, or `max_steps` —
and these are mirrored in the per-bench HP structs across pytorch/plastix/cpp
so the trajectories are directly comparable. The orchestrator's accuracy plot
uses whichever of `epoch`/`step`/`round`/`round_` the history JSONL emits as
the x-axis, in that priority order.

### Phase semantics across impls

Each phase column means the same thing across pytorch/plastix/cpp:

- `forward`   — model evaluation (one pass through the network).
- `loss`      — scalar loss (and dL/dActivation staging in Plastix).
- `backward`  — gradient computation only. In TD-style impls (the imprinting
  learner, the recurrent NCP, the SNN with e-prop) the trace-and-update is
  algorithmically inseparable from the gradient, so it's bundled in `backward`
  and `update` stays at zero.
- `update`    — optimizer step (SGD on each weight; `DoUpdateUnit` +
  `DoUpdateConn` in Plastix).
- `prune`     — `DoPruneUnits + DoPruneConnections` in Plastix; tenure/remove
  (the shrink half) in the imprinting learner library. Zero for static nets.
- `grow`      — `DoAddUnits + DoAddConnections` in Plastix; generate/snapshot
  (the spawn half) in the imprinting learner library. Zero for static nets.
  (`prune` + `grow` together are the old single `structural` phase.)
- `reset`     — `DoResetGlobalState` in Plastix; zero elsewhere.
- `other`     — wall-step minus the sum of phase means: data movement, Python
  loop overhead, `.item()` syncs, host↔device transfers.

If a bench can't separate `backward` from `update` (one library call does
both), bundle into `backward` and leave `update` at zero. This keeps the
columns comparable: PyTorch's `loss.backward()+opt.step()` naturally splits,
Plastix's `DoBackwardPass`/`DoUpdateConn` also splits, TD streams don't — the
convention preserves that honestly.

## Adding a new bench

1. Make a `<bench>/` dir with `pytorch/`, `plastix/`, and/or `cpp/` subdirs.
2. Each impl's `run_benchmark.py` is either the trainer itself (Python) or
   `from cpp_wrapper import main; main()` (C++ sentinel).
3. Wrap the training loop in a `PhaseTimer`; emit `**timer.summary_fields(wall)`
   into the summary dict (Python) or call `Timer.WriteSummary(S, Wall)` before
   `S.Write(...)` (C++).
4. Make sure the summary CSV carries at least `wall_seconds`, a `test_*`
   metric, `n_units`/`n_edges`, plus the phase columns.
5. The C++ target is glob-picked from `<bench>/<impl>/` by
   `cmake/PlastixBench.cmake` — no CMake edits needed unless the bench needs an
   unusual dependency.
