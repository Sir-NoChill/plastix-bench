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
  implementations. Dependencies (`numpy`, `scipy`, `matplotlib`, `torch`)
  are declared in `pyproject.toml`; `uv run` provisions them automatically.

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

## Layout convention

```
<bench>/<impl>/run_benchmark.py     sentinel (Python trainer or thin exec)
<bench>/<impl>/<bench>.cpp          C++ source (for cpp/plastix impls)
_shared_python/common.py            Python helpers: StructuralLog,
                                    PhaseTimer, output paths
_shared_python/cpp_wrapper.py       sentinel that execs the compiled binary
_shared_cpp/common.hpp              raw-C++ helpers: CliArgs,
                                    SummaryWriter, PhaseTimer
_shared_plastix/common.hpp          Plastix-side equivalents (CliArgs,
                                    edge-set utilities, PhaseTimer)
cmake/PlastixBench.cmake            per-bench target wiring + binary paths
```

A bench is discovered automatically: drop `run_benchmark.py` in
`<bench>/<impl>/`, and the orchestrator's glob picks it up. C++ binaries are
wired by `cmake/PlastixBench.cmake`, glob-picked from `<bench>/<impl>/`; no
CMake edits are needed unless the bench needs an unusual dependency (see the
`09_imprintin_learner` special-case in `CMakeLists.txt` for the `il::agent`
library link).

## PhaseTimer and the summary CSV schema

Every training inner loop wraps its `forward` / `loss` / `backward` /
`update` / `structural` / `reset` phases with a `PhaseTimer` (both languages
ship one — see `_shared_{python,cpp,plastix}/common.{py,hpp}`). The timer uses
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
structural_ns_mean, structural_ns_std,
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
- `structural` — `Prune* + Add*` in Plastix; tenure/remove/generate in the
  imprinting learner library. Zero for pytorch.
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
