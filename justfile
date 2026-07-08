# plastix-bench — task runner.
#
# Wraps setup.py / orchestrator.py / cmake into a single entry point so the
# whole suite (datasets → CPU + CUDA builds → runs → graphs) is one command.
#
#   just                 # list recipes
#   just bench-all       # the full pipeline: build both, run both, plot both
#
# CPU vs CUDA: the C++ `plastix/` implementations are compiled twice — host-only
# into `build/`, and through nvcc against a CUDA-enabled Plastix into
# `build-cuda/` (60-71% GPU util observed at runtime). The `pytorch/` impls run
# on CPU in both passes as a fixed reference baseline (orchestrator forces
# `--device cpu`). The `cpp/` (OpenBLAS) impls build against the host OpenBLAS;
# 07's cpp impl additionally needs LAPACKE (liblapacke-dev) and is skipped at
# configure time if that is absent — every other cpp/ impl needs only cblas.

set shell := ["bash", "-euo", "pipefail", "-c"]

# --- configuration ----------------------------------------------------------

# Local Plastix checkout used as the from-source dependency (avoids a network
# fetch). Override on the command line: `just plastix_src=/elsewhere build-all`.
plastix_src := justfile_directory() / ".." / "plastix"

# CUDA toolkit root (honoured by find_package(CUDAToolkit)).
cuda_root := "/usr/local/cuda"

# Build directories for the two configurations.
build_cpu  := "build"
build_cuda := "build-cuda"

# Implementations to run. plastix is the CPU-vs-GPU subject; cpp (OpenBLAS) and
# pytorch are reference baselines. jax (CPU) and the SNN frameworks snn
# (snnTorch) / norse are further baselines, ported for benches 01-05 (jax/snn)
# and 01-02 (norse); each runs only where a <bench>/<impl>/ dir exists, skipped
# elsewhere. Non-plastix impls run on CPU in either pass.
impls := "cpp,plastix,pytorch,jax,snn,norse"

# Number of parallel build jobs.
jobs := num_cpus()

# --- default ----------------------------------------------------------------

# List available recipes.
default:
    @just --list --unsorted

# --- datasets / environment -------------------------------------------------

# Provision the Python environment from pyproject.toml / uv.lock.
sync:
    uv sync

# Download datasets + soundfont (idempotent; skips what already exists).
setup *ARGS:
    uv run python setup.py {{ARGS}}

# Synthesise the 09 audio dataset (needs a system FluidSynth installed).
# Configurable length + spectrum "boxes": total observation dim = FREQ_BINS *
# MAG_BINS. Regenerates automatically when any of these differ from the
# existing dataset. e.g. `just setup-audio 300 100 100` -> 300s, 10000 boxes.
setup-audio DURATION="60" FREQ_BINS="50" MAG_BINS="50" SEED="42":
    uv run python setup.py --stages soundfont,audio \
        --audio-duration {{DURATION}} --audio-seed {{SEED}} \
        --audio-freq-bins {{FREQ_BINS}} --audio-mag-bins {{MAG_BINS}}

# --- configure + build ------------------------------------------------------

# Configure the host-only (CPU) build.
configure-cpu:
    cmake -S . -B {{build_cpu}} -DPLASTIX_SOURCE_DIR="{{plastix_src}}"

# Configure the CUDA build (plastix/ TUs compiled by nvcc against CUDA Plastix).
configure-cuda:
    cmake -S . -B {{build_cuda}} \
        -DPLASTIX_SOURCE_DIR="{{plastix_src}}" \
        -DPLASTIX_BENCH_ENABLE_CUDA=ON \
        -DCUDAToolkit_ROOT="{{cuda_root}}"

# Build the CPU binaries.
build-cpu: configure-cpu
    cmake --build {{build_cpu}} -j {{jobs}}

# Build the CUDA binaries.
build-cuda: configure-cuda
    cmake --build {{build_cuda}} -j {{jobs}}

# Build both configurations.
build-all: build-cpu build-cuda

# --- inspect ----------------------------------------------------------------

# Enumerate discovered (bench, impl) sentinels.
list:
    uv run python orchestrator.py list

# Aggregate the archived runs into the paper's per-phase fraction table.
# Collapses runs_cpu.csv (plastix/pytorch/cpp) + runs_gpu.csv (cuda) into one
# row per benchmark of phase percentages (fwd/bwd/upd/prune/grow/uncat) plus the
# per-bench structural characteristics. Writes _results/phase_table.csv.
# Pass extra flags after `--`, e.g. `just phase-table -- -o /tmp/table.csv`.
phase-table *ARGS:
    uv run python phase_table.py {{ARGS}}

# Aggregate the archived runs into the paper's memory breakdown table
# (dataset/weights/overhead/scratch/max in MiB, per framework). Writes
# _results/memory_table.csv. Same source mapping as phase-table.
memory-table *ARGS:
    uv run python memory_table.py {{ARGS}}

# Regenerate both summary tables (phase + memory) from the current archives.
# Run automatically at the end of every benchmark pass; also runnable standalone.
# The canonical tables are CPU-based (plastix/pytorch/cpp/jax/snn/norse from the
# CPU pass) + the dedicated cuda impl from the GPU pass.
tables: phase-table memory-table

# GPU view of both tables: EVERY framework sourced from the GPU pass
# (runs_gpu.csv), so the vram column + GPU timings show for all frameworks, not
# just cuda. Writes _results/{phase,memory}_table_gpu.csv. Run after `just run-gpu`.
tables-gpu:
    uv run python phase_table.py --gpu
    uv run python memory_table.py --gpu

# Opt-in scaling sweep: the 11_scaling_imprint bench (imprinting-style net) from
# ~10k to ~10M neurons, plastix vs pytorch, one subprocess per (size,framework)
# for clean per-size peak RSS. Records each framework's OOM/timeout ceiling and
# writes the long-format _results/scaling_table.csv (pivots into compute- and
# memory-vs-neurons line charts). HEAVY — not part of run-cpu/run-gpu. Pass
# extra flags after `--`, e.g. `just scaling -- --max-neurons 100000` for a
# quick subset. Builds the CPU binaries first.
scaling *ARGS: build-cpu
    uv run python scaling.py --build-dir {{build_cpu}} {{ARGS}}

# Profile a bench's GPU binary under nsys / nvprof; reports land in _profiles/.
# Auto-selects nsys (falls back to nvprof); override with `--tool nsys|nvprof|both`.
# profile.py flags (--tool/--impl) go straight after the bench; reserve `--` for
# flags forwarded to the bench binary. Needs the CUDA build + a GPU.
#   just profile 06_ccwc_ncp --tool both          # both profilers
#   just profile 06_ccwc_ncp -- --max-steps 200   # forward a flag to the binary
profile BENCH *ARGS: build-cuda
    uv run python profile.py --bench {{BENCH}} --build-dir {{build_cuda}} {{ARGS}}

# Convenience: profile the scaling bench's CUDA-built plastix binary.
profile-scaling *ARGS: build-cuda
    uv run python profile.py --bench 11_scaling_imprint --impl plastix \
        --build-dir {{build_cuda}} {{ARGS}}

# Run the bio-inspired / alt-framework comparison baselines (TensorNEAT, SNN,
# Nengo, evosax) into _results/baselines_table.csv. Frameworks whose optional
# extra isn't installed report status=skipped. Install extras with
# `uv sync --extra <name>` (see baselines/README.md). e.g. `just baselines -- --only snn`.
baselines *ARGS:
    uv run python run_baselines.py {{ARGS}}

# --- run --------------------------------------------------------------------
#
# The orchestrator writes a single _results/runs.csv and plots into _plots/.
# To keep the CPU and CUDA result sets side by side, each pass runs, renders
# its plots, then archives runs.csv -> _results/runs_<cfg>.csv and
# _plots/ -> _plots_<cfg>/. Pass extra orchestrator args after `--`, e.g.
# `just run-cpu -- --quick`.

# ALL frameworks on CPU. cpp / plastix(host) / pytorch / jax / snn / norse, each
# on CPU (device=cpu). Archives to _results/runs_cpu.csv, _plots_cpu/.
run-cpu *ARGS: build-cpu
    @just _run_and_plot {{build_cpu}} cpu cpu "cpp,plastix,pytorch,jax,snn,norse" "{{ARGS}}"

# ALL frameworks on GPU. The dedicated cuda impl + plastix(CUDA build) +
# pytorch/jax/snn/norse (device=cuda); cpp is CPU-only OpenBLAS so it's omitted
# (the `cuda` impl is the GPU C++). Captures per-process VRAM. Archives to
# _results/runs_gpu.csv, _plots_gpu/. This is the source of the `cuda_*` (and any
# GPU) columns in the tables. Needs the CUDA build + a GPU.
run-gpu *ARGS: build-cuda
    @just _run_and_plot {{build_cuda}} gpu cuda "cuda,plastix,pytorch,jax,snn,norse" "{{ARGS}}"

# Legacy: GPU-*built* plastix vs CPU baselines (device=cpu, tag=cuda). Kept for
# back-compat; prefer run-cpu / run-gpu.
run-cuda *ARGS: build-cuda
    @just _run_and_plot {{build_cuda}} cuda cpu "cpp,plastix,pytorch" "{{ARGS}}"

# Render plots from the current _results/runs.csv into _plots/.
plot *KINDS:
    uv run python orchestrator.py plot {{ if KINDS != "" { "--kinds " + KINDS } else { "" } }}

# Shared run+plot+archive helper (private).
_run_and_plot build_dir cfg device run_impls ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "==> [{{cfg}}] running suite (impls={{run_impls}}, device={{device}}, build={{build_dir}})"
    uv run python orchestrator.py run \
        --build-dir {{build_dir}} --impl {{run_impls}} --device {{device}} --tag {{cfg}} {{ARGS}}
    uv run python orchestrator.py plot
    cp _results/runs.csv _results/runs_{{cfg}}.csv
    rm -rf _plots_{{cfg}}
    cp -r _plots _plots_{{cfg}}
    # Refresh the paper summary tables from the freshly-archived runs_{{cfg}}.csv.
    just tables
    # A GPU pass also refreshes the all-framework GPU view (vram + GPU timings).
    if [ "{{cfg}}" = "gpu" ]; then just tables-gpu; fi
    echo "==> [{{cfg}}] graphs in _plots_{{cfg}}/  (aggregate: _results/runs_{{cfg}}.csv)"

# --- pipelines --------------------------------------------------------------

# Full CPU pipeline: build, run, plot, archive.
bench-cpu: build-cpu run-cpu

# Full CUDA pipeline: build, run, plot, archive.
bench-cuda: build-cuda run-cuda

# Everything: build both, run both, render both graph sets.
bench-all: build-all run-cpu run-cuda
    @echo "==> done. CPU graphs in _plots_cpu/, CUDA graphs in _plots_cuda/"

# Quick smoke pass over both configs (each sentinel takes its --quick path).
smoke:
    @just run-cpu -- --quick
    @just run-cuda -- --quick

# --- cleanup ----------------------------------------------------------------

# Remove generated results and plots (keeps build dirs + datasets).
clean-results:
    rm -rf _results _plots _plots_cpu _plots_cuda

# Remove the build trees (forces a fresh configure next build).
clean-build:
    rm -rf {{build_cpu}} {{build_cuda}}

# Remove everything generated (build trees, results, plots).
clean: clean-results clean-build
