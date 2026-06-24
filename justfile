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
# pytorch are reference baselines. Both run on CPU in either pass.
impls := "cpp,plastix,pytorch"

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

# --- run --------------------------------------------------------------------
#
# The orchestrator writes a single _results/runs.csv and plots into _plots/.
# To keep the CPU and CUDA result sets side by side, each pass runs, renders
# its plots, then archives runs.csv -> _results/runs_<cfg>.csv and
# _plots/ -> _plots_<cfg>/. Pass extra orchestrator args after `--`, e.g.
# `just run-cpu -- --quick`.

# Run + plot the CPU configuration; archive to _results/runs_cpu.csv, _plots_cpu/.
run-cpu *ARGS: build-cpu
    @just _run_and_plot {{build_cpu}} cpu "{{ARGS}}"

# Run + plot the CUDA configuration; archive to _results/runs_cuda.csv, _plots_cuda/.
run-cuda *ARGS: build-cuda
    @just _run_and_plot {{build_cuda}} cuda "{{ARGS}}"

# Render plots from the current _results/runs.csv into _plots/.
plot *KINDS:
    uv run python orchestrator.py plot {{ if KINDS != "" { "--kinds " + KINDS } else { "" } }}

# Shared run+plot+archive helper (private).
_run_and_plot build_dir cfg ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "==> [{{cfg}}] running suite (impls={{impls}}, build={{build_dir}})"
    uv run python orchestrator.py run \
        --build-dir {{build_dir}} --impl {{impls}} --tag {{cfg}} {{ARGS}}
    uv run python orchestrator.py plot
    cp _results/runs.csv _results/runs_{{cfg}}.csv
    rm -rf _plots_{{cfg}}
    cp -r _plots _plots_{{cfg}}
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
