"""Shared metadata + helpers for the paper summary tables.

`phase_table.py` and `memory_table.py` both import from here so the acronym
map, the per-bench characteristics, and the archive→framework wiring have a
single source of truth. Everything here is plain stdlib (csv/pathlib).
"""

from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "_results"

# ---------------------------------------------------------------------------
# Graphing-friendly acronyms — used as the `benchmark` column in both tables so
# chart titles stay short. EDIT FREELY: each entry is one line. Any bench id not
# listed falls back to its raw directory name.
# ---------------------------------------------------------------------------
ACRONYM = {
    "01_static_etth1":                 "Dense",    # fixed dense MLP forecaster
    "02_idempotent_imp":               "Sparse",   # iterative magnitude pruning
    "03_bursty_elec2":                 "Bursty",   # punctuated grow/prune bursts
    "04_continuous_small_appliances":  "Cont-S",   # continuous small structural churn
    "05_continuous_large_mackey_glass":"Cont-L",   # continuous large structural churn
    "06_ccwc_ncp":                     "LTC-sine", # LTC net on the noisy-sine SMOKE task —
                                                   # the cpp/plastix/jax impls are sine-only,
                                                   # so this is NOT the psMNIST NCP paper claim.
                                                   # Do not headline it as a Plastix win.
    "07_esn_mackey_class":             "ESN",      # echo state network
    "08_snn_shd":                      "SNN",      # spiking neural network (SHD)
    "09_imprintin_learner":            "Imprint",  # imprinting learner (grows/prunes)
    "10_engineered_sparse_large_nn":   "XL-NN",    # engineered large sparse net
    "11_scaling_imprint":              "Scale",    # imprinting-style scaling sweep bench
}


def acronym(bench_id: str) -> str:
    """Graph-friendly short label for a benchmark id (falls back to the id)."""
    return ACRONYM.get(bench_id, bench_id)


# ---------------------------------------------------------------------------
# Which archive + impl name each framework column is sourced from.
#   (framework label, archive filename, impl value in that archive)
# No single archive holds all four impls (cpp is CPU-only, the dedicated cuda
# impl is GPU-only), so the columns are drawn from two archives.
# ---------------------------------------------------------------------------
FRAMEWORKS = [
    ("plastix", "runs_cpu.csv", "plastix"),
    ("pytorch", "runs_cpu.csv", "pytorch"),
    ("cpp",     "runs_cpu.csv", "cpp"),
    ("cuda",    "runs_gpu.csv", "cuda"),
    # JAX (CPU) baseline — ported for benches 01-05. Cells are blank for benches
    # without a jax impl. Sourced from the CPU pass (runs_cpu.csv).
    ("jax", "runs_cpu.csv", "jax"),
    # SNN frameworks — spiking ports of benches 01-05 (snnTorch) / 01-02 (Norse).
    # Blank where no impl exists. See docs/snn_expressibility.md.
    ("snn", "runs_cpu.csv", "snn"),
    ("norse", "runs_cpu.csv", "norse"),
]

# GPU view: every framework sourced from the GPU pass (runs_gpu.csv) instead of
# the CPU pass. Used by `--gpu` on the table scripts so the vram column and GPU
# timings show for ALL frameworks, not just the dedicated cuda impl. cpp has no
# GPU impl, so it's simply absent from runs_gpu.csv (blank cells).
GPU_FRAMEWORKS = [(label, "runs_gpu.csv", impl) for label, _, impl in FRAMEWORKS]

# ---------------------------------------------------------------------------
# Per-benchmark intrinsic characteristics (NOT measured — these describe the
# algorithm, on the 0-100 scales defined in the task brief):
#   gen_neuron/gen_conn  does it create units/connections at runtime?
#                        (100 = central to the algorithm, 0 = impossible by design)
#   del_neuron/del_conn  does it delete units/connections at runtime? (same scale)
#   sparsity             0 = fully dense, 100 = fully sparse
#   apriori              0 = no initial structure is specified,
#                        100 = nothing about the structure is created/modified at runtime
# These are editorial judgements grounded in each bench's header comment — review
# before publishing.
# ---------------------------------------------------------------------------
BENCH_CHARACTERISTICS = {
    # bench id                         gen_n gen_c del_n del_c spars apri
    "01_static_etth1":                 (   0,    0,    0,    0,    0,  100),  # fixed dense MLP, topology never changes
    "02_idempotent_imp":               (   0,    0,    0,  100,   80,   55),  # iterative magnitude pruning of connections
    "03_bursty_elec2":                 (  60,   60,   20,   70,   50,   55),  # bursts add units+conns, then magnitude-prune
    "04_continuous_small_appliances":  (  70,   70,   10,   60,   40,   30),  # per-step maybe-spawn unit / maybe-kill edge
    "05_continuous_large_mackey_glass":(  80,   85,   60,   75,   60,   20),  # Pareto grow/shrink + rewires + growth bursts
    "06_ccwc_ncp":                     (   0,    0,    0,    0,   75,  100),  # sparse fixed C. elegans (NCP) wiring
    "07_esn_mackey_class":             (   0,    0,    0,    0,   70,  100),  # fixed random reservoir, only readout fit
    "08_snn_shd":                      (   0,    0,    0,    0,   70,  100),  # sparse SNN, fixed topology, e-prop
    "09_imprintin_learner":            ( 100,  100,   80,   80,   70,   10),  # generates/removes features online
    # Not part of the canonical 9-bench suite (no cuda impl); engineered
    # large sparse net that only random-prunes edges. Drop this row if it is
    # out of scope for the paper figure.
    "10_engineered_sparse_large_nn":   (   0,    0,    0,   80,   85,   60),  # prune-only edge removal on a large net
    # Scaling sweep bench (own table via scaling.py); imprinting-style growth.
    "11_scaling_imprint":              ( 100,  100,   20,   40,   80,   15),  # imprinting-style net at controlled scale
}
CHARACTERISTIC_COLS = ("gen_neuron", "gen_conn", "del_neuron", "del_conn",
                       "sparsity", "apriori")


def f(row: dict, key: str) -> float:
    """Read a float cell, treating missing / blank / NaN as 0.0."""
    v = row.get(key, "")
    if v in (None, ""):
        return 0.0
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if x != x else x  # NaN -> 0


def load_archive(path: Path) -> dict[tuple[str, str], dict]:
    """Index a runs_*.csv by (bench, impl)."""
    if not path.exists():
        return {}
    with path.open() as fh:
        return {(r["bench"], r["impl"]): r for r in csv.DictReader(fh)}


def collect_archives(cpu_csv: Path, gpu_csv: Path) -> dict[str, dict]:
    """Load every archive referenced by FRAMEWORKS, honouring explicit cpu/gpu
    overrides. Returns {filename: {(bench,impl): row}}."""
    archives = {name: load_archive(RESULTS / name)
                for name in {fn for _, fn, _ in FRAMEWORKS}}
    archives["runs_cpu.csv"] = load_archive(cpu_csv)
    archives["runs_gpu.csv"] = load_archive(gpu_csv)
    return archives
