# Which benchmarks the paper figures use, and why

The two cross-workload figures (`segmented_bar_perf`, `memory_occupancy`) draw a
fixed set of benchmark columns, listed in `figures_data.py:CANON_BENCHES`. This
file records how that set was chosen against the archived run, so the choice is
reviewable rather than folklore.

## The set

Five benchmarks, all with complete four-framework coverage, non-trivial
runtimes, and working phase timers:

| Bench | Label | Why it earns a column |
|---|---|---|
| `02_idempotent_imp` | Sparse | Iterative magnitude pruning; the canonical static-sparsity workload |
| `03_bursty_elec2` | Bursty | Punctuated grow/prune bursts |
| `04_continuous_small_appliances` | Cont-S | Continuous small structural churn |
| `05_continuous_large_mackey_glass` | Cont-L | Continuous large churn + rewiring |
| `09_imprintin_learner` | Imprint | Online feature generation/removal; the highest-churn workload |

Ordered 02 -> 09 these sweep structural churn from "prune-only" to "fully
online", which is the axis both figures are really about.

## What was excluded, and why

| Bench | Reason |
|---|---|
| `01_static_etth1` (Dense) | **Not in the archives.** The CPU/GPU passes never aggregated it -- it survives only in `runs_cuda.csv`. The per-run summaries *are* on disk, so it can be restored by re-aggregating; it was dropped rather than backfilled. It remains a useful honest control (a dense workload Plastix is not expected to win) if someone re-runs it. |
| `06_ccwc_ncp` (LTC-sine) | `bench_meta.py` flags it: the cpp/plastix/jax impls are **sine-only smoke tasks**, not the psMNIST NCP result. It happens to be Plastix's largest apparent speedup (24x CPU / 17x GPU), which makes it exactly the number that must not be headlined. |
| `07_esn_mackey_class` (ESN) | Wall times are 0.0-0.5 s and only 2/6 phase timers fire. Too small to normalize or decompose meaningfully. |
| `08_snn_shd` (SNN) | No `plastix` row in `runs_cpu.csv`. The on-disk summary shows plastix both far slower and far less accurate than PyTorch (test_acc 0.246 vs 0.773). |
| `10`, `13` | No cuda impl (10); pytorch-only (13). Scaling studies -- they belong in the dedicated scaling figure. |

## What the retained five actually show

Measured, from `runs_cpu.csv` / `runs_gpu.csv`:

| Bench | plastix vs pytorch (CPU) | plastix-CUDA vs pytorch (GPU) | peak RSS vs pytorch |
|---|---|---|---|
| Sparse | 0.03x | **2.56x** | 8.0x less |
| Bursty | 0.22x | **2.61x** | 13.3x less |
| Cont-S | 0.33x | **1.98x** | 12.1x less |
| Cont-L | 0.66x | **3.23x** | 19.5x less |
| Imprint | 0.34x | **2.30x** | 17.3x less |

Two things follow, and the figures are built to show both rather than to
flatter one of them:

1. **The memory claim holds.** Plastix's occupancy tracks live structure: it is
   ~8-20x below PyTorch and at or below the hand-written C++ baseline.
2. **The CPU throughput claim does not.** The portable `plastix` impl is *slower*
   than PyTorch on every churn benchmark. What is true is directional: its
   relative disadvantage shrinks monotonically as churn rises (0.03x -> 0.66x),
   and the **CUDA backend does win, 2.0-3.2x**, on the same workloads. Any claim
   about throughput should be made about the CUDA backend, or about the trend --
   not about `plastix`-on-CPU beating PyTorch, which the data does not support.

Note the CPU column is a like-for-like wall comparison per unit of shared work
(`bench_meta.WORK_AXIS`), so normalization does not rescue it: Plastix takes
per-sample SGD steps where the minibatched impls take one step per batch (bench
02: 100k steps vs 700 for the same 20 rounds). That is a real difference in
optimizer granularity, and it is the leading candidate explanation for the CPU
gap -- but it is a *hypothesis about the benchmark harness*, not something these
numbers establish. It is worth settling before the throughput story is written.

## CPU and GPU views

`figures_data.py` emits both views of the same five benchmarks:

- `*.csv` — **CPU view**: plastix / pytorch / cpp measured on CPU, plus the cuda
  backend from the GPU pass (the original figure's wiring).
- `*_gpu.csv` — **GPU view**: every column from the GPU pass, so the CUDA backend
  is compared against GPU-resident PyTorch and JAX. There is no cpp GPU impl, so
  the third slot carries JAX. Column *prefixes* are unchanged (`cpp_*` is reused
  for JAX) so a single `figure.tex` renders either view; the `standalone_gpu.tex`
  wrapper overrides only the tick and legend labels.
