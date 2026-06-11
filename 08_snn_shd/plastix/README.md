# 06 sparse-snn-shd — sparse SNN on SHD, e-prop learning

Plastix re-implementation of the surrogate-gradient SNN from
[`snn-shd/`](../../snn-shd/), restructured to
play to the framework's strengths instead of fighting them.

## What's different from the Python version

The PyTorch version trains a dense feed-forward LIF network with
**BPTT** through `T` timesteps. That's the worst possible workload for
Plastix: dense layers thrash the per-connection iteration model, and
BPTT's T-deep per-unit state has no home in Plastix's scalar-per-unit
SOA.

This benchmark does three things differently:

| Concern | Python (`snn-shd`) | Plastix (`06-sparse-snn-shd`) |
|---|---|---|
| Input → hidden layer | dense (700 × 256 = 179 200 edges) | K-sparse fan-in (K=32, 256 × 32 = 8 192 edges) |
| Hidden → output | dense (5 120 edges) | dense (5 120 edges) |
| Learning | BPTT over T=50 timesteps | **e-prop** (online, no T-deep state) |
| Backward path through readout | symmetric BPTT (uses W^T) | **random feedback alignment** (fixed random B) |
| Plastix `DoStep` cadence | n/a | one per network timestep |

The K-sparse fan-in is the framework-side win: ~14× fewer connections
to iterate per forward / backward / update sweep. The e-prop choice is
what makes the spatial sweep one-shot (no T-deep state to keep in
per-unit fields).

The design rationale is in [`SPEC.md`](SPEC.md); read that first.

## Building

Added to `traditional-plastix/CMakeLists.txt`:

```bash
cmake -S . -B build -DPLASTIX_BUILD_TRADITIONAL=ON
cmake --build build -j --target 06_sparse_snn_shd
```

The benchmark expects pre-binned `.plxbin` caches that the Python side
produces (one-time, after the SHD HDF5 has been downloaded by tonic):

```bash
uv run python snn-shd/data.py --export --n-bins 50
```

That writes `data/SHD_cache/{train,test}_n50.plxbin`
(~1.1 GB + 317 MB).

## Files

| File | Role |
|---|---|
| `SPEC.md` | Design doc; the 5-phase mapping, field tags, open questions |
| `06_sparse_snn_shd.cpp` | All policies, layer builders, host time-loop, dataset loader, CLI |
| `README.md` | This file |

## Running

```bash
# Smoke (~30 s on one core): tiny network, short training cap.
./build/traditional-plastix/06_sparse_snn_shd \
    --quick --tag smoke --n-hid 64 --fan-in 16 \
    --max-train-rows 512 --max-eval-rows 256

# Default config (~hours on one core; see "performance" below):
./build/traditional-plastix/06_sparse_snn_shd --tag e30 --epochs 30

# Tighter sparsity / wider hidden — explore the design space:
./build/traditional-plastix/06_sparse_snn_shd \
    --tag k64 --fan-in 64 --n-hid 256 --epochs 40 --lr 0.05 \
    --surrogate-slope 5
```

CLI knobs (inherited from `bench::CliArgs` plus benchmark-specific):

| Flag | Default | Notes |
|---|---|---|
| `--n-bins` | 50 | Must match the `.plxbin` cache on disk |
| `--n-hid` | 256 | Hidden width |
| `--fan-in` | 32 | K-sparse input → hidden fan-in (per hidden unit) |
| `--n-classes` | 20 | |
| `--epochs` | 20 | `--quick` divides by 4 |
| `--lr` | 1e-3 | Per-conn SGD learning rate |
| `--beta` | 0.9 | LIF membrane decay (per unit, `BetaTag`) |
| `--beta-trace` | 0.9 | Eligibility-trace decay `β_e` |
| `--threshold` | 1.0 | LIF firing threshold (per unit, `ThresholdTag`) |
| `--surrogate-slope` | 25.0 | Fast-sigmoid slope. Lower values (5–10) widen the surrogate gradient and accelerate e-prop convergence on this task |
| `--weight-scale` | 1.0 | Multiplier on Xavier init for input → hidden |
| `--feedback-scale` | 1.0 | Multiplier on the fixed random feedback B (hidden ← output) |
| `--max-train-rows` | 0 (full) | Per-epoch cap (handy for smoke runs) |
| `--max-eval-rows` | 0 (full) | Val / test cap |
| `--eval-every` | 1 | Epochs between full evaluations |

## Outputs

Match the other Plastix benchmarks:

```
traditional-plastix/results/sparse_snn_shd[_tag].history.jsonl
traditional-plastix/results/sparse_snn_shd[_tag].summary.csv
traditional-plastix/csv/sparse_snn_shd[_tag].test.csv
```

Run `uv run python traditional-plastix/plot_csv.py` to render
`traditional-plastix/plots/sparse_snn_shd[_tag].test.png` alongside the
other five.

Summary-CSV columns: `workload, dataset, n_in, n_hid, n_out, fan_in,
n_bins, n_conns, epochs, lr, beta, beta_trace, threshold,
surrogate_slope, wall_seconds, val_acc_best, test_acc,
test_acc_shuffled, ablation_drop, seed`.

## Performance reality check

For SHD with `--n-bins 50 --n-hid 256 --fan-in 32`: **~21 000 live
connections** (vs 184 000 dense). Per `DoStep`, the framework iterates
those connections four times (Forward Map, Backward Map, UpdateConn
incoming, UpdateConn outgoing). On a single core that lands around 100 k
conn-ops per `DoStep`, ~6 ms per timestep ⇒ ~300 ms per training
example over `T=50` timesteps. Over 8 156 train examples that's ~40
minutes per epoch.

The Python BPTT variant on a single mid-range GPU does the same task in
~3 s per epoch on the dense FFN; on a CPU with BLAS it's still well
under a minute. **Plastix is not competitive in wall-clock on this
workload even at K-sparse fan-in** — the per-conn iteration bottleneck
dominates regardless of how few connections you have.

What this benchmark *does* establish is that the 5-phase pipeline maps
cleanly onto SNN training when learning is online, that the framework
can express LIF + surrogate gradient + e-prop + random feedback without
hacks, and that the per-step cost scales linearly with the chosen
connection density (vary `--fan-in` to confirm).

## Open follow-ups

- **Active-edge gating.** The forward sweep iterates every connection
  regardless of whether the source spiked. Hooking
  `PruneConn`/`AddConn` into a per-timestep "tombstone if source didn't
  spike" toggle would make wall-clock proportional to *actual* spike
  activity, not network density. The framework was built for exactly
  this kind of structural mutation; the SNN is its natural beneficiary.
  Not yet implemented.
- **Adaptive learning rate.** Plain SGD on raw e-prop updates converges
  slowly on SHD vs. BPTT-with-Adam. An Adam-style per-conn second
  moment (one extra `ConnField`) is a small change inside
  `EpropUpdateConn::UpdateIncomingConnection`.
- **Time-step batching.** Plastix's per-conn dispatcher could in
  principle process several timesteps' worth of (binary) spike vectors
  in one loop iteration to amortise the per-conn fixed cost. Out of
  scope here.
