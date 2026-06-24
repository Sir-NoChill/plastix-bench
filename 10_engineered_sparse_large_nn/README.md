# 10 — engineered sparse large NN

A streaming-regression benchmark on a **deep (~1000-layer), sparse, irregular**
network that **grows and shrinks** while running in **pipeline propagation**.
It exists to stress the case Plastix is built for — a large dynamic sparse graph
with skip connections and per-step structural churn — against a native C++
implementation and a PyTorch implementation.

All three impls load the **same** `topology.bin` (produced by `gen.py`), so the
initial network and the data stream are identical. Growth/shrink is parametric
and driven by a shared LCG (below) so the topology evolution matches too.

## Setup
`topology.bin` is a generated artifact (not committed). Generate it once before
running:
```
uv run python 10_engineered_sparse_large_nn/gen.py   # writes topology.bin (~1.5 MB)
```
Tune the workload via `--n-layers / --units-per-layer / --fanin / --n-steps`.

## Shared workload file `topology.bin` (little-endian)
```
magic "ESLN" (4 bytes) | u32 version=1
u32 n_in | u32 n_units | u32 n_edges | u32 n_steps | u32 output_id
layer[n_units]   : u32 (layer index of each unit)
edges[n_edges]   : (u32 src, u32 dst)  with layer[src] < layer[dst]
data[n_steps]    : (f32 x[n_in], f32 target)
```
Units `[0, n_in)` are inputs (layer 0). `output_id` is the single output unit.
Hidden units have **irregular** fan-in: sources are drawn from *any* earlier
layer (skip connections), not just the adjacent one.

## Per-step algorithm (identical in all three impls)
State: `act[n_units]` (init 0) and a per-edge eligibility trace `e[edge]` (init 0).
Constants: `LR=0.01`, `DECAY=0.9` (= γλ).

For each step `t` in `[0, n_steps)`:
1. **Input:** `act[i] = x_t[i]` for `i in [0, n_in)`.
2. **Pipeline forward (one layer-advance):** using the *current* `act`, compute
   for every non-input unit `v`:  `pre[v] = Σ_{(s,v) in edges} w[(s,v)] · act[s]`;
   then set `act[v] = tanh(pre[v])` for hidden units and for `output_id` (the
   output is bounded by tanh too — this benchmark is **not** meant to converge;
   bounding just keeps the metric finite over long horizons and has no effect on
   the per-phase compute we measure). This is a single synchronous sparse mat-vec over the
   *previous* activations — i.e. the signal advances exactly one layer per step
   (Plastix `Propagation::Pipeline` semantics). Do **not** do a full topological
   forward.
3. **Predict / loss:** `yhat = act[output_id]`; `delta = target_t - yhat`;
   accumulate squared error.
4. **Update (TD(λ)-style, uniform over edges):** for every edge `e=(s,v)`:
   `e_trace = DECAY·e_trace + act[s]`; `w[e] += LR · delta · e_trace`.

Metric: `test_mse` = mean `delta²` over the **last 10%** of steps.

## Growth / shrink (parametric, shared LCG)
After a `WARMUP=1000`-step warm-up, every `GROW_EVERY=500` steps:
- **Grow** `GROW_UNITS=4` hidden units while `n_units < MAX_UNITS=6000`. Each new
  unit gets layer `1 + (lcg() % (max_hidden_layer))`, `FANIN=4` incoming edges
  from distinct earlier units (`src` drawn by `lcg() % output_id`, retry if
  `layer[src] >= newlayer` or duplicate), plus one edge `(newunit -> output_id)`.
  New edge weights start at 0.
- **Shrink:** prune `PRUNE_EDGES=8` edges chosen by `lcg() % live_edge_count`
  (skip edges into the output to keep it connected). Removing an edge frees its
  slot; a unit with no remaining edges is considered dead.

**Shared LCG** (use this exact 64-bit generator, seed `0x9E3779B97F4A7C15`, so the
schedule is identical across impls):
```
state = state * 6364136223846793005 + 1442695040888963407   (wrapping u64)
lcg() returns (state >> 33)                                  (31-bit)
```
Maintain edges in insertion order (initial edges first, then growth edges) so
`live_edge_count` and prune indices line up across impls. Exact topology match is
desirable but the load-bearing comparison is **performance on this workload
class**, not bit-identical trajectories.

## Output (summary CSV, parsed by orchestrator.py)
Emit a one-row summary with at least: `workload`, `wall_seconds`, `test_mse`,
`metric_kind=mse`, `n_units`, `n_edges`, `seed`, plus the standard `PhaseTimer`
phase columns (`step_ns_mean`, `forward_ns_mean`/`_std`, `loss_*`, `backward_*`,
`update_*`, `structural_*`, `reset_*`, `other_ns_mean`, `step_count`). Map phases:
forward = step 2; loss = step 3; update = step 4; structural = grow/shrink; the
backward/reset phases stay 0. Also write a `*.history.jsonl` (per-epoch
window MSE + `n_units`/`n_edges`) like the other benches.

CLI (match the suite): `--tag`, `--out-dir`, `--seed`, `--max-steps`,
`--log-every`, `--quick` (cap steps to 2000); C++/plastix also take
`--build-dir` (sentinel) and `--data-dir` (the dir holding `topology.bin`, which
also lives next to the sources). `--quick` must finish well under 10 minutes.
```
