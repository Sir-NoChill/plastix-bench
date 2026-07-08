# baselines/ — bio-inspired / alt-framework comparison harnesses

These are **comparison baselines** in frameworks whose paradigm does not fit the
per-step Plastix transition-function interface (evolutionary population search,
neuromorphic NEF, spiking stacks). Because they don't map onto the
`forward/backward/update/prune/grow` phase decomposition, they do **not** join
the phase/memory tables — they get their own flat comparison table,
`_results/baselines_table.csv`, produced by `run_baselines.py` (`just baselines`).

## Layout

```
baselines/
  common.py                 shared row schema + emit() + skip handling
  <framework>/run_baseline.py   one runnable stub per framework
```

`run_baselines.py` (repo root) runs each `<framework>/run_baseline.py` in its own
subprocess, polls peak RSS (via `orchestrator._poll_memory`), and collects one
row per framework into `_results/baselines_table.csv`.

## Frameworks + mapped tasks

| framework   | paradigm                   | mapped task                              | optional extra          |
|-------------|----------------------------|------------------------------------------|-------------------------|
| tensorneat  | evolutionary NEAT (JAX)     | evolve a topology on a control/regression fitness | git (not on PyPI): `pip install git+https://github.com/EMI-Group/tensorneat.git` |
| snn         | spiking NN (norse/snnTorch) | 08_snn_shd (SHD classification)          | `uv sync --extra snn` (snnTorch is base) |
| nengo       | neuromorphic NEF            | Mackey-Glass 1-step as an NEF network    | `uv sync --extra nengo` |
| evosax      | evolution strategies (JAX)  | weight-ES on a fixed MLP                  | `uv sync --extra evosax` |

## Status: scaffold

Each `run_baseline.py` is a **runnable stub**:
- if its framework isn't installed → emits a row with `status=skipped` + an
  install hint (the harness still completes),
- if installed → emits `status=stub` and a `TODO` marks where the real model
  goes.

## Filling in a baseline

1. `uv sync --extra <framework>` to install it.
2. Implement the `# TODO` in `baselines/<framework>/run_baseline.py`: run the real
   model, then `emit({..., "metric": ..., "wall_seconds": ..., "n_params": ...,
   "status": "ok"}, args.out)`.
3. Wall time is measured inside the stub; peak RSS is polled by `run_baselines.py`.
4. `just baselines` (or `uv run python run_baselines.py --only <framework>`).

## Row schema (`_results/baselines_table.csv`)

`framework, task, paradigm, metric, metric_kind, wall_seconds, peak_rss_mb,
n_params, status, notes`
