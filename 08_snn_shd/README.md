# snn-shd — surrogate-gradient SNN on Spiking Heidelberg Digits

Sixth benchmark in the plastix-bench suite. Trains a recurrent LIF
network with surrogate-gradient BPTT on SHD and uses a **time-shuffle
ablation** to demonstrate that the trained model relies on spike
timing, not just spike counts.

Implements the spec in [the experiment design][spec], integrated with
the same logging/plotting infrastructure (`common.py`) as
the other five benchmarks.

[spec]: ../../  "see the prompt that produced this folder"

## Files

| File | Purpose |
|---|---|
| `data.py`     | SHD loader (via `tonic`), binning, val split, cache, `time_shuffle` transform |
| `model.py`    | `RecurrentSNN`, `FeedforwardSNN`, `GRUBaseline`, param-match helper |
| `train.py`    | Main entry. Trains, evaluates, runs the ablation once, writes `.history.jsonl` / `.summary.csv` / plots / `.ckpt.pt` |
| `evaluate.py` | Loads a `.ckpt.pt`, re-runs the time-shuffle ablation across multiple seeds, writes `.ablation.json` and a bar chart |
| `README.md`   | This file |

Outputs land in the suite's standard directories, **not** alongside this
file:

```
results/snn_shd[_tag].history.jsonl
results/snn_shd[_tag].summary.csv
results/snn_shd[_tag].plot.png
results/snn_shd[_tag].ckpt.pt
results/snn_shd[_tag].ablation.json     (from evaluate.py)
plots/snn_shd[_tag].test.png
plots/snn_shd[_tag].ablation.png       (from evaluate.py)
```

## Quickstart

```bash
# 1. Install the extra deps (one-time).
uv pip install snntorch tonic scikit-learn

# 2. Smoke test (~1 min on CPU, tiny epochs/bins).
uv run python snn-shd/train.py --model rsnn --quick --tag smoke

# 3. Full recurrent-SNN run (cost dominated by BPTT over n_bins=100).
uv run python snn-shd/train.py --model rsnn --tag rsnn80

# 4. Parameter-matched GRU baseline.
uv run python snn-shd/train.py --model gru --tag gru80

# 5. Feedforward SNN ablation.
uv run python snn-shd/train.py --model fsnn --tag fsnn80

# 6. Re-run the time-shuffle ablation across multiple seeds for tighter stats.
uv run python snn-shd/evaluate.py \
    --ckpt results/snn_shd_rsnn80.ckpt.pt --n-shuffle-runs 5
```

The first invocation downloads SHD (~130 MB) into
`data/SHD/`. The binned tensors are cached under
`data/SHD_cache/{train,test}_n<n_bins>.pt` so subsequent
epochs / runs skip the slow tonic ToFrame path.

## CLI flags (`train.py`)

Inherited from `common.add_common_args`:
`--seed`, `--device {auto,cpu,cuda}`, `--data-dir`, `--out-dir`,
`--no-plot`, `--quick`, `--tag`.

SNN-specific:

| Flag | Default | Notes |
|---|---|---|
| `--model` | `rsnn` | One of `rsnn`, `fsnn`, `gru` |
| `--n-bins` | 100 | Time bins per sample; cost scales linearly |
| `--n-hid` | 256 | Hidden width (ignored for GRU; solved to match SNN params) |
| `--beta` | 0.9 | LIF membrane decay (learnable) |
| `--surrogate-slope` | 25.0 | `fast_sigmoid` slope |
| `--epochs` | 80 | `--quick` divides by 4 |
| `--batch` | 128 | |
| `--lr` | 1e-3 | Adam |
| `--val-frac` | 0.10 | Stratified split out of train (SHD ships no val) |
| `--rate-reg` | 0.0 | L2 penalty on mean hidden firing rate |
| `--grad-clip` | 1.0 | BPTT stability |

## Summary CSV schema

`results/snn_shd_<tag>.summary.csv` columns:

```
workload, dataset, model, n_in, n_hid, n_out, n_bins, n_params,
epochs, batch, lr, beta, surrogate_slope, rate_reg, grad_clip,
wall_seconds, val_acc_best, test_acc, test_acc_shuffled,
ablation_drop, firing_rate_final, seed
```

`ablation_drop = test_acc - test_acc_shuffled`. The primary hypothesis
is that this is sharply positive for `rsnn`/`fsnn` and near zero for
`gru` operating on a binarized rate input (it's a sequence model that
also reads timing, so the drop may be non-trivial — that's OK; the
comparison is the SNN's drop vs. its own natural accuracy).

## Acceptance criteria (from the spec)

1. Trained `rsnn` reaches **≥ 65%** test accuracy. The script emits a
   `[warn]` if a non-`--quick` run finishes below this threshold.
2. A quantified time-shuffle drop is reported with a bar chart
   (`evaluate.py` → `.ablation.png`).
3. The parameter-matched GRU baseline is reported in the same table
   (`results/snn_shd_gru*.summary.csv`).
4. This README plus `results/snn_shd_*.summary.csv` (and
   `pip freeze > snn-shd/requirements.txt`) constitute the
   reproduction record.

## Pitfalls / version notes

- `snntorch.RLeaky` state signature has shifted across versions. Tested
  against `snntorch==0.9.4`. If the version differs and `init_rleaky`
  returns the wrong tuple shape, swap `RLeaky` for `Leaky` plus a manual
  `nn.Linear(n_hid, n_hid)` recurrent layer applied to the previous
  hidden spikes.
- `tonic==1.6` emits an overflow warning on a handful of SHD samples
  during HDF5 timestamp cast; suppressed in `data.py`. Doesn't affect
  binned frames.
- BPTT cost grows linearly with `--n-bins`. Drop to 50 for cheap runs
  before lowering anything else.
- If the SNN sits near chance (~5%): firing rate is probably 0 (dead
  network) or 1 (saturated). Inspect `firing_rate` in the log; tune
  `--beta` and `--rate-reg`. Spec section 9.

## Pinning versions

```bash
uv pip freeze | grep -E 'snntorch|tonic|torch|numpy|scikit-learn' \
    > snn-shd/requirements.txt
```
