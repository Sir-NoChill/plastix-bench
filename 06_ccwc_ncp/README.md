# ccwc -- Compact C. elegans-Wired Controller vs. a large dense RNN

Reproduces the headline Neural Circuit Policy claim (Lechner et al. 2020):
a tiny, sparsely-wired network of continuous-time neurons matches a much
larger fully-connected baseline and degrades more gracefully under
test-time noise. Lives alongside the other `` PyTorch
benchmarks and shares their logging/plotting infrastructure
(`common.py`).

Three models, one task, one robustness sweep:

- **A -- wired NCP**: Liquid Time-Constant (LTC) layer on a sparse
  `AutoNCP` (sensory → inter → command → motor) wiring. The wiring's motor
  count matches the task's output dim, so the core emits the answer
  directly — no head needed.
- **B -- dense LTC**: same LTC neuron but with a fully-connected
  topology, same neuron count as A. Isolates the *wiring* contribution
  from the *neuron-model* contribution.
- **C -- LSTM baseline**: single-layer `nn.LSTM`, sized so its parameter
  count lands at roughly 5--10x A's. The headline non-spiking baseline.

## Files

| File | Purpose |
|---|---|
| `data.py`       | Noisy-sine smoke task + (permuted) sequential MNIST loader + Gaussian-noise injector |
| `models.py`     | `WiredNCPModel`, `DenseLTCModel`, `LSTMBaselineModel`, `build_model`, `count_params` |
| `train.py`      | Trains A / B / C (or any subset), writes per-epoch history, summary CSV, and checkpoints |
| `robustness.py` | Loads the saved checkpoints and runs the test-time Gaussian-noise sweep |
| `README.md`     | This file |

Outputs land in the suite-wide directories, not next to this file:

```
results/ccwc_<model>[_tag].history.jsonl
results/ccwc_<model>[_tag].summary.csv
results/ccwc_<model>[_tag].ckpt.pt
results/ccwc_all[_tag].summary.csv              # joint table
results/ccwc[_tag].robustness.{json,csv}        # from robustness.py
plots/ccwc_<model>[_tag].test.png
plots/ccwc[_tag].robustness_vs_sigma.png        # from robustness.py
```

## Quickstart

```bash
# 0. One-time install. ncps brings AutoNCP + LTC/CfC; torchvision is for MNIST.
uv pip install ncps torchvision

# 1. Smoke test (~10 s on CPU). Trains all three on the noisy-sine task.
uv run python ccwc/train.py --task sine --model all --quick --tag smoke

# 2. Real comparison: permuted sequential MNIST. GPU recommended for LTC.
uv run python ccwc/train.py --task psmnist --model all --epochs 30 --tag pmnist0

# 3. Robustness sweep over the saved checkpoints.
uv run python ccwc/robustness.py --tag pmnist0
```

If LTC is slow (it integrates an ODE per step — spec section 9), swap to
the closed-form CfC for A and B and keep the LSTM as is:

```bash
uv run python ccwc/train.py --task psmnist --model all \
    --variant cfc --epochs 30 --tag pmnist0_cfc
uv run python ccwc/robustness.py --tag pmnist0_cfc
```

## CLI flags (`train.py`)

Inherited from `common.add_common_args`:
`--seed`, `--device {auto,cpu,cuda}`, `--data-dir`, `--out-dir`,
`--no-plot`, `--quick`, `--tag`.

ccwc-specific:

| Flag | Default | Notes |
|---|---|---|
| `--task` | `sine` | One of `sine`, `psmnist` |
| `--model` | `all` | `A`, `B`, `C`, or `all` |
| `--variant` | `ltc` | `ltc` (slow, faithful) or `cfc` (fast closed-form) |
| `--units` | 32 | Neuron count for A and B (must be > `n_out` for `AutoNCP`) |
| `--lstm-hidden` | 96 | Sized so C's param count is ~5-10x A's on these tasks |
| `--mixed-memory` | off | Extra memory cell augmenting LTC/CfC (spec section 9) |
| `--epochs` | 20 | `--quick` divides by 4 |
| `--batch` | 64 | |
| `--lr` | 1e-3 | Adam |
| `--grad-clip` | 1.0 | LTC ODE step can be stiff; clipping helps |
| `--no-permute` | off | Use vanilla sMNIST instead of pixel-permuted psMNIST |
| `--perm-seed` | 12345 | Pixel permutation seed (held fixed across A/B/C/runs) |
| `--sine-{train,val,test}` | 512/128/128 | Sine sample counts |
| `--sine-seq-len` | 64 | Sine sequence length |
| `--sine-noise` | 0.1 | Train-time noise std for the sine task |

## CLI flags (`robustness.py`)

| Flag | Default | Notes |
|---|---|---|
| `--out-dir`   | `results` | Where train.py wrote the `.ckpt.pt` files |
| `--plots-dir` | `plots`   | Output PNG location |
| `--tag`       | `""`                  | Must match the `--tag` used at train time |
| `--device`    | `auto`                | |
| `--sigmas`    | `0.0,0.05,0.1,0.2,0.4,0.8` | Test-time noise std sweep |
| `--noise-seed`| `2026`                | Deterministic noise per sigma |
| `--models`    | `A,B,C`               | Subset to sweep |

## Summary CSV schema

`results/ccwc_<model>[_tag].summary.csv` columns:

```
workload, task, model, variant, input_size, units, n_out, n_params,
epochs, batch, lr, mixed_memory, wall_seconds, metric_kind,
val_metric_best, test_metric, seed
```

`metric_kind` is `acc` for psmnist (higher is better) and `mse` for sine
(lower is better). `units` carries the neuron count for A/B and the LSTM
hidden size for C. The joint `ccwc_all[_tag].summary.csv` adds
`params_ratio_to_A` so the headline parameter-gap can be read off
directly.

## Results table

Fill in after running. The robustness column is the test metric at
`σ = 0.2` from `ccwc[_tag].robustness.csv`.

| Model | Neurons | Params | Clean test (acc / MSE) | Test @ σ=0.2 | s/epoch |
|---|---|---|---|---|---|
| A — wired NCP   | 32 | — | — | — | — |
| B — dense LTC   | 32 | — | — | — | — |
| C — LSTM        | 96 (hidden) | — | — | — | — |

2-3 sentence interpretation: does A approach C on clean accuracy with far
fewer parameters? Does A degrade more gracefully than C under noise? How
much of the robustness comes specifically from the *wiring* (A vs. B) vs.
the LTC neuron itself (B vs. C)?

## Acceptance criteria (from the spec, section 8)

1. All three models train (loss decreases) on the smoke `sine` task.
2. A filled-in results table with parameter counts (`ccwc_all*.summary.csv`).
3. A robustness-vs-σ plot comparing all three
   (`plots/ccwc[_tag].robustness_vs_sigma.png`).
4. README discusses A-vs-B-vs-C: wiring contribution vs. neuron
   contribution vs. baseline.

## Pitfalls

- **`AutoNCP(units, outputs)` requires `units > outputs`** — for psMNIST
  (10 classes) `--units 32` works; don't drop below 16.
- **Shape convention**: `(B, T, F)` with `batch_first=True`. psMNIST is
  reshaped to `(B, 784, 1)` internally; sine is `(B, T, 2)`.
- **LTC is slow** — full psMNIST runs are dominated by the per-step ODE
  integration. Swap to `--variant cfc` if epochs drag (spec section 9).
- **LSTM hidden size** drives C's param count quadratically. Default 96
  puts C at ~6x A on psMNIST; raise to 128 for ~16x, lower to 80 for ~4x.
- **Robustness only after training** — `robustness.py` will refuse to run
  if any checkpoint for `--models` is missing.
- **Tag consistency** — `robustness.py --tag X` looks for
  `results/ccwc_{A,B,C}_X.ckpt.pt`; mismatched tags = missing
  checkpoint error.

## References

- Lechner, Hasani, Amini, Henzinger, Rus & Grosu (2020), *Neural circuit
  policies enabling auditable autonomy*, Nature Machine Intelligence
  2(10):642-652. DOI 10.1038/s42256-020-00237-3.
- Hasani, Lechner, Amini, Rus & Grosu (2021), *Liquid Time-constant
  Networks*, AAAI. arXiv:2006.04439.
- `ncps` library (LTC/CfC + AutoNCP, PyTorch & Keras):
  https://github.com/mlech26l/ncps.
