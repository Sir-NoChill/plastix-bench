"""ccwc data utilities: sinusoid smoke test + (permuted) sequential MNIST.

Two tasks live behind the same `--task` switch in train.py:

- `sine`: noisy-sine regression. Generates one or more sinusoids with
  additive Gaussian noise; the network must predict the next clean
  sample given the previous noisy ones. Mirrors the ncps "first steps"
  example so the smoke test confirms the pipeline end-to-end. CPU-fast.

- `psmnist`: permuted sequential MNIST. Each 28x28 image is reshaped to
  a length-784 stream of single pixel values; the digit label is read
  off the final time step. A fixed permutation (seeded) is applied to
  the pixel order to stress long-range memory. Standard long-dependency
  RNN benchmark used in the LTC / NCP papers.

All loaders return tensors shaped `(B, T, F)` with `batch_first=True`,
matching the ncps LTC / CfC API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Smoke test: noisy-sine regression
# ---------------------------------------------------------------------------

def make_sine_dataset(
    n_train: int = 256,
    n_val: int = 64,
    n_test: int = 64,
    seq_len: int = 128,
    noise_std: float = 0.1,
    seed: int = 0,
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Generate noisy-sine sequences and their clean next-step targets.

    Each sample is a sequence of (sin(t), cos(t)) pairs sampled along a
    random phase + frequency; the input adds Gaussian noise, the target is
    the next clean value. Returns (train, val, test) TensorDatasets each
    holding `(N, T, 2)` inputs and `(N, T, 2)` targets.

    The split seeds are derived from `seed` so the three splits are
    deterministic and independent of one another.
    """
    return (
        _sine_split(n_train, seq_len, noise_std, seed=seed * 3 + 1),
        _sine_split(n_val,   seq_len, noise_std, seed=seed * 3 + 2),
        _sine_split(n_test,  seq_len, noise_std, seed=seed * 3 + 3),
    )


def _sine_split(n: int, seq_len: int, noise_std: float,
                seed: int) -> TensorDataset:
    rng = np.random.default_rng(seed)
    freq = rng.uniform(0.5, 2.0, size=(n, 1)).astype(np.float32)
    phase = rng.uniform(0.0, 2 * np.pi, size=(n, 1)).astype(np.float32)
    t = np.linspace(0, 2 * np.pi, seq_len + 1, dtype=np.float32)[None, :]
    angle = freq * t + phase                       # (n, T+1)
    sin = np.sin(angle).astype(np.float32)
    cos = np.cos(angle).astype(np.float32)
    clean = np.stack([sin, cos], axis=-1)          # (n, T+1, 2)
    noise = noise_std * rng.standard_normal(clean[:, :-1].shape).astype(np.float32)
    x = clean[:, :-1] + noise                      # (n, T, 2) noisy input
    y = clean[:, 1:]                               # (n, T, 2) clean target
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))


# ---------------------------------------------------------------------------
# Real task: (permuted) sequential MNIST
# ---------------------------------------------------------------------------

class PSMNISTDataset(torch.utils.data.Dataset):
    """Wrap an MNIST array as a length-784 sequence with one pixel per step.

    Holds the entire reshaped tensor in CPU memory (47 MB for the train
    split at float32, well within reach). Sequence access is then a cheap
    slice; no per-sample reshape during training.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, perm: np.ndarray | None):
        # X: (N, 28, 28) uint8. Normalise to [0,1] and apply the pixel-order
        # permutation once at construction time.
        flat = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        if perm is not None:
            flat = flat[:, perm]
        self.X = torch.from_numpy(flat).unsqueeze(-1)   # (N, 784, 1)
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_psmnist(
    data_dir: Path,
    permute: bool = True,
    perm_seed: int = 12345,
    val_frac: float = 0.10,
    seed: int = 0,
):
    """Return (train_ds, val_ds, test_ds, n_in, n_classes, perm).

    Downloads MNIST via torchvision on first call. `perm` is returned so
    it can be saved alongside the checkpoint for downstream evaluation.
    """
    from torchvision.datasets import MNIST

    data_dir.mkdir(parents=True, exist_ok=True)
    tr = MNIST(root=str(data_dir), train=True,  download=True)
    te = MNIST(root=str(data_dir), train=False, download=True)
    Xtr = tr.data.numpy(); ytr = tr.targets.numpy()
    Xte = te.data.numpy(); yte = te.targets.numpy()

    if permute:
        perm = np.random.default_rng(perm_seed).permutation(28 * 28)
    else:
        perm = None

    # Carve a stratified val slice out of train; deterministic in `seed`.
    rng = np.random.default_rng(seed)
    val_idx, train_idx = [], []
    for cls in range(10):
        cls_idx = np.flatnonzero(ytr == cls)
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(val_frac * len(cls_idx))))
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())
    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    rng.shuffle(train_idx); rng.shuffle(val_idx)

    train_ds = PSMNISTDataset(Xtr[train_idx], ytr[train_idx], perm)
    val_ds   = PSMNISTDataset(Xtr[val_idx],   ytr[val_idx],   perm)
    test_ds  = PSMNISTDataset(Xte, yte, perm)
    return train_ds, val_ds, test_ds, 1, 10, perm


def make_loaders(train_ds, val_ds, test_ds, batch: int,
                 num_workers: int = 0):
    common = dict(batch_size=batch, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


def add_gaussian_noise(x: torch.Tensor, sigma: float,
                       seed: int | None = None) -> torch.Tensor:
    """Return x + N(0, sigma^2). Deterministic when `seed` is provided.

    Used by the robustness sweep (train-clean -> eval-noisy). sigma=0
    returns x unchanged (avoids the RNG roundtrip)."""
    if sigma <= 0:
        return x
    if seed is None:
        return x + sigma * torch.randn_like(x)
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    noise = torch.randn(x.shape, generator=g, device=x.device,
                        dtype=x.dtype) * sigma
    return x + noise
