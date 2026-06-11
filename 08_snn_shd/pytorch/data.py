"""SHD data utilities: binning, val split, caching, time-shuffle ablation.

Spiking Heidelberg Digits (Cramer et al. 2020): 20 spoken-digit classes,
8332-ish train / 2088 test samples, 700 input cochlear channels. We bin
each sample into `n_bins` frames using tonic's ToFrame so the downstream
network sees a fixed `(T, B, 700)` tensor.

The training set ships without a designated validation slice; this module
carves ~10% out, stratified by label, with a fixed seed for
reproducibility. Two test speakers are held out from train, so don't
expect train_acc and test_acc to converge.

The TimeShuffle transform is the centerpiece of the temporal-coding
ablation (spec section 6.2): permute the order of the `n_bins` time
steps independently per sample, preserving per-channel spike counts but
destroying timing.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


# tonic 1.6 raises overflow warnings on a few SHD samples (h5 timestamp
# cast). They don't affect frames; silence them so the log stays useful.
warnings.filterwarnings(
    "ignore",
    message=".*(overflow|invalid value).*encountered.*",
    category=RuntimeWarning,
)


def _binned_path(data_dir: Path, split: str, n_bins: int) -> Path:
    return data_dir / "SHD_cache" / f"{split}_n{n_bins}.pt"


_N_CHANNELS = 700

# Header layout for the flat binary cache the C++ benchmark consumes.
# 24 bytes little-endian: magic, n_samples, n_bins, n_channels, dtype, reserved.
# Body: n_samples * n_bins * n_channels float32, then n_samples int64 labels.
_PLXBIN_MAGIC = 0x53484430                         # 'SHD0'
_PLXBIN_DTYPE_F32 = 0


def export_plxbin(payload: dict, path: Path) -> None:
    """Dump a {'X', 'y', 'n_bins'} payload as a flat .plxbin file."""
    X = np.ascontiguousarray(payload["X"], dtype=np.float32)
    y = np.ascontiguousarray(payload["y"], dtype=np.int64)
    n_samples, n_bins, n_channels = X.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        header = np.array([_PLXBIN_MAGIC, n_samples, n_bins, n_channels,
                           _PLXBIN_DTYPE_F32, 0], dtype=np.uint32)
        f.write(header.tobytes())
        f.write(X.tobytes())
        f.write(y.tobytes())


def ensure_plxbin(data_dir: Path, split: str, n_bins: int) -> Path:
    """Materialise (and cache) the .plxbin file the C++ benchmark reads.
    Returns the path. Re-runs the binning if the cache is missing."""
    payload = _materialise(data_dir, split, n_bins)
    out = data_dir / "SHD_cache" / f"{split}_n{n_bins}.plxbin"
    if not out.exists():
        export_plxbin(payload, out)
    return out


def _ensure_shd_h5(data_dir: Path, split: str) -> Path:
    """Download + unzip the SHD HDF5 if absent. tonic does this for us
    as a side effect of its (broken) loader, but we go around tonic for
    binning so we'd otherwise need to reimplement the download."""
    name = "shd_train.h5" if split == "train" else "shd_test.h5"
    h5_path = data_dir / "SHD" / name
    if h5_path.exists():
        return h5_path
    # Triggering tonic's loader just to fetch the .h5 is the path of least
    # surprise — same URL and checksum as the public dataset.
    import tonic
    from tonic.datasets import SHD
    SHD(save_to=str(data_dir), train=(split == "train"))
    return h5_path


def _materialise(data_dir: Path, split: str, n_bins: int):
    """Bin SHD's HDF5 spike events into a (N, n_bins, 700) float32 tensor.

    tonic 1.6's SHD loader silently corrupts data: `spikes/times` is
    stored as float16 (range 0–~1.0 sec) and tonic multiplies by 1e6 to
    convert to microseconds, which overflows float16 and produces nans
    that cast to garbage frame indices. Every binned frame ends up at
    zero. We read the HDF5 directly and bin manually."""
    import h5py

    cache = _binned_path(data_dir, split, n_bins)
    if cache.exists():
        return torch.load(cache, weights_only=False)

    h5_path = _ensure_shd_h5(data_dir, split)

    with h5py.File(h5_path, "r") as f:
        labels = np.asarray(f["labels"]).astype(np.int64)
        n_samples = len(labels)
        X = np.zeros((n_samples, n_bins, _N_CHANNELS), dtype=np.float32)
        # Per-sample bin width is chosen so that the union of bins covers
        # this specific sample's full spike-time range — equivalent to
        # tonic.ToFrame's per-sample n_time_bins normalisation.
        for i in range(n_samples):
            t = np.asarray(f["spikes/times"][i], dtype=np.float32)
            u = np.asarray(f["spikes/units"][i], dtype=np.int64)
            if t.size == 0:
                continue
            t_max = float(t.max()) + 1e-9
            bins = np.minimum((t / t_max * n_bins).astype(np.int64),
                              n_bins - 1)
            # Binarize to {0,1}: spike if any event in (bin, channel).
            flat = bins * _N_CHANNELS + u
            np.put(X[i].reshape(-1), flat, 1.0)

    cache.parent.mkdir(parents=True, exist_ok=True)
    payload = {"X": X, "y": labels, "n_bins": n_bins}
    torch.save(payload, cache)
    return payload


class SHDTensorDataset(Dataset):
    """Holds the cached binned tensors in CPU memory; cheap to index."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)               # (N, T, 700) float32
        self.y = torch.from_numpy(y)               # (N,) int64

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def stratified_val_split(y: np.ndarray, val_frac: float, seed: int):
    """Indices for a stratified val split; remainder is train."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(val_frac * len(cls_idx))))
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def load_shd(data_dir: Path, n_bins: int, val_frac: float = 0.10,
             seed: int = 0):
    """Return (train_ds, val_ds, test_ds, n_in, n_classes)."""
    train_payload = _materialise(data_dir, "train", n_bins)
    test_payload = _materialise(data_dir, "test", n_bins)

    Xtr, ytr = train_payload["X"], train_payload["y"]
    Xte, yte = test_payload["X"], test_payload["y"]

    train_idx, val_idx = stratified_val_split(ytr, val_frac, seed)
    train_ds = SHDTensorDataset(Xtr[train_idx], ytr[train_idx])
    val_ds = SHDTensorDataset(Xtr[val_idx], ytr[val_idx])
    test_ds = SHDTensorDataset(Xte, yte)

    n_in = Xtr.shape[2]
    n_classes = int(max(ytr.max(), yte.max())) + 1
    return train_ds, val_ds, test_ds, n_in, n_classes


def make_loaders(train_ds, val_ds, test_ds, batch: int, num_workers: int = 0):
    """SHD frames are already in CPU memory; no need for workers by default."""
    common = dict(batch_size=batch, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


def _cli_export() -> None:
    """`python data.py --export --n-bins 50` materialises both splits as
    .plxbin files for the C++ benchmark to consume. Idempotent — re-uses
    the .pt cache if it's already present."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--export", action="store_true")
    p.add_argument("--n-bins", type=int, default=50)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args()
    if not args.export:
        p.error("nothing to do; pass --export")
    for split in ("train", "test"):
        path = ensure_plxbin(args.data_dir, split, args.n_bins)
        print(f"[plxbin] {split} -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def time_shuffle(batch_x: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    """Permute the time axis independently per sample. Input is (B, T, C);
    returns the same shape with per-sample row-permutations applied.

    Preserves per-channel spike counts (each sample's column sums are
    unchanged), so any remaining classification signal must come from
    information other than timing."""
    B, T, C = batch_x.shape
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    perm = torch.stack([torch.randperm(T, generator=g) for _ in range(B)])
    perm = perm.to(batch_x.device)
    gather_idx = perm.unsqueeze(-1).expand(B, T, C)
    return torch.gather(batch_x, dim=1, index=gather_idx)


if __name__ == "__main__":
    _cli_export()
