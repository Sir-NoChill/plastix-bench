"""
Workload 9 -- imprinting-learner on the audio-prediction benchmark.

PyTorch implementation: streaming linear TD(λ) over the 2500-dim binary audio
observation. This mirrors what the C++ and Plastix imprinting learners
compute at the output-weight layer (without the dynamic pattern/memory
generation). The metric reported alongside the other impls is

    test_mse = mean (V_t - G_t)^2 over the final 10% of steps,

where G_t = Σ_k γ^k r_{t+k} is the offline discounted return.

Dataset: APBD packed-bits binary file (same format the C++/Plastix benches
consume). The reader is a self-contained numpy unpacker -- no audio
dependencies needed.

Usage:
    uv run python 09_imprintin_learner/pytorch/run_benchmark.py \
        --quick --tag smoke
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common/pytorch"))
from common import (  # noqa: E402
    PhaseTimer,
    StructuralLog,
    add_common_args,
    output_paths,
    plot_run,
    plot_test_curve,
    resolve_device,
    test_plot_path,
    write_summary_csv,
)


OBS_DIM = 2500
PACKED_BYTES = (OBS_DIM + 7) // 8  # 313
RECORD_BYTES = PACKED_BYTES + 1
HEADER_BYTES = 16
MAGIC = b"APBD"
FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# APBD reader (matches dataset.hpp on the C++ side)
# ---------------------------------------------------------------------------

def _resolve_dataset(data_dir: Path) -> Path | None:
    candidates = [
        data_dir / "audio_prediction" / "dataset.bin",
        data_dir / "audio" / "dataset.bin",
        data_dir / "dataset.bin",
        Path("09_imprintin_learner/cpp/examples/output/dataset.bin"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_apbd(path: Path, max_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, R) with X shape (N, 2500) uint8 in {0,1}, R shape (N,) float32."""
    with path.open("rb") as f:
        head = f.read(HEADER_BYTES)
        if head[:4] != MAGIC:
            raise RuntimeError(f"Not an APBD file (bad magic): {path}")
        version, n_steps = struct.unpack("<IQ", head[4:HEADER_BYTES])
        if version != FORMAT_VERSION:
            raise RuntimeError(f"Unsupported APBD version {version}")
        n = min(max_steps, n_steps) if max_steps > 0 else n_steps
        body = np.frombuffer(f.read(n * RECORD_BYTES), dtype=np.uint8)
    body = body.reshape(n, RECORD_BYTES)
    obs_packed = body[:, :PACKED_BYTES]
    rewards = body[:, PACKED_BYTES].view(np.int8).astype(np.float32)
    # LSB-first bit ordering, matching prepare-cpp.py / dataset.hpp.
    X = np.unpackbits(obs_packed, axis=1, bitorder="little")[:, :OBS_DIM]
    return X, rewards


# ---------------------------------------------------------------------------
# Offline discounted return (same for every impl)
# ---------------------------------------------------------------------------

def compute_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    G = np.zeros_like(rewards)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = float(rewards[t]) + gamma * acc
        G[t] = acc
    return G


# ---------------------------------------------------------------------------
# Linear TD(λ) -- the actual learner
# ---------------------------------------------------------------------------
#
# State: w (weight vector), e (eligibility trace). Per step t:
#
#   V_t   = w · x_t
#   δ_t   = r_t + γ·V_t - V_{t-1}                (Plastix/C++ convention)
#   e_t   = γ·λ·e_{t-1}; e_t[i] = 1 for active i   (replacing trace)
#   w_{t+1} = w_t + (α / nnz) · δ_t · e_t
#
# x_t is a binary 2500-d vector with ~50 set bits per step. Replacing traces
# (Sutton & Barto §12.7) keep |e| bounded which avoids the runaway-update
# instability that plain accumulating traces hit with constant α and dense
# binary input. The α/nnz scaling matches the effective step magnitude that
# a per-feature-adaptive learner (SwiftTD) settles to under the same η bound;
# without it constant-α TD(λ) on this stream diverges.

def td_lambda_run(
    X: np.ndarray, R: np.ndarray, G: np.ndarray,
    *, gamma: float, lam: float, alpha: float,
    log_every: int, log: StructuralLog, device: str,
    timer: PhaseTimer,
) -> tuple[torch.Tensor, dict]:
    N, D = X.shape
    # The orchestrator-driven runs sit on CPU; the device argument exists so
    # standalone invocation can `--device cuda`. We do all ops in float32.
    Xt = torch.from_numpy(X.astype(np.float32)).to(device)
    Gt = torch.from_numpy(G).to(device)
    Rt = torch.from_numpy(R).to(device)

    w = torch.zeros(D, device=device)
    e = torch.zeros(D, device=device)
    predictions = torch.zeros(N, device=device)
    ones = torch.ones(D, device=device)

    decay = gamma * lam

    window_sse = 0.0
    window_cnt = 0
    epoch = 0

    # We bundle the TD-style "δ + trace + w.add_" inside `backward` (the
    # autograd analogue) and leave `update` at zero — the TD update is
    # algorithmically inseparable from the gradient.
    with torch.no_grad():
        V_old = float(torch.dot(w, Xt[0]).item())
        for t in range(N):
            x_t = Xt[t]

            timer.tick()
            V_t = torch.dot(w, x_t)
            timer.mark_forward()

            v_t_f = float(V_t.item())
            r_t_f = float(Rt[t].item())
            delta = r_t_f + gamma * v_t_f - V_old
            e.mul_(decay)
            e = torch.where(x_t > 0, ones, e)
            nnz = float(x_t.sum().item())
            step_alpha = alpha / max(nnz, 1.0)
            w.add_(e, alpha=step_alpha * delta)
            timer.mark_backward()
            timer.step_done()

            predictions[t] = V_t
            V_old = v_t_f

            err = v_t_f - float(Gt[t].item())
            window_sse += err * err
            window_cnt += 1

            if window_cnt >= log_every or t + 1 == N:
                window_mse = window_sse / window_cnt
                epoch += 1
                log.log(epoch, D, D, edges=None, val_loss=window_mse,
                        train_loss=window_mse, train_step=t + 1,
                        test_mse=window_mse, n_features=D)
                print(f"[ep {epoch}] step={t+1} window_mse={window_mse:.4f}")
                window_sse = 0.0
                window_cnt = 0
    return predictions, {"V_final": float(V_old)}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    path = _resolve_dataset(args.data_dir)
    if path is None:
        raise SystemExit(
            f"[err] audio-prediction dataset.bin not found under {args.data_dir}.\n"
            f"      Generate via 09_imprintin_learner/cpp/examples/"
            f"prepare-cpp.py, or pass --data-dir <dir-with-dataset.bin>."
        )

    max_steps = args.max_steps
    log_every = args.log_every
    if args.quick:
        max_steps = min(max_steps, 1000)
        log_every = min(log_every, 200)

    X, R = _load_apbd(path, max_steps)
    N = X.shape[0]
    G = compute_returns(R, args.gamma)

    print(f"[info] device={device}  dataset={path}  steps={N}  "
          f"gamma={args.gamma}  alpha={args.alpha}  lambda={args.lam}  "
          f"quick={int(args.quick)}")

    hist_path, summary_path, plot_path = output_paths(args, "audio_imprinting")
    log = StructuralLog(hist_path)

    timer = PhaseTimer()
    t0 = time.perf_counter()
    predictions, _ = td_lambda_run(
        X, R, G,
        gamma=args.gamma, lam=args.lam, alpha=args.alpha,
        log_every=log_every, log=log, device=device, timer=timer,
    )
    wall = time.perf_counter() - t0

    Gt = torch.from_numpy(G).to(predictions.device)
    tail = max(1, N // 10)
    diff_tail = predictions[-tail:] - Gt[-tail:]
    test_mse = float((diff_tail * diff_tail).mean().item())
    full_mse = float(((predictions - Gt) ** 2).mean().item())

    log.flush()

    summary = {
        "workload": "09_imprintin_learner",
        "dataset": "audio_prediction",
        "max_steps": N,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "lambda": args.lam,
        "wall_seconds": round(wall, 3),
        "val_mse_final": round(full_mse, 6),
        "test_mse": round(test_mse, 6),
        "metric_kind": "mse",
        "n_units": OBS_DIM,
        "n_edges": OBS_DIM,
        "seed": args.seed,
        **timer.summary_fields(wall),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))

    print(f"[done] wall={wall:.2f}s  test_mse={test_mse:.6f}  "
          f"full_mse={full_mse:.6f}  (step_count={timer.step_count})")
    print(f"[done] wrote {hist_path}, {summary_path}")

    if not args.no_plot:
        plot_run(log.records, plot_path,
                 title=f"Audio-Prediction TD(λ) -- α={args.alpha} λ={args.lam}")
        print(f"[done] wrote {plot_path}")
        tpath = test_plot_path(args, "audio_imprinting")
        plot_test_curve(
            log.records, tpath,
            title=f"Audio-Prediction TD(λ) test MSE -- α={args.alpha} λ={args.lam}",
            metric_key="test_mse", ylabel="window MSE",
            higher_is_better=False,
        )
        print(f"[done] wrote {tpath}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--max-steps", type=int, default=20000,
                   help="cap the number of timesteps to run (0 = all)")
    p.add_argument("--log-every", type=int, default=2000)
    p.add_argument("--gamma", type=float, default=0.99,
                   help="discount factor (matches C++/Plastix default)")
    p.add_argument("--alpha", type=float, default=3e-3,
                   help="per-step TD step size (matches Plastix's alpha_init)")
    p.add_argument("--lambda", dest="lam", type=float, default=0.9,
                   help="eligibility-trace decay")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
