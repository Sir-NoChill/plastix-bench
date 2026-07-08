"""Train CCWC models — wired NCP (A), dense LTC (B), LSTM baseline (C).

Mirrors the structure of the other  benchmarks: per-epoch
StructuralLog with the test metric as an extra, a summary CSV, and a
test-curve PNG. Headline test of the spec is the robustness sweep run by
the separate `robustness.py` script — this file trains the models and
saves checkpoints it can consume.

Usage:
    # Smoke test (all three models, sine task)
    uv run python ccwc/train.py --task sine --model all --quick

    # Real comparison (psmnist)
    uv run python ccwc/train.py --task psmnist --model all --epochs 20

The --model flag accepts {A, B, C, all}. When `all`, the three models are
trained sequentially and each gets its own row in the summary CSV.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common/pytorch"))
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    MemoryProbe,
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
from data import (  # noqa: E402
    add_gaussian_noise,
    load_psmnist,
    make_loaders,
    make_sine_dataset,
)
from models import build_model, count_params  # noqa: E402


# Workload base name. All artefacts land under results/ and
# plots/ with the suffix `_<model>[_<tag>]` appended.
WORKLOAD = "ccwc"


# ---------------------------------------------------------------------------
# Task setup
# ---------------------------------------------------------------------------

def _make_task(args):
    """Returns (train_loader, val_loader, test_loader, input_size, n_out,
    metric_kind). metric_kind is 'acc' for psmnist, 'mse' for sine."""
    if args.task == "sine":
        train_ds, val_ds, test_ds = make_sine_dataset(
            n_train=args.sine_train, n_val=args.sine_val, n_test=args.sine_test,
            seq_len=args.sine_seq_len, noise_std=args.sine_noise,
            seed=args.seed,
        )
        loaders = make_loaders(train_ds, val_ds, test_ds, batch=args.batch)
        return (*loaders, 2, 2, "mse")
    if args.task == "psmnist":
        train_ds, val_ds, test_ds, n_in, n_classes, _perm = load_psmnist(
            args.data_dir, permute=not args.no_permute,
            perm_seed=args.perm_seed, val_frac=args.val_frac, seed=args.seed,
        )
        loaders = make_loaders(train_ds, val_ds, test_ds, batch=args.batch)
        return (*loaders, n_in, n_classes, "acc")
    raise ValueError(f"unknown task: {args.task!r}")


# ---------------------------------------------------------------------------
# Per-task loss + metric
# ---------------------------------------------------------------------------

def _task_loss(metric_kind: str, logits: torch.Tensor,
               y: torch.Tensor) -> torch.Tensor:
    if metric_kind == "acc":
        return F.cross_entropy(logits, y)
    return F.mse_loss(logits, y)


def _task_metric(metric_kind: str, logits: torch.Tensor,
                 y: torch.Tensor) -> tuple[float, int]:
    """Return (running-sum, n) so the eval loop can keep a streaming average."""
    if metric_kind == "acc":
        correct = (logits.argmax(-1) == y).sum().item()
        return float(correct), int(y.size(0))
    se = F.mse_loss(logits, y, reduction="sum").item()
    return se, int(y.numel())


def _finalise_metric(metric_kind: str, running: float, n: int) -> float:
    return running / max(n, 1)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, opt, device, metric_kind: str,
                    grad_clip: float, timer) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0; m_sum = 0.0; m_n = 0; n_batches = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        timer.tick()
        logits = model(xb)
        timer.mark_forward()
        loss = _task_loss(metric_kind, logits, yb)
        timer.mark_loss()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        timer.mark_backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        timer.mark_update()
        timer.step_done()
        loss_sum += float(loss.item()); n_batches += 1
        v, n = _task_metric(metric_kind, logits.detach(), yb)
        m_sum += v; m_n += n
    return loss_sum / max(n_batches, 1), _finalise_metric(metric_kind, m_sum, m_n)


@torch.no_grad()
def evaluate(model, loader, device, metric_kind: str,
             noise_sigma: float = 0.0,
             noise_seed: int | None = None) -> tuple[float, float]:
    """Returns (mean_loss, metric_value).

    `noise_sigma > 0` adds Gaussian noise to the inputs before the forward
    pass — used by both the training-time clean evaluation (sigma=0) and the
    robustness sweep (sigma > 0)."""
    model.eval()
    loss_sum = 0.0; m_sum = 0.0; m_n = 0; n_batches = 0
    for i, (xb, yb) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if noise_sigma > 0:
            sd = None if noise_seed is None else noise_seed + i
            xb = add_gaussian_noise(xb, noise_sigma, seed=sd)
        logits = model(xb)
        loss_sum += float(_task_loss(metric_kind, logits, yb).item())
        n_batches += 1
        v, n = _task_metric(metric_kind, logits, yb)
        m_sum += v; m_n += n
    return loss_sum / max(n_batches, 1), _finalise_metric(metric_kind, m_sum, m_n)


# ---------------------------------------------------------------------------
# One model = one summary row
# ---------------------------------------------------------------------------

def run_one_model(args, model_name: str, loaders, input_size: int,
                  n_out: int, metric_kind: str, device: str,
                  probe: MemoryProbe, measure_weights: bool = False) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, val_loader, test_loader = loaders
    model = build_model(
        model_name, input_size=input_size, units=args.units,
        n_out=n_out, task=args.task, variant=args.variant,
        lstm_hidden=args.lstm_hidden, mixed_memory=args.mixed_memory,
    ).to(device)
    n_params = count_params(model)

    # Each model gets its own filename, so robustness.py can pick them up.
    sub_tag = f"{model_name}" + (f"_{args.tag}" if args.tag else "")
    saved_tag = args.tag
    args.tag = sub_tag
    hist_path, summary_path, plot_path = output_paths(args, WORKLOAD)
    args.tag = saved_tag

    log = StructuralLog(hist_path)
    n_units = (input_size + getattr(model, "n_hid", 0) + n_out)
    n_edges = n_params

    # Initial point.
    v_loss0, v_metric0 = evaluate(model, val_loader, device, metric_kind)
    t_loss0, t_metric0 = evaluate(model, test_loader, device, metric_kind)
    log.log(0, n_units=n_units, n_edges=n_edges, edges=None,
            val_loss=v_loss0, val_metric=v_metric0,
            test_loss=t_loss0, test_metric=t_metric0,
            train_loss=None, model=model_name)

    epochs = max(1, args.epochs // (4 if args.quick else 1))
    print(f"[info] model={model_name}  params={n_params}  "
          f"input_size={input_size}  units={args.units}  n_out={n_out}  "
          f"variant={args.variant}  epochs={epochs}  lr={args.lr}  "
          f"batch={args.batch}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # First model's construction is representative of the run's weight RSS;
    # only measure it once so the delta isn't summed across all three models.
    if measure_weights:
        probe.end_weights()
    best_metric = (-1.0 if metric_kind == "acc" else float("inf"))
    best_state = None

    timer = PhaseTimer()
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        tr_loss, tr_metric = train_one_epoch(
            model, train_loader, opt, device, metric_kind,
            grad_clip=args.grad_clip, timer=timer,
        )
        v_loss, v_metric = evaluate(model, val_loader, device, metric_kind)
        t_loss, t_metric = evaluate(model, test_loader, device, metric_kind)
        log.log(ep, n_units=n_units, n_edges=n_edges, edges=None,
                val_loss=v_loss, val_metric=v_metric,
                test_loss=t_loss, test_metric=t_metric,
                train_loss=tr_loss, train_metric=tr_metric,
                model=model_name, epoch=ep)
        improved = (v_metric > best_metric) if metric_kind == "acc" \
            else (v_metric < best_metric)
        if improved:
            best_metric = v_metric
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        flag = "  *" if improved else ""
        if metric_kind == "acc":
            print(f"[{model_name} ep {ep:>3d}] train_loss={tr_loss:.4f} "
                  f"train_acc={tr_metric:.3f}  val_acc={v_metric:.3f}  "
                  f"test_acc={t_metric:.3f}{flag}")
        else:
            print(f"[{model_name} ep {ep:>3d}] train_loss={tr_loss:.4f} "
                  f"train_mse={tr_metric:.4f}  val_mse={v_metric:.4f}  "
                  f"test_mse={t_metric:.4f}{flag}")
    wall = time.perf_counter() - t0
    log.flush()

    # Restore best-val checkpoint for the headline test number.
    if best_state is not None:
        model.load_state_dict(best_state)
    t_loss, t_metric = evaluate(model, test_loader, device, metric_kind)

    summary = {
        "workload": WORKLOAD,
        "task": args.task,
        "model": model_name,
        "variant": args.variant if model_name != "C" else "lstm",
        "input_size": input_size,
        "units": args.units if model_name != "C" else args.lstm_hidden,
        "n_out": n_out,
        "n_params": n_params,
        "epochs": epochs, "batch": args.batch, "lr": args.lr,
        "mixed_memory": int(args.mixed_memory),
        "wall_seconds": round(wall, 3),
        "metric_kind": metric_kind,
        "val_metric_best": round(best_metric if best_state is not None
                                 else v_metric0, 6),
        "test_metric": round(t_metric, 6),
        "seed": args.seed,
        **timer.summary_fields(wall),
        **probe.summary_fields(),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))

    # Save the best checkpoint so robustness.py can run without retraining.
    ckpt_path = Path(str(plot_path).replace(".plot.png", ".ckpt.pt"))
    payload = {
        "state_dict": model.state_dict(),
        "args": vars(args),
        "summary": summary,
        "model_name": model_name,
        "task": args.task,
        "input_size": input_size,
        "n_out": n_out,
        "units": args.units,
        "lstm_hidden": args.lstm_hidden,
        "variant": args.variant,
        "mixed_memory": args.mixed_memory,
    }
    torch.save(payload, ckpt_path)

    print(f"[done] model={model_name}  wall={wall:.1f}s  "
          f"params={n_params}  best_val={summary['val_metric_best']:.4f}  "
          f"test={t_metric:.4f}")
    print(f"[done] wrote {hist_path}, {summary_path}, {ckpt_path}")

    if not args.no_plot:
        plot_run(log.records, plot_path,
                 title=f"CCWC {args.task} {model_name} "
                       f"(units={args.units}, params={n_params})")
        args.tag = sub_tag
        tpath = test_plot_path(args, WORKLOAD)
        args.tag = saved_tag
        ylabel = "test accuracy" if metric_kind == "acc" else "test MSE"
        higher = (metric_kind == "acc")
        plot_test_curve(log.records, tpath,
                        title=f"CCWC {args.task} {model_name} -- "
                              f"{ylabel}",
                        metric_key="test_metric", ylabel=ylabel,
                        higher_is_better=higher)
        print(f"[done] wrote {plot_path}, {tpath}")
    return summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(args) -> list[dict]:
    device = resolve_device(args.device)

    probe = MemoryProbe()
    probe.start()

    loaders_and_meta = _make_task(args)
    loaders = loaders_and_meta[:3]
    input_size, n_out, metric_kind = loaders_and_meta[3:]
    probe.end_dataset()
    selected = ["A", "B", "C"] if args.model == "all" else [args.model]

    summaries: list[dict] = []
    for i, name in enumerate(selected):
        summaries.append(run_one_model(
            args, name, loaders, input_size, n_out, metric_kind, device,
            probe, measure_weights=(i == 0),
        ))

    # Joint summary: one row per model, plus a 'params_ratio' column for the
    # headline parameter-gap claim (n_params / n_params(A)).
    if len(summaries) > 1:
        joint = []
        a_params = next((s["n_params"] for s in summaries if s["model"] == "A"),
                        summaries[0]["n_params"])
        for s in summaries:
            row = dict(s)
            row["params_ratio_to_A"] = round(s["n_params"] / max(a_params, 1), 3)
            joint.append(row)
        suffix = f"_{args.tag}" if args.tag else ""
        joint_path = args.out_dir / f"{WORKLOAD}_all{suffix}.summary.csv"
        write_summary_csv(joint, joint_path, columns=list(joint[0].keys()))
        print(f"[done] joint summary -> {joint_path}")
    return summaries


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--task", choices=["sine", "psmnist"], default="sine")
    p.add_argument("--model", choices=["A", "B", "C", "all"], default="all")
    p.add_argument("--variant", choices=["ltc", "cfc"], default="ltc",
                   help="dynamical-neuron variant for A and B; CfC is the "
                        "closed-form fast LTC analogue (spec section 9)")
    p.add_argument("--units", type=int, default=32,
                   help="neuron count for A and B (AutoNCP requires units > n_out)")
    # LSTM hidden=96 puts model C at roughly 6x A's params on psmnist (4.3k
    # vs ~28k), inside the 5-10x band the spec asks for. Tune per task.
    p.add_argument("--lstm-hidden", type=int, default=96)
    p.add_argument("--mixed-memory", action="store_true",
                   help="augment the LTC/CfC with an extra memory cell")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.10)
    # psmnist-specific
    p.add_argument("--no-permute", action="store_true",
                   help="use vanilla sequential MNIST (no pixel permutation)")
    p.add_argument("--perm-seed", type=int, default=12345)
    # sine-specific
    p.add_argument("--sine-train", type=int, default=512)
    p.add_argument("--sine-val", type=int, default=128)
    p.add_argument("--sine-test", type=int, default=128)
    p.add_argument("--sine-seq-len", type=int, default=64)
    p.add_argument("--sine-noise", type=float, default=0.1)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
