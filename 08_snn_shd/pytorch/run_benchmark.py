"""Train a surrogate-gradient SNN (or RNN baseline) on SHD.

Mirrors the structure of the other 0X_*.py benchmarks:
- Per-epoch StructuralLog records carrying test accuracy as an extra so
  the existing common.plot_test_curve helper can render the test curve.
- Summary CSV under results/ with one headline row.
- Test-set plot under plots/.

The headline experiment (spec section 6.2) is the time-shuffle ablation:
after training, evaluate the model twice on the test set — once normally,
once with each sample's time-axis permuted independently. The drop is
written into the summary CSV and the per-epoch log (initial 0 / final
shuffled_drop).

Usage:
    uv run python snn-shd/train.py --model rsnn --quick   # smoke
    uv run python snn-shd/train.py --model rsnn           # full
    uv run python snn-shd/train.py --model gru            # baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# common.py lives one dir up.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common/pytorch"))
sys.path.insert(0, str(HERE))

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
from data import load_shd, make_loaders, time_shuffle  # noqa: E402
from model import build_model, count_params  # noqa: E402


def evaluate(model, loader, device, shuffle_seed: int | None = None,
             rate_reg: float = 0.0) -> tuple[float, float, float]:
    """Returns (mean_loss, accuracy, mean_firing_rate)."""
    model.eval()
    nll_sum = 0.0
    correct = 0
    seen = 0
    rate_sum = 0.0
    rate_batches = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if shuffle_seed is not None:
                xb = time_shuffle(xb, seed=shuffle_seed + seen)
            xb_t = xb.permute(1, 0, 2)             # (T, B, C)
            logits, fr = model(xb_t)
            nll_sum += F.cross_entropy(logits, yb, reduction="sum").item()
            correct += (logits.argmax(-1) == yb).sum().item()
            seen += yb.size(0)
            rate_sum += float(fr.item())
            rate_batches += 1
    rate = rate_sum / max(rate_batches, 1)
    return nll_sum / max(seen, 1), correct / max(seen, 1), rate


def train_one_epoch(model, loader, opt, device, grad_clip: float,
                    rate_reg: float, timer) -> tuple[float, float, float]:
    model.train()
    nll_sum = 0.0
    correct = 0
    seen = 0
    rate_sum = 0.0
    rate_batches = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        xb_t = xb.permute(1, 0, 2)                 # (T, B, C)
        timer.tick()
        logits, fr = model(xb_t)
        timer.mark_forward()
        loss = F.cross_entropy(logits, yb)
        if rate_reg > 0:
            loss = loss + rate_reg * (fr ** 2)
        timer.mark_loss()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        timer.mark_backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        timer.mark_update()
        timer.step_done()
        nll_sum += F.cross_entropy(logits.detach(), yb,
                                   reduction="sum").item()
        correct += (logits.argmax(-1) == yb).sum().item()
        seen += yb.size(0)
        rate_sum += float(fr.item())
        rate_batches += 1
    return (nll_sum / max(seen, 1), correct / max(seen, 1),
            rate_sum / max(rate_batches, 1))


def run(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    n_bins = args.n_bins // (2 if args.quick else 1)
    n_bins = max(20, n_bins)
    train_ds, val_ds, test_ds, n_in, n_classes = load_shd(
        args.data_dir, n_bins=n_bins, val_frac=args.val_frac, seed=args.seed,
    )
    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, batch=args.batch,
    )

    # If --model gru, build a recurrent SNN first to source the parameter
    # budget, then build the matched GRU.
    if args.model == "gru":
        snn_for_match = build_model("rsnn", n_in, args.n_hid, n_classes,
                                    beta=args.beta,
                                    surrogate_slope=args.surrogate_slope)
        model = build_model("gru", n_in, args.n_hid, n_classes,
                            beta=args.beta,
                            surrogate_slope=args.surrogate_slope,
                            match_to=snn_for_match).to(device)
        del snn_for_match
    else:
        model = build_model(args.model, n_in, args.n_hid, n_classes,
                            beta=args.beta,
                            surrogate_slope=args.surrogate_slope).to(device)

    n_params = count_params(model)
    print(f"[info] device={device}  model={args.model}  n_in={n_in}  "
          f"n_hid={getattr(model, 'n_hid', '-')}  n_out={n_classes}  "
          f"n_bins={n_bins}  params={n_params}  "
          f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.9, 0.999))

    hist_path, summary_path, plot_path = output_paths(args, "snn_shd")
    log = StructuralLog(hist_path)

    # n_units / n_edges report total neurons and total trainable weights so
    # the existing structural plots stay populated (they're static here).
    n_units = (n_in + getattr(model, "n_hid", 0) + n_classes)
    n_edges = n_params

    # Initial point.
    val_loss0, val_acc0, val_rate0 = evaluate(model, val_loader, device)
    test_loss0, test_acc0, _ = evaluate(model, test_loader, device)
    log.log(0, n_units=n_units, n_edges=n_edges, edges=None,
            val_loss=val_loss0, val_acc=val_acc0,
            test_acc=test_acc0, test_loss=test_loss0,
            firing_rate=val_rate0, epoch=0, train_loss=None)

    epochs = args.epochs // (4 if args.quick else 1)
    epochs = max(1, epochs)
    print(f"[info] epochs={epochs}  lr={args.lr}  batch={args.batch}  "
          f"rate_reg={args.rate_reg}  grad_clip={args.grad_clip}")

    best_val = -1.0
    best_state = None
    timer = PhaseTimer()
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, tr_rate = train_one_epoch(
            model, train_loader, opt, device,
            grad_clip=args.grad_clip, rate_reg=args.rate_reg,
            timer=timer,
        )
        val_loss, val_acc, val_rate = evaluate(model, val_loader, device)
        test_loss, test_acc, _ = evaluate(model, test_loader, device)
        log.log(ep, n_units=n_units, n_edges=n_edges, edges=None,
                val_loss=val_loss, val_acc=val_acc,
                test_acc=test_acc, test_loss=test_loss,
                firing_rate=tr_rate, epoch=ep, train_loss=tr_loss,
                train_acc=tr_acc)
        improved = val_acc > best_val
        if improved:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(f"[ep {ep:>3d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} "
              f"val_acc={val_acc:.3f} test_acc={test_acc:.3f} "
              f"rate={tr_rate:.3f}{'  *' if improved else ''}")
    wall = time.perf_counter() - t0

    # Restore best-val checkpoint for final reporting.
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_rate = evaluate(model, test_loader, device)
    # Primary ablation: re-eval the same trained model with each test
    # sample's time axis permuted. Spec section 6.2.
    _, shuffled_acc, _ = evaluate(model, test_loader, device,
                                   shuffle_seed=args.seed + 9999)
    log.flush()

    summary = {
        "workload": "snn_shd",
        "dataset": "SHD",
        "model": args.model,
        "n_in": n_in, "n_hid": getattr(model, "n_hid", 0),
        "n_out": n_classes, "n_bins": n_bins,
        "n_params": n_params,
        "epochs": epochs, "batch": args.batch, "lr": args.lr,
        "beta": args.beta, "surrogate_slope": args.surrogate_slope,
        "rate_reg": args.rate_reg, "grad_clip": args.grad_clip,
        "wall_seconds": round(wall, 3),
        "val_acc_best": round(best_val, 6),
        "test_acc": round(test_acc, 6),
        "test_acc_shuffled": round(shuffled_acc, 6),
        "ablation_drop": round(test_acc - shuffled_acc, 6),
        "firing_rate_final": round(test_rate, 6),
        "seed": args.seed,
        **timer.summary_fields(wall),
    }
    write_summary_csv([summary], summary_path, columns=list(summary.keys()))

    print(f"[done] wall={wall:.1f}s  test_acc={test_acc:.3f}  "
          f"shuffled={shuffled_acc:.3f}  drop={test_acc - shuffled_acc:+.3f}")
    print(f"[done] wrote {hist_path}, {summary_path}")

    # Also persist the best checkpoint so evaluate.py can run further
    # ablations without retraining.
    ckpt_path = Path(str(plot_path).replace(".plot.png", ".ckpt.pt"))
    torch.save({"state_dict": model.state_dict(),
                "args": vars(args), "summary": summary},
               ckpt_path)
    print(f"[done] wrote {ckpt_path}")

    if not args.no_plot:
        plot_run(log.records, plot_path,
                 title=f"SNN-SHD {args.model} -- n_hid={getattr(model, 'n_hid', 0)} "
                       f"n_bins={n_bins}")
        print(f"[done] wrote {plot_path}")
        tpath = test_plot_path(args, "snn_shd")
        plot_test_curve(log.records, tpath,
                        title=f"SNN-SHD test accuracy -- {args.model} "
                              f"(shuffled={shuffled_acc:.3f})",
                        metric_key="test_acc", ylabel="test accuracy",
                        higher_is_better=True)
        print(f"[done] wrote {tpath}")

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--model", choices=["rsnn", "fsnn", "gru"], default="rsnn",
                   help="recurrent SNN, feedforward SNN, or GRU baseline")
    p.add_argument("--n-bins", type=int, default=100,
                   help="time bins per sample (tune 50-250)")
    p.add_argument("--n-hid", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.9,
                   help="LIF membrane decay (closer to 1 = slower leak)")
    p.add_argument("--surrogate-slope", type=float, default=25.0,
                   help="slope of the fast-sigmoid surrogate gradient")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--rate-reg", type=float, default=0.0,
                   help="L2 penalty on mean hidden firing rate")
    p.add_argument("--grad-clip", type=float, default=1.0)
    args = p.parse_args()

    summary = run(args)
    if summary["test_acc"] < 0.65 and not args.quick and args.model == "rsnn":
        print(f"[warn] test_acc={summary['test_acc']:.3f} < 0.65 acceptance "
              f"threshold; consider --epochs higher, --beta tuning, or "
              f"--rate-reg 1e-4", file=sys.stderr)


if __name__ == "__main__":
    main()
