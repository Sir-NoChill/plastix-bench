#!/usr/bin/env python3
"""Overlay il_audio_pred predictions on the audio-prediction dataset visuals.

Reads the predictions CSV (step,reward,prediction,n_features) emitted by the
il_audio_pred harness together with the dataset (metadata.json + audio.wav) and
produces a stacked, time-aligned figure:

  1. the audio input (log-magnitude spectrogram),
  2. the learner's GVF prediction V(t) with chord/reward event markers, and
  3. the feature-population trajectory.

Usage:
  python plot_il.py --predictions il_predictions.csv --data-dir ./output \
      --time-range 0 180 --output il_overlay.png
  python plot_il.py --predictions il_predictions.csv --time-range 0 3600 \
      --no-spectrogram --output il_overview.png
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REWARD_COLORS = {1.0: "green", -1.0: "red", 0.0: "gray"}


def load_predictions(path):
    steps, preds, nfeat = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            preds.append(float(row["prediction"]))
            nfeat.append(int(row["n_features"]))
    return np.asarray(steps), np.asarray(preds, dtype=float), np.asarray(nfeat)


def spectrogram_panel(ax, audio, sr, step_size, t0, t1, fft_size=1024):
    start_step = int(t0 * sr / step_size)
    end_step = int(t1 * sr / step_size)
    n_freqs = fft_size // 2
    spec = np.zeros((n_freqs, end_step - start_step))
    for i, t in enumerate(range(start_step, end_step)):
        end = t * step_size + step_size
        start = end - fft_size
        if start < 0:
            w = np.zeros(fft_size)
            w[fft_size - (end - max(0, start)):] = audio[max(0, start):end]
        else:
            w = audio[start:end]
        spec[:, i] = np.abs(np.fft.rfft(w))[:n_freqs]
    taxis = np.arange(start_step, end_step) * step_size / sr
    faxis = np.arange(n_freqs) * sr / fft_size
    ax.pcolormesh(taxis, faxis, np.log1p(spec), shading="auto", cmap="viridis")
    ax.set_ylabel("Freq (Hz)")
    ax.set_title("Audio input (log-magnitude spectrogram)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--data-dir", default="./output")
    ap.add_argument("--time-range", type=float, nargs=2, default=[0.0, 180.0],
                    metavar=("START", "END"))
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-spectrogram", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        meta = json.load(f)
    sr = meta["sample_rate"]
    step_size = meta["step_size"]
    events = meta["events"]

    steps, preds, nfeat = load_predictions(args.predictions)
    times = steps * step_size / sr

    t0, t1 = args.time_range
    mask = (times >= t0) & (times <= t1)
    if not mask.any():
        raise SystemExit("no prediction steps in the requested time range")
    # Per-event chord vlines clutter wide windows; only draw them when zoomed in.
    draw_chord_lines = (t1 - t0) <= 300.0

    use_spec = not args.no_spectrogram
    ratios = [2, 3, 1] if use_spec else [3, 1]
    fig, axes = plt.subplots(len(ratios), 1, figsize=(14, 2.8 * len(ratios)),
                             sharex=True, gridspec_kw={"height_ratios": ratios})
    axes = np.atleast_1d(axes)
    i = 0

    if use_spec:
        try:
            import soundfile as sf
            audio, asr = sf.read(os.path.join(args.data_dir, "audio.wav"),
                                 dtype="float64")
            spectrogram_panel(axes[i], audio, asr, step_size, t0, t1)
        except Exception as e:  # noqa: BLE001
            axes[i].text(0.5, 0.5, f"spectrogram unavailable: {e}",
                         ha="center", va="center", transform=axes[i].transAxes)
        i += 1

    # Prediction panel: V(t) on the left axis, actual rewards on a twin axis.
    axp = axes[i]
    i += 1
    axp.plot(times[mask], preds[mask], color="black", linewidth=1.0,
             label="IL prediction V(t)")
    axp.axhline(0, color="gray", linewidth=0.5)
    axp.set_ylabel("Prediction V")
    axr = axp.twinx()
    axr.set_ylabel("Reward")
    axr.set_ylim(-1.3, 1.3)
    seen = set()
    for ev in events:
        ct, rt, rw = ev["chord_time"], ev["reward_time"], ev["reward"]
        if rt < t0 or ct > t1:
            continue
        c = REWARD_COLORS.get(rw, "blue")
        if draw_chord_lines:
            axp.axvline(ct, color=c, alpha=0.45, linewidth=1.0)
            axp.axvline(rt, color=c, alpha=0.35, linewidth=1.0, linestyle="--")
        lbl = f"reward {int(rw):+d}" if rw not in seen else None
        seen.add(rw)
        axr.plot(rt, rw, "o", color=c, markersize=6, label=lbl)
    axp.set_title("IL GVF prediction vs reward events "
                  "(solid=chord onset, dashed=reward time, o=delivered reward)")
    h1, l1 = axp.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    axp.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    # Feature-population panel.
    axf = axes[i]
    axf.plot(times[mask], nfeat[mask], color="purple", linewidth=1.0)
    axf.set_ylabel("# features")
    axf.set_xlabel("Time (s)")
    axf.set_title("Feature population (generation/removal)")

    fig.suptitle("Imprinting learner on the Audio Prediction Benchmark",
                 y=1.002, fontsize=13)
    fig.tight_layout()

    out = args.output or os.path.join(args.data_dir, "plots", "il_overlay.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
