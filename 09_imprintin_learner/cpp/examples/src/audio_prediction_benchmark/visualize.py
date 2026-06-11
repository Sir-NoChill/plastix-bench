"""Visualization utilities for audio prediction benchmark.

Provides functions for plotting waveforms, FFT spectra, binary observations,
spectrograms, reward signals, and event timelines. Can also be run as a
CLI tool to generate all plots from a dataset directory.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from .preprocessing import preprocess_step


def plot_waveform(audio, events, sample_rate, ax=None, time_range=None):
    """Plot audio waveform with chord and reward event markers.

    Args:
        audio: 1D audio array.
        events: List of event dicts with 'chord_time', 'reward_time', etc.
        sample_rate: Audio sample rate.
        ax: Matplotlib axes (created if None).
        time_range: Optional (start, end) in seconds to zoom in.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))

    times = np.arange(len(audio)) / sample_rate
    ax.plot(times, audio, linewidth=0.3, color="steelblue")

    # Color map for rewards
    reward_colors = {1.0: "green", -1.0: "red", 0.0: "gray"}

    for event in events:
        color = reward_colors.get(event["reward"], "blue")
        ax.axvline(event["chord_time"], color=color, alpha=0.5, linewidth=0.8,
                    linestyle="-")
        ax.axvline(event["reward_time"], color=color, alpha=0.3, linewidth=0.8,
                    linestyle="--")

    if time_range:
        ax.set_xlim(time_range)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform with Events (solid=chord, dashed=reward)")
    return ax


def plot_fft_spectrum(audio_chunk, sample_rate, ax=None):
    """Plot FFT magnitude spectrum for a single audio chunk.

    Args:
        audio_chunk: 1D audio array (typically 1024 samples).
        sample_rate: Audio sample rate.
        ax: Matplotlib axes (created if None).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    spectrum = np.fft.rfft(audio_chunk)
    magnitudes = np.abs(spectrum)
    freqs = np.fft.rfftfreq(len(audio_chunk), d=1.0 / sample_rate)

    ax.plot(freqs, magnitudes, linewidth=0.8, color="darkblue")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("FFT Magnitude Spectrum")
    return ax


def plot_binary_observation(obs_vector, n_freq_bins=50, n_mag_bins=50, ax=None):
    """Plot 2D heatmap of a binarized observation grid.

    Args:
        obs_vector: 1D binary vector of shape (n_freq_bins * n_mag_bins,).
        n_freq_bins: Number of frequency bins (columns).
        n_mag_bins: Number of magnitude bins (rows).
        ax: Matplotlib axes (created if None).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    grid = obs_vector.reshape(n_mag_bins, n_freq_bins)
    ax.imshow(grid, aspect="auto", origin="lower", cmap="binary",
              interpolation="nearest")
    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("Magnitude Bin")
    ax.set_title(f"Binary Observation ({int(obs_vector.sum())} ones)")
    return ax


def plot_spectrogram(audio, sample_rate, step_size=640, fft_size=1024,
                     ax=None, time_range=None):
    """Plot spectrogram over time using the benchmark's FFT parameters.

    Args:
        audio: 1D audio array.
        sample_rate: Audio sample rate.
        step_size: Samples per time step.
        fft_size: FFT window size.
        ax: Matplotlib axes (created if None).
        time_range: Optional (start_step, end_step) to limit range.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    n_steps = len(audio) // step_size

    if time_range:
        start_step, end_step = time_range
        start_step = max(0, start_step)
        end_step = min(n_steps, end_step)
    else:
        start_step, end_step = 0, min(n_steps, 5000)  # Limit for performance

    n_display = end_step - start_step
    n_freqs = fft_size // 2
    spectrogram = np.zeros((n_freqs, n_display))

    for i, t in enumerate(range(start_step, end_step)):
        end = t * step_size + step_size
        start = end - fft_size
        if start < 0:
            window = np.zeros(fft_size)
            valid_start = max(0, start)
            window[fft_size - (end - valid_start):] = audio[valid_start:end]
        else:
            window = audio[start:end]

        spectrum = np.fft.rfft(window)
        spectrogram[:, i] = np.abs(spectrum)[:n_freqs]

    time_axis = np.arange(start_step, end_step) * step_size / sample_rate
    freq_axis = np.arange(n_freqs) * sample_rate / fft_size

    ax.pcolormesh(time_axis, freq_axis, np.log1p(spectrogram), shading="auto",
                  cmap="viridis")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram (log magnitude)")
    return ax


def plot_reward_signal(rewards, step_size=640, sample_rate=16384, ax=None):
    """Plot reward values over time steps.

    Args:
        rewards: 1D array of per-step rewards.
        step_size: Samples per time step (for time axis).
        sample_rate: Audio sample rate (for time axis).
        ax: Matplotlib axes (created if None).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3))

    times = np.arange(len(rewards)) * step_size / sample_rate
    nonzero = rewards != 0
    ax.stem(times[nonzero], rewards[nonzero], linefmt="C0-", markerfmt="C0o",
            basefmt="k-")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reward")
    ax.set_title(f"Reward Signal ({np.count_nonzero(rewards)} non-zero)")
    ax.axhline(0, color="gray", linewidth=0.5)
    return ax


def plot_event_timeline(metadata, ax=None):
    """Plot timeline showing chord types and reward delivery.

    Args:
        metadata: Metadata dict with 'events' list.
        ax: Matplotlib axes (created if None).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3))

    events = metadata["events"]
    reward_colors = {1.0: "green", -1.0: "red", 0.0: "gray"}

    for i, event in enumerate(events):
        color = reward_colors.get(event["reward"], "blue")
        label = f"{event['instrument']}:{event['chord_name']}"

        ax.plot(event["chord_time"], 0, "o", color=color, markersize=6)
        ax.plot(event["reward_time"], 0, "x", color=color, markersize=6)
        ax.plot([event["chord_time"], event["reward_time"]], [0, 0],
                "-", color=color, alpha=0.3, linewidth=2)

        if i < 20:  # Label first 20 events
            ax.annotate(label, (event["chord_time"], 0),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=6, rotation=45, ha="left")

    ax.set_xlabel("Time (s)")
    ax.set_title("Event Timeline (o=chord, x=reward)")
    ax.set_yticks([])
    return ax


def _get_audio_chunk(audio, step_index, step_size, fft_size=1024):
    """Extract an audio chunk for a given step index."""
    end = step_index * step_size + step_size
    start = end - fft_size
    if start < 0:
        chunk = np.zeros(fft_size)
        valid_start = max(0, start)
        chunk[fft_size - (end - valid_start):] = audio[valid_start:end]
    else:
        chunk = audio[start:end]
    return chunk


def _gather_samples_per_type(events, n_samples, sample_rate, step_size):
    """Group events by instrument:chord and pick n_samples step indices per type."""
    from collections import defaultdict
    by_type = defaultdict(list)
    for event in events:
        key = f"{event['instrument']}:{event['chord_name']}"
        step = int((event["chord_time"] + 0.1) * sample_rate / step_size)
        by_type[key].append((step, event))

    result = {}
    for key, items in by_type.items():
        # Spread samples evenly across available events
        indices = np.linspace(0, len(items) - 1, min(n_samples, len(items)),
                              dtype=int)
        result[key] = [items[i] for i in indices]
    return result


def plot_fft_grid(audio, events, sample_rate, step_size=640, fft_size=1024,
                  n_samples=3):
    """Plot grid of FFT spectra: rows=chord types, cols=samples."""
    samples = _gather_samples_per_type(events, n_samples, sample_rate, step_size)
    type_names = sorted(samples.keys())
    n_types = len(type_names)
    n_cols = max(len(v) for v in samples.values())

    fig, axes = plt.subplots(n_types, n_cols, figsize=(5 * n_cols, 3.5 * n_types),
                             squeeze=False)

    for row, name in enumerate(type_names):
        for col in range(n_cols):
            ax = axes[row][col]
            if col < len(samples[name]):
                step, event = samples[name][col]
                chunk = _get_audio_chunk(audio, step, step_size, fft_size)
                plot_fft_spectrum(chunk, sample_rate, ax=ax)
                ax.set_title(f"{name} (t={event['chord_time']:.1f}s)", fontsize=10)
            else:
                ax.set_visible(False)
            if col > 0:
                ax.set_ylabel("")
            if row < n_types - 1:
                ax.set_xlabel("")

    fig.suptitle("FFT Spectra by Chord Type", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_binary_grid(audio, events, sample_rate, step_size=640, fft_size=1024,
                     n_freq_bins=50, n_mag_bins=50, max_magnitude=50.0,
                     n_samples=3):
    """Plot grid of binary observations: rows=chord types, cols=samples."""
    samples = _gather_samples_per_type(events, n_samples, sample_rate, step_size)
    type_names = sorted(samples.keys())
    n_types = len(type_names)
    n_cols = max(len(v) for v in samples.values())

    fig, axes = plt.subplots(n_types, n_cols, figsize=(4 * n_cols, 3.5 * n_types),
                             squeeze=False)

    for row, name in enumerate(type_names):
        for col in range(n_cols):
            ax = axes[row][col]
            if col < len(samples[name]):
                step, event = samples[name][col]
                obs = preprocess_step(
                    audio, step, step_size=step_size, fft_size=fft_size,
                    n_freq_bins=n_freq_bins, n_mag_bins=n_mag_bins,
                    max_magnitude=max_magnitude,
                )
                plot_binary_observation(obs, n_freq_bins, n_mag_bins, ax=ax)
                ax.set_title(f"{name} (t={event['chord_time']:.1f}s)", fontsize=10)
            else:
                ax.set_visible(False)
            if col > 0:
                ax.set_ylabel("")
            if row < n_types - 1:
                ax.set_xlabel("")

    fig.suptitle("Binary Observations by Chord Type", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize audio prediction benchmark dataset"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./output",
        help="Dataset directory (default: ./output)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save plots (default: data-dir/plots)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show plots interactively"
    )
    parser.add_argument(
        "--step", type=int, default=None,
        help="Specific time step to visualize FFT and binary observation"
    )
    parser.add_argument(
        "--n-samples", type=int, default=3,
        help="Number of samples per chord type in grid plots (default: 3)"
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "plots")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data_dir}...")
    audio, sample_rate = sf.read(os.path.join(args.data_dir, "audio.wav"),
                                  dtype="float64")
    rewards = np.load(os.path.join(args.data_dir, "rewards.npy"))
    with open(os.path.join(args.data_dir, "metadata.json")) as f:
        metadata = json.load(f)

    events = metadata["events"]
    step_size = metadata.get("step_size", 640)

    # 1. Waveform (first 120 seconds)
    print("Plotting waveform...")
    fig, ax = plt.subplots(figsize=(14, 4))
    plot_waveform(audio, events, sample_rate, ax=ax, time_range=(0, 120))
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "waveform.png"), dpi=150)
    plt.close(fig)

    # 2. Spectrogram (first 120 seconds)
    print("Plotting spectrogram...")
    end_step = int(120 * sample_rate / step_size)
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_spectrogram(audio, sample_rate, step_size=step_size,
                     ax=ax, time_range=(0, end_step))
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "spectrogram.png"), dpi=150)
    plt.close(fig)

    # 3. FFT spectrum grid (n_samples x chord types)
    print(f"Plotting FFT spectrum grid ({args.n_samples} samples per chord type)...")
    fig = plot_fft_grid(audio, events, sample_rate, step_size=step_size,
                        n_samples=args.n_samples)
    fig.savefig(os.path.join(args.output_dir, "fft_spectrum.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 4. Binary observation grid (n_samples x chord types)
    print(f"Plotting binary observation grid ({args.n_samples} samples per chord type)...")
    fig = plot_binary_grid(audio, events, sample_rate, step_size=step_size,
                           n_samples=args.n_samples)
    fig.savefig(os.path.join(args.output_dir, "binary_observation.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 5. Reward signal
    print("Plotting reward signal...")
    fig, ax = plt.subplots(figsize=(14, 3))
    plot_reward_signal(rewards, step_size=step_size, sample_rate=sample_rate, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "reward_signal.png"), dpi=150)
    plt.close(fig)

    # 6. Event timeline
    print("Plotting event timeline...")
    fig, ax = plt.subplots(figsize=(14, 3))
    plot_event_timeline(metadata, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "event_timeline.png"), dpi=150)
    plt.close(fig)

    print(f"All plots saved to {args.output_dir}/")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
