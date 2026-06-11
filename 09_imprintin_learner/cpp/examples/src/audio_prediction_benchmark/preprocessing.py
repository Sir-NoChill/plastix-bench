"""FFT + binarization preprocessing for audio prediction benchmark.

Converts raw audio into binary observation vectors by applying FFT to
overlapping windows and binarizing the magnitude spectrum onto a 2D grid.
"""

import numpy as np


def preprocess_step(audio, step_index, step_size=640, fft_size=1024,
                    n_freq_bins=50, n_mag_bins=50, max_magnitude=50.0):
    """Preprocess a single time step into a binary observation vector.

    Args:
        audio: Full audio array (1D numpy array).
        step_index: Time step index (0-based).
        step_size: Number of new samples per time step.
        fft_size: FFT window size (uses overlap with previous step).
        n_freq_bins: Number of frequency bins (columns in grid).
        n_mag_bins: Number of magnitude bins (rows in grid).
        max_magnitude: Maximum magnitude for y-axis clipping.

    Returns:
        Binary observation vector of shape (n_freq_bins * n_mag_bins,), dtype uint8,
        with exactly n_freq_bins ones.
    """
    # Extract window of fft_size samples ending at step_index * step_size + step_size
    end = step_index * step_size + step_size
    start = end - fft_size

    if start < 0:
        # Pad with zeros for early steps
        window = np.zeros(fft_size)
        valid_start = max(0, start)
        window[fft_size - (end - valid_start):] = audio[valid_start:end]
    else:
        window = audio[start:end]

    # FFT: real FFT gives fft_size//2 + 1 components; take first 512
    spectrum = np.fft.rfft(window)
    magnitudes = np.abs(spectrum)[:fft_size // 2]  # 512 components

    # Divide 512 frequencies into n_freq_bins columns
    freqs_per_bin = len(magnitudes) // n_freq_bins
    remainder = len(magnitudes) % n_freq_bins

    obs = np.zeros(n_freq_bins * n_mag_bins, dtype=np.uint8)

    freq_start = 0
    for col in range(n_freq_bins):
        # Distribute remainder frequencies across first bins
        bin_size = freqs_per_bin + (1 if col < remainder else 0)
        freq_end = freq_start + bin_size

        # Max magnitude in this frequency band
        mag = np.max(magnitudes[freq_start:freq_end]) if bin_size > 0 else 0.0

        # Clamp to [0, max_magnitude]
        mag = min(mag, max_magnitude)

        # Quantize into row index [0, n_mag_bins - 1]
        row = int(mag / max_magnitude * (n_mag_bins - 1))
        row = min(row, n_mag_bins - 1)

        # Set cell (row, col) to 1 in flattened vector
        obs[row * n_freq_bins + col] = 1

        freq_start = freq_end

    return obs


def preprocess_audio(audio, sample_rate, step_size=640, fft_size=1024,
                     n_freq_bins=50, n_mag_bins=50, max_magnitude=50.0):
    """Preprocess full audio into binary observation matrix.

    Args:
        audio: Full audio array (1D numpy array).
        sample_rate: Audio sample rate in Hz (for reference, not used in computation).
        step_size: Number of new samples per time step.
        fft_size: FFT window size.
        n_freq_bins: Number of frequency bins (columns in grid).
        n_mag_bins: Number of magnitude bins (rows in grid).
        max_magnitude: Maximum magnitude for y-axis clipping.

    Returns:
        numpy array of shape (n_steps, n_freq_bins * n_mag_bins), dtype uint8.
    """
    n_steps = len(audio) // step_size
    obs_dim = n_freq_bins * n_mag_bins
    observations = np.zeros((n_steps, obs_dim), dtype=np.uint8)

    for t in range(n_steps):
        observations[t] = preprocess_step(
            audio, t,
            step_size=step_size,
            fft_size=fft_size,
            n_freq_bins=n_freq_bins,
            n_mag_bins=n_mag_bins,
            max_magnitude=max_magnitude,
        )

    return observations
