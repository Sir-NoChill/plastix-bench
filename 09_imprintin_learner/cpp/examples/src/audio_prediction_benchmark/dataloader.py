"""Dataset and streaming data loader for audio prediction benchmark."""

import json
import os

import numpy as np
import soundfile as sf

from .preprocessing import preprocess_step


class AudioPredictionDataset:
    """Dataset for the audio prediction benchmark.

    Loads audio and reward data from disk and provides indexed or
    sequential access to (observation, reward) pairs.

    Args:
        data_dir: Directory containing audio.wav, rewards.npy, and metadata.json.
        preprocess: If True, return binarized FFT observations. If False, return
            raw audio chunks.
        step_size: Samples per time step.
        fft_size: FFT window size.
        n_freq_bins: Number of frequency bins for binarization.
        n_mag_bins: Number of magnitude bins for binarization.
        max_magnitude: Maximum magnitude for clipping.
    """

    def __init__(self, data_dir, preprocess=True, step_size=640, fft_size=1024,
                 n_freq_bins=50, n_mag_bins=50, max_magnitude=50.0):
        self.data_dir = data_dir
        self.preprocess = preprocess
        self.step_size = step_size
        self.fft_size = fft_size
        self.n_freq_bins = n_freq_bins
        self.n_mag_bins = n_mag_bins
        self.max_magnitude = max_magnitude

        # Load audio
        audio_path = os.path.join(data_dir, "audio.wav")
        self.audio, self.sample_rate = sf.read(audio_path, dtype="float64")

        # Load rewards
        rewards_path = os.path.join(data_dir, "rewards.npy")
        self.rewards = np.load(rewards_path)

        # Load metadata if available
        metadata_path = os.path.join(data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None

        self.n_steps = len(self.rewards)

    def __len__(self):
        return self.n_steps

    def __getitem__(self, t):
        """Get (observation, reward) for time step t.

        Args:
            t: Time step index.

        Returns:
            Tuple of (observation, reward). Observation is either a binary
            vector (2500-dim) or raw audio chunk (fft_size samples).
        """
        if t < 0 or t >= self.n_steps:
            raise IndexError(f"Step index {t} out of range [0, {self.n_steps})")

        if self.preprocess:
            obs = preprocess_step(
                self.audio, t,
                step_size=self.step_size,
                fft_size=self.fft_size,
                n_freq_bins=self.n_freq_bins,
                n_mag_bins=self.n_mag_bins,
                max_magnitude=self.max_magnitude,
            )
        else:
            end = t * self.step_size + self.step_size
            start = end - self.fft_size
            if start < 0:
                obs = np.zeros(self.fft_size)
                valid_start = max(0, start)
                obs[self.fft_size - (end - valid_start):] = self.audio[valid_start:end]
            else:
                obs = self.audio[start:end].copy()

        reward = float(self.rewards[t])
        return obs, reward

    def __iter__(self):
        """Iterate over all time steps sequentially."""
        for t in range(self.n_steps):
            yield self[t]

    def get_audio(self):
        """Return the full raw audio array."""
        return self.audio

    def get_rewards(self):
        """Return the full reward array."""
        return self.rewards


class StreamingDataLoader:
    """Streaming data loader for sequential access to the benchmark.

    Wraps AudioPredictionDataset for online/streaming access,
    yielding one (observation, reward) pair at a time.

    Args:
        dataset: An AudioPredictionDataset instance.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._index = 0

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self.dataset):
            raise StopIteration
        obs, reward = self.dataset[self._index]
        self._index += 1
        return obs, reward

    def __len__(self):
        return len(self.dataset)

    def reset(self):
        """Reset the loader to the beginning."""
        self._index = 0

    @property
    def current_step(self):
        """Return the current step index."""
        return self._index
