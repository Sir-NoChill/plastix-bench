"""Export the audio-prediction dataset to a packed binary file for the C++ benchmark.

The C++ side reads each step as a packed bit-vector of obs_dim bits plus a
reward, where obs_dim = n_freq_bins * n_mag_bins is configurable (the "boxes").

Wire format (little-endian), format version 2:

    Header (24 bytes)
        offset  0:  4 bytes   magic "APBD"  (Audio-Prediction Benchmark Data)
        offset  4:  uint32    format version (2; readers also accept v1)
        offset  8:  uint64    n_steps  (number of Step records that follow)
        offset 16:  uint64    obs_dim  (= n_freq_bins * n_mag_bins)

    Record (packed_bytes + 1 per step, repeated n_steps times)
        offset  0:  packed_bytes = ceil(obs_dim/8) bytes packed bits, LSB-first
                    byte k, bit j  ->  observation index (8*k + j)
                    trailing bits in the final byte are unused and zero
        offset packed_bytes: int8  reward in {-1, 0, +1}

    (Version 1 is the legacy format: a 16-byte header with no obs_dim field;
     obs_dim is implicitly 2500. Readers still load v1 files.)

The LSB-first packing is chosen so that ``bs.test(i)`` on the C++ side equals
the i-th element of the original numpy observation. See the docstring of
``load_step`` (suggested in the README discussion) for the reverse mapping.

Typical usage:

    python prepare-cpp.py --data-dir ./output --output ./output/dataset.bin
    python prepare-cpp.py --generate --duration 60 --seed 42
"""

import argparse
import os
import struct
import sys
import time

import numpy as np

from audio_prediction_benchmark import AudioPredictionDataset
from audio_prediction_benchmark import generate as apb_generate


MAGIC = b"APBD"
VERSION = 2
# v2 header: magic, version, n_steps, obs_dim. obs_dim = n_freq_bins * n_mag_bins,
# so the box count is whatever the spectrum decomposition produces — readers
# pick it up from the header and size their record stride accordingly.
HEADER_STRUCT = struct.Struct("<4sIQQ")  # magic, version, n_steps, obs_dim
assert HEADER_STRUCT.size == 24


def pack_observation(obs: np.ndarray, obs_dim: int) -> bytes:
    """Pack a length-obs_dim {0,1} vector into ceil(obs_dim/8) LSB-first bytes."""
    if obs.shape != (obs_dim,):
        raise ValueError(f"Expected observation of shape ({obs_dim},), got {obs.shape}")
    # np.packbits with bitorder='little' packs obs[8k + j] into byte k, bit j (LSB-first).
    packed = np.packbits(obs.astype(np.uint8), bitorder="little")
    expected = (obs_dim + 7) // 8
    if packed.shape != (expected,):
        raise RuntimeError(f"Packed shape {packed.shape}, expected ({expected},)")
    return packed.tobytes()


def encode_reward(reward: float) -> int:
    """Cast a float reward to int8; refuse anything we can't represent."""
    rounded = int(round(reward))
    if abs(rounded - reward) > 1e-6:
        raise ValueError(f"Reward {reward!r} is not (approximately) an integer")
    if not -128 <= rounded <= 127:
        raise ValueError(f"Reward {rounded} does not fit in int8")
    return rounded


def run_generation(passthrough_args):
    """Invoke apb_generate.main with the given argv, leaving sys.argv intact."""
    saved = sys.argv
    sys.argv = ["apb-generate"] + passthrough_args
    try:
        apb_generate.main()
    finally:
        sys.argv = saved


def main():
    parser = argparse.ArgumentParser(
        description="Convert the audio-prediction dataset into a packed binary file for the C++ benchmark."
    )
    parser.add_argument(
        "--data-dir", default="./output",
        help="Directory containing audio.wav, rewards.npy, metadata.json (default: ./output)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Destination binary path (default: <data-dir>/dataset.bin)",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Run apb-generate first; remaining args are forwarded.",
    )
    parser.add_argument(
        "--step-size", type=int, default=640,
        help="Samples per time step (must match how the dataset was generated)",
    )
    parser.add_argument(
        "--fft-size", type=int, default=1024,
        help="FFT window size",
    )
    parser.add_argument(
        "--n-freq-bins", type=int, default=50,
        help="Frequency bins (columns)",
    )
    parser.add_argument(
        "--n-mag-bins", type=int, default=50,
        help="Magnitude bins (rows)",
    )
    parser.add_argument(
        "--max-magnitude", type=float, default=50.0,
        help="Max magnitude for clipping",
    )
    parser.add_argument(
        "--progress-every", type=int, default=1000,
        help="Print progress every N steps (0 disables progress logging)",
    )
    args, passthrough = parser.parse_known_args()

    obs_dim = args.n_freq_bins * args.n_mag_bins
    if obs_dim <= 0:
        parser.error("n_freq_bins and n_mag_bins must both be positive")
    packed_bytes = (obs_dim + 7) // 8
    record_bytes = packed_bytes + 1

    if args.generate:
        gen_args = list(passthrough)
        if "--output-dir" not in gen_args:
            gen_args += ["--output-dir", args.data_dir]
        print(f"Running apb-generate with args: {gen_args}")
        run_generation(gen_args)
    elif passthrough:
        parser.error(f"Unrecognized arguments (use --generate to forward them): {passthrough}")

    output_path = args.output or os.path.join(args.data_dir, "dataset.bin")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Loading dataset from {args.data_dir} ...")
    dataset = AudioPredictionDataset(
        args.data_dir,
        preprocess=True,
        step_size=args.step_size,
        fft_size=args.fft_size,
        n_freq_bins=args.n_freq_bins,
        n_mag_bins=args.n_mag_bins,
        max_magnitude=args.max_magnitude,
    )
    n_steps = len(dataset)
    print(f"Dataset has {n_steps} steps (obs_dim={obs_dim} = "
          f"{args.n_freq_bins} freq x {args.n_mag_bins} mag bins); "
          f"writing binary to {output_path}")

    start = time.time()
    # Write header first; we know n_steps up front so no need for a backpatch.
    with open(output_path, "wb") as f:
        f.write(HEADER_STRUCT.pack(MAGIC, VERSION, n_steps, obs_dim))

        # Reusable buffer per record to avoid per-step allocations.
        record = bytearray(record_bytes)
        for t in range(n_steps):
            obs, reward = dataset[t]
            record[:packed_bytes] = pack_observation(obs, obs_dim)
            # struct.pack_into writes int8 (signed) at offset packed_bytes.
            struct.pack_into("<b", record, packed_bytes, encode_reward(reward))
            f.write(record)

            if args.progress_every and (t + 1) % args.progress_every == 0:
                elapsed = time.time() - start
                rate = (t + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n_steps - (t + 1)) / rate if rate > 0 else float("inf")
                print(f"  step {t + 1}/{n_steps}  ({rate:.0f} steps/s, ETA {eta:.0f}s)")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    expected = HEADER_STRUCT.size + n_steps * record_bytes
    actual = os.path.getsize(output_path)
    if actual != expected:
        raise RuntimeError(f"File size mismatch: wrote {actual}, expected {expected}")
    print(f"Done. Wrote {n_steps} records ({size_mb:.1f} MiB) in {time.time() - start:.1f}s.")


if __name__ == "__main__":
    main()
