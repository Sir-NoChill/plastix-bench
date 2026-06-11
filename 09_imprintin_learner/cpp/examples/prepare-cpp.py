"""Export the audio-prediction dataset to a packed binary file for the C++ benchmark.

The C++ side reads each step as:

    struct Step {
        std::bitset<2500> observation;
        int8_t            reward;
    };

Wire format (little-endian):

    Header (16 bytes)
        offset  0:  4 bytes   magic "APBD"  (Audio-Prediction Benchmark Data)
        offset  4:  uint32    format version (currently 1)
        offset  8:  uint64    n_steps  (number of Step records that follow)

    Record (314 bytes per step, repeated n_steps times)
        offset  0:  313 bytes packed bits, LSB-first
                    byte k, bit j  ->  observation index (8*k + j)
                    the last 4 bits of byte 312 are unused and zero
        offset  313: int8     reward in {-1, 0, +1}

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


OBS_DIM = 2500              # n_freq_bins (50) * n_mag_bins (50)
PACKED_BYTES = (OBS_DIM + 7) // 8  # 313
RECORD_BYTES = PACKED_BYTES + 1    # 314 (+1 for int8 reward)

MAGIC = b"APBD"
VERSION = 1
HEADER_STRUCT = struct.Struct("<4sIQ")  # magic, version, n_steps
assert HEADER_STRUCT.size == 16


def pack_observation(obs: np.ndarray) -> bytes:
    """Pack a length-2500 {0,1} vector into 313 LSB-first bytes."""
    if obs.shape != (OBS_DIM,):
        raise ValueError(f"Expected observation of shape ({OBS_DIM},), got {obs.shape}")
    # np.packbits with bitorder='little' packs obs[8k + j] into byte k, bit j (LSB-first).
    packed = np.packbits(obs.astype(np.uint8), bitorder="little")
    if packed.shape != (PACKED_BYTES,):
        raise RuntimeError(f"Packed shape {packed.shape}, expected ({PACKED_BYTES},)")
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

    if args.n_freq_bins * args.n_mag_bins != OBS_DIM:
        parser.error(
            f"n_freq_bins * n_mag_bins must equal {OBS_DIM} to match std::bitset<{OBS_DIM}>, "
            f"got {args.n_freq_bins * args.n_mag_bins}"
        )

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
    print(f"Dataset has {n_steps} steps; writing binary to {output_path}")

    start = time.time()
    # Write header first; we know n_steps up front so no need for a backpatch.
    with open(output_path, "wb") as f:
        f.write(HEADER_STRUCT.pack(MAGIC, VERSION, n_steps))

        # Reusable buffer per record to avoid per-step allocations.
        record = bytearray(RECORD_BYTES)
        for t in range(n_steps):
            obs, reward = dataset[t]
            record[:PACKED_BYTES] = pack_observation(obs)
            # struct.pack_into writes int8 (signed) at offset PACKED_BYTES.
            struct.pack_into("<b", record, PACKED_BYTES, encode_reward(reward))
            f.write(record)

            if args.progress_every and (t + 1) % args.progress_every == 0:
                elapsed = time.time() - start
                rate = (t + 1) / elapsed if elapsed > 0 else 0.0
                eta = (n_steps - (t + 1)) / rate if rate > 0 else float("inf")
                print(f"  step {t + 1}/{n_steps}  ({rate:.0f} steps/s, ETA {eta:.0f}s)")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    expected = HEADER_STRUCT.size + n_steps * RECORD_BYTES
    actual = os.path.getsize(output_path)
    if actual != expected:
        raise RuntimeError(f"File size mismatch: wrote {actual}, expected {expected}")
    print(f"Done. Wrote {n_steps} records ({size_mb:.1f} MiB) in {time.time() - start:.1f}s.")


if __name__ == "__main__":
    main()
