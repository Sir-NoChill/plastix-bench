"""Export a *scaled* audio-prediction dataset to the packed binary the scaled
C++ harness reads.

Identical to ../prepare-cpp.py except the observation grid is ~10x larger:
158 frequency bins x 158 magnitude bins = 24964 dims (vs 50x50 = 2500). The
preprocessing logic itself is unchanged — it already takes n_freq_bins /
n_mag_bins as parameters — so we reuse the parent `audio_prediction_benchmark`
package and only change the grid defaults + the OBS_DIM the binary is packed to.

Typical usage (from this directory, with the examples venv / `uv run`):

    uv run --project .. python prepare-cpp.py --generate --duration 60 --seed 42
    uv run --project .. python prepare-cpp.py --data-dir output --output output/dataset.bin
"""

import argparse
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the parent example's package (preprocessing/generation) without copying.
_PARENT_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(_PARENT_SRC))

from audio_prediction_benchmark import AudioPredictionDataset
from audio_prediction_benchmark import generate as apb_generate


# Scaled grid: 158 * 158 = 24964 (~10x the original 2500).
N_FREQ_BINS = 158
N_MAG_BINS = 158
OBS_DIM = N_FREQ_BINS * N_MAG_BINS  # 24964
PACKED_BYTES = (OBS_DIM + 7) // 8   # 3121
RECORD_BYTES = PACKED_BYTES + 1     # 3122

MAGIC = b"APBD"
VERSION = 1
HEADER_STRUCT = struct.Struct("<4sIQ")  # magic, version, n_steps
assert HEADER_STRUCT.size == 16


def pack_observation(obs: np.ndarray) -> bytes:
    """Pack a length-OBS_DIM {0,1} vector into PACKED_BYTES LSB-first bytes."""
    if obs.shape != (OBS_DIM,):
        raise ValueError(f"Expected observation of shape ({OBS_DIM},), got {obs.shape}")
    packed = np.packbits(obs.astype(np.uint8), bitorder="little")
    if packed.shape != (PACKED_BYTES,):
        raise RuntimeError(f"Packed shape {packed.shape}, expected ({PACKED_BYTES},)")
    return packed.tobytes()


def encode_reward(reward: float) -> int:
    rounded = int(round(reward))
    if abs(rounded - reward) > 1e-6:
        raise ValueError(f"Reward {reward!r} is not (approximately) an integer")
    if not -128 <= rounded <= 127:
        raise ValueError(f"Reward {rounded} does not fit in int8")
    return rounded


def run_generation(passthrough_args):
    saved = sys.argv
    sys.argv = ["apb-generate"] + passthrough_args
    try:
        apb_generate.main()
    finally:
        sys.argv = saved


def main():
    parser = argparse.ArgumentParser(
        description="Convert the audio-prediction dataset into a packed binary "
                    f"file for the scaled C++ benchmark ({OBS_DIM}-dim)."
    )
    parser.add_argument("--data-dir", default="./output")
    parser.add_argument("--output", default=None,
                        help="Destination binary (default: <data-dir>/dataset.bin)")
    parser.add_argument("--generate", action="store_true",
                        help="Run apb-generate first; remaining args are forwarded.")
    parser.add_argument("--step-size", type=int, default=640)
    parser.add_argument("--fft-size", type=int, default=1024)
    parser.add_argument("--n-freq-bins", type=int, default=N_FREQ_BINS)
    parser.add_argument("--n-mag-bins", type=int, default=N_MAG_BINS)
    parser.add_argument("--max-magnitude", type=float, default=50.0)
    parser.add_argument("--progress-every", type=int, default=1000)
    args, passthrough = parser.parse_known_args()

    if args.n_freq_bins * args.n_mag_bins != OBS_DIM:
        parser.error(
            f"n_freq_bins * n_mag_bins must equal {OBS_DIM} to match the scaled "
            f"ObservationDim, got {args.n_freq_bins * args.n_mag_bins}"
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

    print(f"Loading dataset from {args.data_dir} (grid "
          f"{args.n_freq_bins}x{args.n_mag_bins}={OBS_DIM}) ...")
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
    with open(output_path, "wb") as f:
        f.write(HEADER_STRUCT.pack(MAGIC, VERSION, n_steps))
        record = bytearray(RECORD_BYTES)
        for t in range(n_steps):
            obs, reward = dataset[t]
            record[:PACKED_BYTES] = pack_observation(obs)
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
