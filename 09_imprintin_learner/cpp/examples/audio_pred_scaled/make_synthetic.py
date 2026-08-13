"""Write a synthetic APBD dataset for the scaled harness — no audio/FluidSynth.

Produces a well-formed dataset.bin (same wire format the real pipeline emits)
with random ActiveBits-hot observations and a sparse reward signal, so the
scaled C++ harness can be exercised end-to-end without the heavy audio
toolchain. Pure stdlib — no numpy / venv required.

    python make_synthetic.py --steps 5000 --output output/synth.bin
"""

import argparse
import os
import random
import struct

# Must match audio_pred_scaled/dataset.hpp.
N_FREQ_BINS = 158
N_MAG_BINS = 158
OBS_DIM = N_FREQ_BINS * N_MAG_BINS   # 24964
ACTIVE_BITS = N_FREQ_BINS            # one set bit per frequency column
PACKED_BYTES = (OBS_DIM + 7) // 8    # 3121
RECORD_BYTES = PACKED_BYTES + 1      # 3122

MAGIC = b"APBD"
VERSION = 1
HEADER_STRUCT = struct.Struct("<4sIQ")


def make_record(rng: random.Random) -> bytes:
    """One record: ACTIVE_BITS distinct set bits (one per column, like the real
    grid where each frequency bin lights exactly one magnitude row)."""
    record = bytearray(RECORD_BYTES)
    for col in range(N_FREQ_BINS):
        row = rng.randrange(N_MAG_BINS)
        idx = row * N_FREQ_BINS + col
        record[idx >> 3] |= 1 << (idx & 7)
    # Sparse reward in {-1, 0, +1}, mostly 0.
    r = rng.choices((0, 1, -1), weights=(0.92, 0.04, 0.04))[0]
    struct.pack_into("<b", record, PACKED_BYTES, r)
    return bytes(record)


def main() -> None:
    p = argparse.ArgumentParser(description="Synthesize a scaled APBD dataset.")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="output/synth.bin")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    rng = random.Random(args.seed)

    with open(args.output, "wb") as f:
        f.write(HEADER_STRUCT.pack(MAGIC, VERSION, args.steps))
        for _ in range(args.steps):
            f.write(make_record(rng))

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"wrote {args.steps} steps x {OBS_DIM}-dim ({ACTIVE_BITS}-hot) "
          f"-> {args.output} ({size_mb:.1f} MiB)")


if __name__ == "__main__":
    main()
