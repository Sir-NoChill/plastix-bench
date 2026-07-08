#!/usr/bin/env python3
"""CUDA profiling harness — run a bench's GPU binary under nsys / nvprof.

Wraps the compiled CUDA benchmark binary (the dedicated `cuda/` impl, or the
CUDA-built `plastix/` impl) in an NVIDIA profiler and drops the reports under
`_profiles/`. Two backends are supported and auto-detected:

  nsys   (Nsight Systems)  -> <name>.nsys-rep timeline + cuda_gpu_kern_sum /
                              cuda_api_sum summary CSVs (via `nsys stats`).
  nvprof (legacy)          -> <name>.nvprof.csv  GPU-kernel + API summary
                              (+ optional <name>.nvvp visual timeline).

`--tool {auto,nsys,nvprof,both}` (default auto = nsys if present else nvprof).
Each backend is gated on `shutil.which`; a missing tool is reported and skipped
rather than failing. Extra flags after `--` pass through to the bench binary.

    uv run python profile.py --bench 06_ccwc_ncp                 # auto
    uv run python profile.py --bench 06_ccwc_ncp --tool nvprof
    uv run python profile.py --bench 06_ccwc_ncp --tool both -- --max-steps 200

Requires a GPU and the CUDA build (`just build-cuda`); `just profile <bench>`
builds it first. ncu (Nsight Compute) is not installed here — if you add it, a
per-kernel `ncu --set full` path slots in next to `_run_nsys` below.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILES = HERE / "_profiles"
DATA_DIR = HERE / "common" / "pytorch" / "data"


def _binary(bench: str, impl: str, build_dir: str) -> Path:
    """Mirror the sentinel contract: <build>/<bench>/<impl>/run_benchmark."""
    return HERE / build_dir / bench / impl / "run_benchmark"


def _bench_args(out_dir: Path, extra: list[str]) -> list[str]:
    """Minimal args a compiled bench binary accepts (the sentinel strips
    --build-dir before exec, so we do not pass it). Default to a short run."""
    args = ["--out-dir", str(out_dir), "--seed", "0",
            "--data-dir", str(DATA_DIR)]
    if not any(a in ("--quick", "--max-steps") for a in extra):
        args.append("--quick")
    return args + extra


def _run(cmd: list[str]) -> int:
    print("    $ " + " ".join(cmd))
    return subprocess.call(cmd)


def _run_nsys(binary: Path, bargs: list[str], stem: Path) -> bool:
    if not shutil.which("nsys"):
        print("[profile] nsys not found — skipping (install NVIDIA Nsight Systems)")
        return False
    rc = _run(["nsys", "profile", "-o", str(stem), "--force-overwrite", "true",
               "--stats=true", str(binary), *bargs])
    if rc != 0:
        print(f"[profile] nsys profile exited {rc}")
        return False
    # Post-process the .nsys-rep into summary CSVs (best-effort). `nsys stats`
    # wants one --report per report and --force-export to refresh a stale sqlite;
    # --format csv --output <stem> writes <stem>_<report>.csv.
    _run(["nsys", "stats", "--report", "cuda_gpu_kern_sum",
          "--report", "cuda_api_sum", "--format", "csv",
          "--force-export=true", "--output", str(stem), f"{stem}.nsys-rep"])
    print(f"[profile] nsys -> {stem}.nsys-rep (+ {stem.name}_cuda_gpu_kern_sum.csv)")
    return True


def _run_nvprof(binary: Path, bargs: list[str], stem: Path) -> bool:
    if not shutil.which("nvprof"):
        print("[profile] nvprof not found — skipping (legacy CUDA toolkit tool)")
        return False
    csv_out = f"{stem}.nvprof.csv"
    rc = _run(["nvprof", "--csv", "--log-file", csv_out,
               "--normalized-time-unit", "ns", str(binary), *bargs])
    if rc != 0:
        print(f"[profile] nvprof exited {rc} (see {csv_out})")
        return False
    print(f"[profile] nvprof -> {csv_out}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="benchmark dir, e.g. 06_ccwc_ncp")
    ap.add_argument("--impl", default="cuda", help="impl to profile (default: cuda)")
    ap.add_argument("--build-dir", default="build-cuda",
                    help="build tree holding the GPU binary (default: build-cuda)")
    ap.add_argument("--tool", choices=("auto", "nsys", "nvprof", "both"),
                    default="auto")
    ap.add_argument("--out-dir", type=Path, default=PROFILES)
    ap.add_argument("passthrough", nargs="*",
                    help="extra flags forwarded to the bench binary (after --)")
    args = ap.parse_args()

    binary = _binary(args.bench, args.impl, args.build_dir)
    if not binary.exists():
        sys.exit(f"[profile] binary not found: {binary}\n"
                 f"          build it first (just build-cuda) or check --impl/--build-dir")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"{args.bench}_{args.impl}"
    bargs = _bench_args(args.out_dir / "_run", args.passthrough)

    tool = args.tool
    if tool == "auto":
        tool = "nsys" if shutil.which("nsys") else "nvprof"
        print(f"[profile] auto-selected tool: {tool}")

    ran = False
    if tool in ("nsys", "both"):
        ran |= _run_nsys(binary, bargs, stem)
    if tool in ("nvprof", "both"):
        ran |= _run_nvprof(binary, bargs, stem)

    if not ran:
        sys.exit("[profile] no profiler ran — install nsys or nvprof, and ensure a GPU is visible")
    print(f"[profile] done — reports under {args.out_dir}/")


if __name__ == "__main__":
    main()
