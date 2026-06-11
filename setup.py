#!/usr/bin/env python3
"""Setup pipeline — fetch the datasets and soundfonts the suite needs at run time.

None of these artifacts are committed to git (see .gitignore); this script
obtains them on demand, reusing the same download paths the benchmarks use so
the result is identical to running each bench once. It is idempotent: every
stage skips work whose output already exists.

Stages (run in this order):

  csvs       ETTh1 / elec2 / appliances CSVs                       (01, 03, 04)
  mnist      (permuted) sequential MNIST via torchvision           (06)
  shd        Spiking Heidelberg Digits via tonic + .plxbin cache   (08)
  soundfont  FluidR3_GM.sf2 used to synthesise the audio dataset   (09)
  audio      generate 09's packed dataset.bin (needs FluidSynth)   (09)

Usage:
    uv run python setup.py                     # everything except audio gen
    uv run python setup.py --all               # everything, incl. audio
    uv run python setup.py --stages csvs,soundfont
    uv run python setup.py --data-dir common/pytorch/data --skip mnist

The `audio` stage additionally needs a system FluidSynth (its Python deps come
from the 09_imprintin_learner/cpp/examples/ sub-project, run via uv):
    # Arch:   sudo pacman -S fluidsynth
    # Debian: sudo apt install fluidsynth
    uv run python setup.py --stages soundfont,audio
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "common/pytorch" / "data"
EXAMPLES_DIR = HERE / "09_imprintin_learner" / "cpp" / "examples"

# Dataset URLs — mirror the constants in each bench's pytorch/run_benchmark.py.
ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
ELEC2_URL = "https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/elec.csv"
APPLIANCES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"

# SoundFont used by the audio-prediction generator (09). Same source as the
# wget in 09_imprintin_learner/cpp/examples/README.md.
SOUNDFONT_URL = "https://sourceforge.net/projects/androidframe/files/soundfonts/FluidR3_GM.sf2/download"
SOUNDFONT_PATH = EXAMPLES_DIR / "FluidR3_GM.sf2"

STAGES = ("csvs", "mnist", "shd", "soundfont", "audio")
# `audio` is opt-in: it needs a system FluidSynth + the `audio` extra, so it is
# excluded from the default run.
DEFAULT_STAGES = ("csvs", "mnist", "shd", "soundfont")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rel(p: Path) -> Path:
    """Display path relative to the repo root when possible, else absolute."""
    try:
        return p.relative_to(HERE)
    except ValueError:
        return p


def _stream_download(url: str, dest: Path, *, min_bytes: int = 1) -> None:
    """Stream `url` to `dest` (atomic via a .part temp). Skips if dest exists
    and is at least `min_bytes` long."""
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"  [skip] {_rel(dest)} already present "
              f"({dest.stat().st_size / 1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get ] {url}\n         -> {_rel(dest)}")
    req = urllib.request.Request(url, headers={"User-Agent": "plastix-bench/0.1"})
    with urllib.request.urlopen(req) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.replace(dest)
    print(f"         done ({dest.stat().st_size / 1e6:.1f} MB)")


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"  [run ] {' '.join(cmd)}" + (f"   (cwd={cwd})" if cwd else ""))
    subprocess.run(cmd, cwd=cwd, check=True)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_csvs(data_dir: Path) -> None:
    print("[csvs] ETTh1 / elec2 / appliances")
    _stream_download(ETTH1_URL, data_dir / "ETTh1.csv", min_bytes=1_000)
    _stream_download(ELEC2_URL, data_dir / "elec2.csv", min_bytes=1_000)
    _stream_download(APPLIANCES_URL, data_dir / "energydata_complete.csv",
                     min_bytes=1_000)


def stage_mnist(data_dir: Path) -> None:
    print("[mnist] sequential MNIST (torchvision)")
    try:
        from torchvision.datasets import MNIST
    except ImportError as e:
        raise SystemExit(
            "[mnist] torchvision is required. Run via `uv run python setup.py` "
            f"so the declared deps are available. ({e})")
    MNIST(root=str(data_dir), train=True, download=True)
    MNIST(root=str(data_dir), train=False, download=True)
    print(f"  [ok  ] MNIST under {_rel(data_dir / 'MNIST')}")


def stage_shd(data_dir: Path) -> None:
    print("[shd] Spiking Heidelberg Digits + .plxbin cache")
    # Reuse the bench's own exporter: it downloads the HDF5 via tonic and
    # materialises the train/test .plxbin files the C++ benchmark reads.
    exporter = HERE / "08_snn_shd" / "pytorch" / "data.py"
    _run([sys.executable, str(exporter), "--export",
          "--data-dir", str(data_dir), "--n-bins", "50"])


def stage_soundfont(_data_dir: Path) -> None:
    print("[soundfont] FluidR3_GM.sf2")
    # ~140 MB; place it where the audio generator looks by default (its CWD).
    _stream_download(SOUNDFONT_URL, SOUNDFONT_PATH, min_bytes=1_000_000)


def stage_audio(_data_dir: Path) -> None:
    print("[audio] generate 09 audio-prediction dataset.bin")
    if not SOUNDFONT_PATH.exists():
        raise SystemExit(
            "[audio] soundfont missing — run the `soundfont` stage first.")
    if shutil.which("fluidsynth") is None:
        print("  [skip] system FluidSynth not found; cannot synthesise audio.\n"
              "         Install it (Arch: `sudo pacman -S fluidsynth`, "
              "Debian: `sudo apt install fluidsynth`),\n"
              "         then re-run: uv run python setup.py --stages audio")
        return
    out = EXAMPLES_DIR / "output" / "dataset.bin"
    if out.exists():
        print(f"  [skip] {_rel(out)} already present "
              f"({out.stat().st_size / 1e6:.1f} MB)")
        return
    # prepare-cpp.py forwards to the apb generator; run it from the examples
    # dir so (a) the default ./FluidR3_GM.sf2 path resolves and (b) `uv run`
    # picks up examples/pyproject.toml, which declares the audio-gen deps
    # (pretty_midi / pyfluidsynth / soundfile) and the apb package itself.
    _run(["uv", "run", "python", "prepare-cpp.py",
          "--generate", "--data-dir", "output", "--output", "output/dataset.bin",
          "--duration", "60", "--seed", "42"],
         cwd=EXAMPLES_DIR)
    print(f"  [ok  ] {_rel(out)}")


STAGE_FNS = {
    "csvs": stage_csvs,
    "mnist": stage_mnist,
    "shd": stage_shd,
    "soundfont": stage_soundfont,
    "audio": stage_audio,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Fetch datasets + soundfonts for the plastix-bench suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help=f"dataset root (default: {DEFAULT_DATA_DIR.relative_to(HERE)})")
    p.add_argument("--stages", default=None,
                   help="comma-separated subset of: " + ",".join(STAGES))
    p.add_argument("--skip", default=None,
                   help="comma-separated stages to skip")
    p.add_argument("--all", action="store_true",
                   help="run every stage, including `audio`")
    args = p.parse_args()

    if args.stages:
        requested = [s.strip() for s in args.stages.split(",") if s.strip()]
        unknown = [s for s in requested if s not in STAGE_FNS]
        if unknown:
            p.error(f"unknown stage(s): {', '.join(unknown)}; "
                    f"choose from {', '.join(STAGES)}")
    elif args.all:
        requested = list(STAGES)
    else:
        requested = list(DEFAULT_STAGES)

    if args.skip:
        skip = {s.strip() for s in args.skip.split(",")}
        requested = [s for s in requested if s not in skip]

    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"plastix-bench setup — data-dir={data_dir}")
    print(f"stages: {', '.join(requested) or '(none)'}\n")

    for stage in requested:
        STAGE_FNS[stage](data_dir)
        print()

    print("setup complete.")


if __name__ == "__main__":
    main()
