# Audio Prediction Benchmark

Implementation of the Audio Prediction Benchmark from Chapter 9 (Section 9.3) of Khurram Javed's thesis. Generates a continuous audio stream of instrument chords followed by delayed scalar rewards, then preprocesses the audio into binary observation vectors via FFT + binarization.

## System Prerequisites

**Option A**: Install FluidSynth system-wide (if you have sudo):

```bash
sudo apt install fluidsynth fluid-soundfont-gm
```

**Option B**: Build FluidSynth locally (no sudo required):

```bash
mkdir -p _build_deps && cd _build_deps
git clone --depth 1 --branch v2.3.4 https://github.com/FluidSynth/fluidsynth.git
cd fluidsynth && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$(pwd)/../../../venv -DCMAKE_BUILD_TYPE=Release \
  -Denable-libsndfile=off -Denable-dbus=off -Denable-pulseaudio=off \
  -Denable-jack=off -Denable-pipewire=off -Denable-sdl2=off -Denable-readline=off
cmake --build . -j$(nproc)
cmake --install .
cd ../../..
```

Then set `LD_LIBRARY_PATH` before running any scripts:

```bash
export LD_LIBRARY_PATH=$(pwd)/venv/lib:$LD_LIBRARY_PATH
```

**SoundFont**: Download `FluidR3_GM.sf2` into the project root:

```bash
wget -O FluidR3_GM.sf2 "https://sourceforge.net/projects/androidframe/files/soundfonts/FluidR3_GM.sf2/download"
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

Or install dependencies without the package:

```bash
pip install -r requirements.txt
```

## Quick Start

Generate a 60-second test dataset:

```bash
apb-generate --duration 60 --seed 42
```

This creates `./output/` with:
- `audio.wav` — raw audio
- `rewards.npy` — per-step reward signal
- `metadata.json` — event timeline and parameters

Generate the full 1-hour benchmark:

```bash
apb-generate --duration 3600
```

Visualize a dataset:

```bash
apb-visualize --data-dir ./output
```

## Run the imprinting learner (C++)

The C++ harness (`il_audio_pred`) consumes a *packed binary* dataset
(`output/dataset.bin`), not the raw `audio.wav`. `prepare-cpp.py` both generates
the audio and exports that binary in one step. Run these from this `examples/`
directory (with the venv active) so the SoundFont and package resolve.

One-shot — builds the binary, generates a dataset if missing, runs the learner:

```bash
./run.sh                  # 60 s dataset, all steps -> il_predictions.csv
DURATION=3600 ./run.sh    # full 1-hour benchmark
./run.sh 5000             # only the first 5000 steps
```

Or the individual steps:

```bash
# 0. build the harness (from the repo root)
cmake --preset default && cmake --build build/default --target il_audio_pred

# 1. generate audio + export the packed dataset (-> ./output + ./output/dataset.bin)
python prepare-cpp.py --generate --duration 3600 --seed 42

# 2. run the learner (0 = all steps); IL_* env vars sweep hyperparameters
../build/default/il_audio_pred ./output/dataset.bin 0 il_predictions.csv
IL_ETA=2.0 ../build/default/il_audio_pred ./output/dataset.bin 10000

# 3. plot the prediction overlay (last 180 s of the 1-hour run)
python plot_il.py --predictions il_predictions.csv --data-dir ./output \
    --time-range 3420 3600 --output ./output/plots/il_last180s.png
```

To sweep hyperparameters, `sweep_il.py` runs the harness + plot per config and
names each output after its config:

```bash
python sweep_il.py --eta 0.05,0.1,0.2           # last 180 s by default
python sweep_il.py --delay-max 20,64 --jobs 4   # parallel sweep
python sweep_il.py --help                       # all knobs + options
```

The harness reads these `IL_*` overrides: `IL_GAMMA`, `IL_ALPHA`, `IL_ETA`,
`IL_EPSILON_Z`, `IL_K_PATTERN`, `IL_K_MEMORY`, `IL_MEMORY_DELAY_MIN/MAX`,
`IL_MEMORY_WINDOW_MIN/MAX`.

## Dataset Generation

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--duration` | 3600 | Total audio duration in seconds |
| `--inter-sound-time-range` | 15 30 | Min/max seconds between sounds |
| `--reward-delay-range` | 3 5 | Min/max seconds delay between chord and reward |
| `--combinations` | guitar:C_major:1 piano:C_major:-1 piano:D_major:0 | Sound-reward mappings |
| `--sample-rate` | 16384 | Audio sample rate in Hz |
| `--step-size` | 640 | Samples per time step |
| `--soundfont` | ./FluidR3_GM.sf2 (relative to CWD) | Path to SoundFont file |
| `--output-dir` | ./output | Output directory |
| `--seed` | None | Random seed |

### Combination Format

Each combination is `instrument:chord:reward`. Available instruments: `piano`, `guitar`, `electric_guitar`, `violin`, `trumpet`, `flute`, `organ`. Available chords: `C_major`, `D_major`, `E_major`, `F_major`, `G_major`, `A_major`, `B_major`, `C_minor`, `D_minor`, `E_minor`, `A_minor`, `C_major7`, `D_minor7`, `G7`, and single notes `C4` through `B4`.

## Preprocessing

The preprocessing pipeline converts raw audio into 2500-dimensional binary observation vectors:

1. Extract a 1024-sample window (640 new + 384 overlap) at each time step
2. Apply FFT, take magnitude of first 512 frequency components
3. Divide into 50 frequency bins, compute max magnitude per bin
4. Quantize each magnitude into one of 50 rows on a grid
5. Flatten to a binary vector with exactly 50 ones

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `step_size` | 640 | Samples per time step |
| `fft_size` | 1024 | FFT window size |
| `n_freq_bins` | 50 | Frequency bins (grid columns) |
| `n_mag_bins` | 50 | Magnitude bins (grid rows) |
| `max_magnitude` | 50.0 | Max magnitude for clipping |

## Python API

```python
from audio_prediction_benchmark import AudioPredictionDataset, StreamingDataLoader

# Load dataset with preprocessing
dataset = AudioPredictionDataset("./output", preprocess=True)

# Random access
obs, reward = dataset[100]
print(obs.shape)  # (2500,)
print(obs.sum())  # 50

# Streaming access
loader = StreamingDataLoader(dataset)
for obs, reward in loader:
    # Process one step at a time
    pass
```

## Visualization

Generate all plots from a dataset:

```bash
apb-visualize --data-dir ./output
```

Plots saved to `./output/plots/`:
- `waveform.png` — Audio waveform with chord/reward markers
- `spectrogram.png` — FFT spectrogram over time
- `fft_spectrum.png` — FFT magnitude for a single time step
- `binary_observation.png` — 2D heatmap of binarized observation
- `reward_signal.png` — Reward values over time
- `event_timeline.png` — Chord types and reward delivery timeline

## Benchmark Specification

- **Default variation**: Guitar C chord (+1), Piano C chord (-1), Piano D chord (0)
- **Timing**: 15-30s between chords, 3-5s delay between chord and reward
- **Duration**: ~1 hour (96,000 time steps at default settings)
- **Sampling**: 16,384 Hz, 640 samples per step (~40ms)
- **Observation**: 2500-dim binary vector (50 frequency bins x 50 magnitude bins)
