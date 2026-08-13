# Audio Prediction Benchmark — Scaled (~10×)

A copy of the [audio-prediction benchmark](../README.md) with the observation
sampling scaled up ~10×, to give the imprinting learner vastly more observation
"boxes" to build pattern/memory units on top of — a step toward the
millions-of-neurons goal.

| | Original | Scaled (here) |
|---|---|---|
| frequency bins | 50 | **158** |
| magnitude bins | 50 | **158** |
| observation dim (grid boxes) | 2500 | **24964** (≈10×) |
| active boxes / step | 50 | **158** |
| harness | `il_audio_pred` | `il_audio_pred_scaled` |
| capacity | 16384 | **32768** |

Only the *sampling* is scaled — the preprocessing logic is unchanged and reused
from the parent `../src/audio_prediction_benchmark` package (it already takes
`n_freq_bins` / `n_mag_bins` as parameters). The C++ side (`dataset.hpp`,
the harness) is copied because it hard-codes `ObservationDim`.

The decomposition 158×158 splits the 10× roughly evenly between frequency and
magnitude resolution (158 < 512 FFT components, so frequency stays meaningful).
To re-tune, change the constants in `dataset.hpp`, `prepare-cpp.py`, and
`make_synthetic.py` together (their product must match), keeping
`ObservationDim` < 32768 — see the cap note below.

## Run it

```bash
# Real audio (needs the soundfont + uv; soundfont fetched by ../../../../setup.py)
./run.sh                  # 60 s dataset, all steps
DURATION=3600 ./run.sh    # full 1-hour benchmark

# Or the steps by hand, from the parent examples/ dir:
uv run --project . python audio_pred_scaled/prepare-cpp.py --generate \
    --duration 60 --seed 42 \
    --data-dir audio_pred_scaled/output --output audio_pred_scaled/output/dataset.bin
../build/default/il_audio_pred_scaled audio_pred_scaled/output/dataset.bin 0 preds.csv
```

### Without the audio toolchain

`make_synthetic.py` writes a well-formed dataset (158-hot random observations,
sparse rewards) so the scaled harness can be exercised with no FluidSynth:

```bash
python make_synthetic.py --steps 5000 --output output/synth.bin
../../build/default/il_audio_pred_scaled output/synth.bin 0 preds.csv
# Force generation to fire (default eta throttles it):
IL_ETA=2.0 IL_ALPHA=0.01 ../../build/default/il_audio_pred_scaled output/synth.bin 500
```

The harness reads the same `IL_*` overrides as the original, plus `IL_CAPACITY`.

## The feature-id width (the scaling wall, now configurable)

Pattern members and memory sources are stored as `il::FeatureIdRef` in the arena
(`include/imprinting/feature_arena.hpp`). Its width is selected at build time by
the CMake cache var **`IL_FEATURE_ID_BITS`** (16 / 32 / 64):

| width | max features | note |
|---|---|---|
| 16 (default) | 32767 | smallest footprint; 24964 obs leave ~7800 ids for generated units |
| 32 | ~2.1e9 | millions-of-neurons scale |
| 64 | unbounded | |

The default-16 build caps growth — a synthetic run at `IL_ETA=2.0` reaches
~28000 features within ~500 steps and would throw once a feature id ≥ 32768 is
referenced. To push the *network* toward millions of neurons:

```bash
cmake --preset default -DIL_FEATURE_ID_BITS=32
cmake --build build/default --target il_audio_pred_scaled
IL_CAPACITY=2000000 IL_ETA=2.0 ../../build/default/il_audio_pred_scaled output/synth.bin
```

(`IL_FEATURE_ID_BITS` only lifts the *addressing* cap; actual neuron count is
still bounded by the runtime `IL_CAPACITY` and the generation/removal budget.)
