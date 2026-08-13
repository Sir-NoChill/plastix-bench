#!/usr/bin/env bash
# One-shot driver for the SCALED audio-prediction benchmark (158x158 = 24964-dim).
#
# Builds il_audio_pred_scaled if needed, generates a packed dataset via the
# scaled prepare-cpp.py if missing (real FluidSynth audio), then runs the
# imprinting learner on it.
#
# Usage:
#   ./run.sh                  # 60 s dataset, all steps -> il_predictions.csv
#   DURATION=3600 ./run.sh    # full 1-hour benchmark
#   ./run.sh 5000             # only the first 5000 steps
#
# For a quick check WITHOUT the audio toolchain, generate a synthetic dataset
# instead and point the binary at it:
#   python make_synthetic.py --steps 5000 --output output/synth.bin
#   ../../build/default/il_audio_pred_scaled output/synth.bin 0 il_predictions.csv

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
examples_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$examples_dir/.." && pwd)"
build_dir="$repo_root/build/default"

duration="${DURATION:-60}"
seed="${SEED:-42}"
data_dir="${DATA_DIR:-$script_dir/output}"
dataset_path="$data_dir/dataset.bin"
max_steps="${1:-0}"
out_csv="${2:-il_predictions.csv}"

bin="$build_dir/il_audio_pred_scaled"
if [[ ! -x "$bin" ]]; then
  echo ">>> Building il_audio_pred_scaled ..."
  cmake --preset default > /dev/null
  cmake --build "$build_dir" --target il_audio_pred_scaled
fi

if [[ "${FORCE:-0}" == "1" || ! -f "$dataset_path" ]]; then
  echo ">>> Generating scaled dataset (duration=${duration}s, seed=${seed}) ..."
  mkdir -p "$data_dir"
  # Run via uv from the examples project so the apb deps + the ./FluidR3_GM.sf2
  # default path resolve (the soundfont lives in the parent examples dir).
  ( cd "$examples_dir" && \
    uv run --project . python "$script_dir/prepare-cpp.py" --generate \
      --data-dir "$data_dir" \
      --output "$dataset_path" \
      --duration "$duration" \
      --seed "$seed" )
else
  echo ">>> Reusing dataset at $dataset_path (FORCE=1 to regenerate)"
fi

echo ">>> Running il_audio_pred_scaled ($max_steps steps, 0 = all) ..."
exec "$bin" "$dataset_path" "$max_steps" "$out_csv"
