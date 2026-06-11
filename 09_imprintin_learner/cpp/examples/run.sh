#!/usr/bin/env bash
# One-shot driver: build the il_audio_pred binary if needed, generate the packed
# binary dataset via prepare-cpp.py if missing, then run the imprinting learner.
#
# Usage:
#   ./run.sh                  # defaults, reuse any cached dataset, all steps
#   ./run.sh 5000             # only feed the first 5000 steps to the learner
#   ./run.sh 0 preds.csv      # all steps; write predictions to preds.csv
#
# Knobs (env vars):
#   DURATION=60   seconds of audio to render (only used when (re)generating)
#   SEED=42       generation RNG seed
#   DATA_DIR=...  dataset/output directory (default: <examples>/output)
#   FORCE=1       regenerate the dataset even if dataset.bin exists
#   IL_*          hyperparameter overrides read by il_audio_pred (see its source)

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
build_dir="$repo_root/build/default"

duration="${DURATION:-60}"
seed="${SEED:-42}"
data_dir="${DATA_DIR:-$script_dir/output}"
dataset_path="$data_dir/dataset.bin"
max_steps="${1:-0}"
out_csv="${2:-il_predictions.csv}"

# Activate the local venv if present (the apb package + prepare-cpp.py need it).
if [[ -f "$script_dir/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$script_dir/venv/bin/activate"
fi
# If fluidsynth was built into the venv prefix (README option B), pick it up.
if [[ -d "$script_dir/venv/lib" ]]; then
  export LD_LIBRARY_PATH="$script_dir/venv/lib:${LD_LIBRARY_PATH:-}"
fi

bin="$build_dir/il_audio_pred"
if [[ ! -x "$bin" ]]; then
  echo ">>> Building il_audio_pred ..."
  cmake --preset default > /dev/null
  cmake --build "$build_dir" --target il_audio_pred
fi

if [[ "${FORCE:-0}" == "1" || ! -f "$dataset_path" ]]; then
  echo ">>> Generating dataset (duration=${duration}s, seed=${seed}) ..."
  mkdir -p "$data_dir"
  # cd so apb-generate finds ./FluidR3_GM.sf2 at its default path.
  ( cd "$script_dir" && \
    python prepare-cpp.py --generate \
      --data-dir "$data_dir" \
      --output "$dataset_path" \
      --duration "$duration" \
      --seed "$seed" )
else
  echo ">>> Reusing dataset at $dataset_path (FORCE=1 to regenerate)"
fi

echo ">>> Running il_audio_pred ($max_steps steps, 0 = all) ..."
exec "$bin" "$dataset_path" "$max_steps" "$out_csv"
