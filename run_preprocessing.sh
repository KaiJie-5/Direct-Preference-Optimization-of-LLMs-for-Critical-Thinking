#!/bin/bash
set -euo pipefail

# Usage:
#   bash run_preprocessing_energy.sh /path/to/transcripts-energy.html
#
# INPUT_PATH may be either one HTML file containing multiple interviews or a
# directory containing one HTML file per interview.

CONDA_ENV="${CONDA_ENV:-dpo}"
PROJECT_DIR="${PROJECT_DIR:-/iridisfs/home/kjl1a21/Direct-Preference-Optimization-of-LLMs-for-Critical-Thinking}"
SCRATCH_DPO="${SCRATCH_DPO:-/iridisfs/scratch/kjl1a21/DPO}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_DPO}/data/transcripts-energy-preprocessed}"
INPUT_PATH="${1:-${INPUT_PATH:-}}"

if [[ -z "${INPUT_PATH}" ]]; then
    echo "Usage: bash run_preprocessing_energy.sh /path/to/transcripts-energy.html" >&2
    echo "The input may also be a directory containing HTML interview files." >&2
    exit 2
fi

if [[ ! -e "${INPUT_PATH}" ]]; then
    echo "Input path does not exist: ${INPUT_PATH}" >&2
    exit 2
fi

source ~/.bashrc
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

echo "Project directory: ${PROJECT_DIR}"
echo "Input path: ${INPUT_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Python: $(which python)"
python --version

python -m preprocessing.cli html \
    --input-path "${INPUT_PATH}" \
    --raw-html-dir "${OUTPUT_DIR}/raw_html" \
    --segments-dir "${OUTPUT_DIR}/segments_jsonl" \
    --manifest-path "${OUTPUT_DIR}/preprocessing_manifest.json" \
    --dataset-id "transcripts-energy" \
    --domain "energy services" \
    --overwrite

echo "Preprocessing complete."
echo "Segments with interview_turns: ${OUTPUT_DIR}/segments_jsonl"
echo "Manifest: ${OUTPUT_DIR}/preprocessing_manifest.json"
