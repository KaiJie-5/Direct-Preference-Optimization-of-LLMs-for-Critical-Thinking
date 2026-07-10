#!/bin/bash

# Some HPC login shells export nounset through SHELLOPTS. Disable it before
# loading Conda's shell integration so system startup files cannot fail on
# optional variables.
set +u
set -eo pipefail

# Preprocess UKDA Study 4688 without modifying the source archive.
# Override INPUT_PATH or OUTPUT_DIR when the HPC layout changes.

CONDA_ENV="${CONDA_ENV:-dpo}"
PROJECT_DIR="${PROJECT_DIR:-/iridisfs/home/kjl1a21/Direct-Preference-Optimization-of-LLMs-for-Critical-Thinking}"
INPUT_PATH="${INPUT_PATH:-/iridisfs/scratch/kjl1a21/DPO/interview_datasets/UKDA-4688-rtf}"
OUTPUT_DIR="${OUTPUT_DIR:-/iridisfs/scratch/kjl1a21/DPO/data/UKDA-4688-rtf-preprocessed}"
OVERWRITE="${OVERWRITE:-false}"

if [[ ! -d "${INPUT_PATH}/rtf" ]]; then
    echo "UKDA 4688 transcript directory does not exist: ${INPUT_PATH}/rtf" >&2
    exit 2
fi

CONDA_BASE="${CONDA_BASE:-}"
if [[ -z "${CONDA_BASE}" && -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    CONDA_BASE="$("${CONDA_EXE}" info --base)"
fi
if [[ -z "${CONDA_BASE}" ]] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
fi
if [[ -z "${CONDA_BASE}" ]]; then
    for candidate in "${HOME}/miniconda3" "${HOME}/anaconda3"; do
        if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
            CONDA_BASE="${candidate}"
            break
        fi
    done
fi
if [[ -z "${CONDA_BASE}" || ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "Unable to locate Conda. Set CONDA_BASE to the Conda installation directory." >&2
    exit 2
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

ARGS=(
    rtf
    --profile ukda-4688
    --input-path "${INPUT_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --strict-inventory
)
if [[ "${OVERWRITE}" == "true" ]]; then
    ARGS+=(--overwrite)
fi

echo "Input archive: ${INPUT_PATH}"
echo "Derived output: ${OUTPUT_DIR}"
python -m preprocessing.cli "${ARGS[@]}"

echo "Preprocessing complete"
echo "Manifest: ${OUTPUT_DIR}/preprocessing_manifest.json"
echo "QA report: ${OUTPUT_DIR}/preprocessing_qa.json"
echo "Target filter audit: ${OUTPUT_DIR}/target_filter_audit.jsonl"
