#!/bin/bash
#SBATCH --job-name=dpo_reflect_4688
#SBATCH --partition=quad_h200
#SBATCH --account=ecs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=2-12:00:00
#SBATCH --output=dpo_reflect_4688_%j.out
#SBATCH --error=dpo_reflect_4688_%j.err
#SBATCH --open-mode=append
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kjl1a21@soton.ac.uk

set -eo pipefail
set +u

CONDA_ENV="${CONDA_ENV:-dpo}"
PROJECT_DIR="${PROJECT_DIR:-/iridisfs/home/kjl1a21/Direct-Preference-Optimization-of-LLMs-for-Critical-Thinking}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/reflective_questions_enrichment_ukda4688.json}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
SOURCE_RUN_DIR="/iridisfs/scratch/kjl1a21/DPO/data/UKDA-4688-rtf-enriched/ukda4688_single_pass_four_codes_1228874_20260713_235755"
MODEL_PATH="/iridisfs/scratch/kjl1a21/DPO/models/teacher/deepseek-ai__DeepSeek-R1-Distill-Llama-70B"
PROMPT_PATH="${PROJECT_DIR}/prompts/enrichment/reflective_questions_enrichment_ukda4688.txt"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file does not exist: $1" >&2
        exit 2
    fi
}

require_directory() {
    if [[ ! -d "$1" ]]; then
        echo "Required directory does not exist: $1" >&2
        exit 2
    fi
}

require_file "${CONFIG_PATH}"
require_file "${PROMPT_PATH}"
require_directory "${SOURCE_RUN_DIR}"
require_file "${SOURCE_RUN_DIR}/run_manifest.json"
require_directory "${MODEL_PATH}"
if [[ -n "${RESUME_RUN_DIR}" ]]; then
    require_directory "${RESUME_RUN_DIR}"
    RESUME_RUN_DIR="$(readlink -f "${RESUME_RUN_DIR}")"
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
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

ARGS=(--config "${CONFIG_PATH}")
if [[ -n "${RESUME_RUN_DIR}" ]]; then
    ARGS+=(--resume "${RESUME_RUN_DIR}")
fi

echo "Starting UKDA 4688 stage-two reflective-question enrichment"
echo "Config: ${CONFIG_PATH}"
echo "Resume run: ${RESUME_RUN_DIR:-new run}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
date

python -m reflective_enrichment.cli "${ARGS[@]}"

echo "UKDA 4688 reflective-question enrichment complete"
date
