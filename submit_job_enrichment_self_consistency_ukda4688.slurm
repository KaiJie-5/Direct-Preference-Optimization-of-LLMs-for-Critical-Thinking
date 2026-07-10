#!/bin/bash
#SBATCH --job-name=dpo_enrich_4688
#SBATCH --partition=quad_h200
#SBATCH --account=ecs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-12:00:00
#SBATCH --output=dpo_enrich_4688_%j.out
#SBATCH --error=dpo_enrich_4688_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kjl1a21@soton.ac.uk

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-dpo}"
PROJECT_DIR="${PROJECT_DIR:-/iridisfs/home/kjl1a21/Direct-Preference-Optimization-of-LLMs-for-Critical-Thinking}"
SCRATCH_DPO="${SCRATCH_DPO:-/iridisfs/scratch/kjl1a21/DPO}"

MODEL_PATH="${MODEL_PATH:-${SCRATCH_DPO}/models/teacher/deepseek-ai__DeepSeek-R1-Distill-Llama-70B}"
SEGMENTS_PATH="${SEGMENTS_PATH:-${SCRATCH_DPO}/data/UKDA-4688-rtf-preprocessed/segments_jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_DPO}/data/UKDA-4688-rtf-enriched}"
PROMPT_PATH="${PROMPT_PATH:-${PROJECT_DIR}/prompts/enrichment/self_consistency_placeholder.txt}"

# These analysis inputs are deliberately required at submission time. The UKDA
# archive does not define this project's codebook or research questions.
CODEBOOK_PATH="${CODEBOOK_PATH:-}"
RESEARCH_QUESTIONS_FILE="${RESEARCH_QUESTIONS_FILE:-}"

CONTEXT_SCOPE="turn_window"
CONTEXT_TURNS_BEFORE=20
CONTEXT_TURNS_AFTER=20
SELF_CONSISTENCY_SAMPLES=5
MAX_NEW_TOKENS=32768
TEMPERATURE=0.6
TOP_P=0.95
LIMIT="${LIMIT:-}"

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

if [[ -z "${CODEBOOK_PATH}" ]]; then
    echo "CODEBOOK_PATH must be supplied when submitting this job." >&2
    exit 2
fi
if [[ -z "${RESEARCH_QUESTIONS_FILE}" ]]; then
    echo "RESEARCH_QUESTIONS_FILE must point to a UTF-8 file with one question per line." >&2
    exit 2
fi
require_directory "${MODEL_PATH}"
require_directory "${SEGMENTS_PATH}"
require_file "${CODEBOOK_PATH}"
require_file "${PROMPT_PATH}"
require_file "${RESEARCH_QUESTIONS_FILE}"

mapfile -t RESEARCH_QUESTIONS < <(
    sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
        -e '/^$/d' -e '/^#/d' "${RESEARCH_QUESTIONS_FILE}"
)
if [[ ${#RESEARCH_QUESTIONS[@]} -eq 0 ]]; then
    echo "RESEARCH_QUESTIONS_FILE contains no research questions." >&2
    exit 2
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="ukda4688_self_consistency_${SLURM_JOB_ID}_${TIMESTAMP}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/enrichment.log") 2> >(tee -a "${RUN_DIR}/enrichment.err" >&2)

source ~/.bashrc
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

ARGS=(
    --segments-path "${SEGMENTS_PATH}"
    --output-dir "${RUN_DIR}"
    --codebook-path "${CODEBOOK_PATH}"
    --strategy self_consistency
    --context-scope "${CONTEXT_SCOPE}"
    --context-turns-before "${CONTEXT_TURNS_BEFORE}"
    --context-turns-after "${CONTEXT_TURNS_AFTER}"
    --prompt-path "${PROMPT_PATH}"
    --teacher-backend transformers
    --model-path "${MODEL_PATH}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --self-consistency-samples "${SELF_CONSISTENCY_SAMPLES}"
    --force-think-prefix
    --torch-dtype auto
    --device-map auto
)
if [[ -n "${LIMIT}" ]]; then
    ARGS+=(--limit "${LIMIT}")
fi
for question in "${RESEARCH_QUESTIONS[@]}"; do
    ARGS+=(--research-question "${question}")
done

printf '%q ' python -m enrichment.cli "${ARGS[@]}" > "${RUN_DIR}/command.txt"
printf '\n' >> "${RUN_DIR}/command.txt"
python -m enrichment.cli "${ARGS[@]}"

echo "Enrichment complete: ${RUN_DIR}"
