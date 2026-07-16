#!/bin/bash
#SBATCH --job-name=dpo_codes_4688
#SBATCH --partition=quad_h200
#SBATCH --account=ecs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-12:00:00
#SBATCH --output=dpo_codes_4688_%j.out
#SBATCH --error=dpo_codes_4688_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kjl1a21@soton.ac.uk

set -eo pipefail
set +u

CONDA_ENV="${CONDA_ENV:-dpo}"
PROJECT_DIR="${PROJECT_DIR:-/iridisfs/home/kjl1a21/Direct-Preference-Optimization-of-LLMs-for-Critical-Thinking}"
SCRATCH_DPO="${SCRATCH_DPO:-/iridisfs/scratch/kjl1a21/DPO}"

MODEL_PATH="${MODEL_PATH:-${SCRATCH_DPO}/models/teacher/deepseek-ai__DeepSeek-R1-Distill-Llama-70B}"
SEGMENTS_PATH="${SEGMENTS_PATH:-${SCRATCH_DPO}/data/UKDA-4688-rtf-preprocessed/segments_jsonl}"
EXCLUDE_RECORDS_PATH="${EXCLUDE_RECORDS_PATH:-${SCRATCH_DPO}/data/UKDA-4688-rtf-preprocessed/enrichment_exclusions.jsonl}"
CODEBOOK_PATH="${SCRATCH_DPO}/data/codebooks/example_codes_v1.json"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_DPO}/data/UKDA-4688-rtf-enriched}"
PROMPT_PATH="${PROMPT_PATH:-${PROJECT_DIR}/prompts/enrichment/self_consistency_four_codes.txt}"
RESUME_DIR="${RESUME_DIR:-}"

# The codebook and research questions are fixed for this run.
RESEARCH_QUESTIONS=(
    "What factors and processes affect household choices of where to live?"
    "How does employment shape family life?"
)

CONTEXT_SCOPE="turn_window"
CONTEXT_TURNS_BEFORE=20
CONTEXT_TURNS_AFTER=20
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

require_directory "${MODEL_PATH}"
require_directory "${SEGMENTS_PATH}"
require_file "${EXCLUDE_RECORDS_PATH}"
require_file "${CODEBOOK_PATH}"
require_file "${PROMPT_PATH}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${RESUME_DIR}" ]]; then
    require_directory "${RESUME_DIR}"
    RUN_DIR="$(readlink -f "${RESUME_DIR}")"
else
    RUN_ID="ukda4688_single_pass_four_codes_${SLURM_JOB_ID}_${TIMESTAMP}"
    RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
    mkdir -p "${RUN_DIR}"
fi
exec > >(tee -a "${RUN_DIR}/enrichment.log") 2> >(tee -a "${RUN_DIR}/enrichment.err" >&2)

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

ARGS=(
    --segments-path "${SEGMENTS_PATH}"
    --exclude-records-path "${EXCLUDE_RECORDS_PATH}"
    --output-dir "${RUN_DIR}"
    --codebook-path "${CODEBOOK_PATH}"
    --strategy single_pass
    --context-scope "${CONTEXT_SCOPE}"
    --context-turns-before "${CONTEXT_TURNS_BEFORE}"
    --context-turns-after "${CONTEXT_TURNS_AFTER}"
    --prompt-path "${PROMPT_PATH}"
    --teacher-backend transformers
    --model-path "${MODEL_PATH}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
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

if [[ -n "${RESUME_DIR}" ]]; then
    ARGS+=(--resume "${RUN_DIR}")
    COMMAND_PATH="${RUN_DIR}/resume_command_${SLURM_JOB_ID}_${TIMESTAMP}.txt"
else
    COMMAND_PATH="${RUN_DIR}/command.txt"
fi
printf '%q ' python -m enrichment.cli "${ARGS[@]}" > "${COMMAND_PATH}"
printf '\n' >> "${COMMAND_PATH}"
python -m enrichment.cli "${ARGS[@]}"

echo "Enrichment complete: ${RUN_DIR}"
echo "This single-pass run is final stage-one output; do not submit it to debate ranking."
