#!/bin/bash
#SBATCH --job-name=dpo_enrich_sc
#SBATCH --partition=quad_h200
#SBATCH --account=ecs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-12:00:00
#SBATCH --output=dpo_enrich_sc_%j.out
#SBATCH --error=dpo_enrich_sc_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kjl1a21@soton.ac.uk

# =========================
# Easy configuration block
# =========================
CONDA_ENV="dpo"
PROJECT_DIR="/iridisfs/home/kjl1a21/Direct-Preference-Optimization-of-LLMs-for-Critical-Thinking"
SCRATCH_DPO="/iridisfs/scratch/kjl1a21/DPO"

MODEL_PATH="${SCRATCH_DPO}/models/teacher/deepseek-ai__DeepSeek-R1-Distill-Llama-70B"
SEGMENTS_PATH="${SCRATCH_DPO}/data/transcripts-energy-preprocessed/segments_jsonl"
CODEBOOK_PATH="${SCRATCH_DPO}/data/codebooks/example_codes_v1.json"
OUTPUT_ROOT="${SCRATCH_DPO}/data/transcripts-energy-enriched"
PROMPT_PATH="${PROJECT_DIR}/prompts/enrichment/self_consistency_placeholder.txt"

STRATEGY="self_consistency"
CONTEXT_SCOPE="full_interview"
TEACHER_BACKEND="transformers"
# Smoke-run setting. Restore to 5 together with clearing LIMIT for the full run.
SELF_CONSISTENCY_SAMPLES=5
TEMPERATURE=0.6
TOP_P=0.95
MAX_NEW_TOKENS=32768
TORCH_DTYPE="auto"
DEVICE_MAP="auto"

# Add one or more research questions for this run, for example:
# RESEARCH_QUESTIONS=("How do participants describe smart technology use in relation to energy efficiency?")
RESEARCH_QUESTIONS=(
    "What are some ways that people experience interactions with energy services or energy management systems?"
    "What interactions do people have with energy services or energy management systems?"
    "What barriers or challenges do people face when interacting with energy services or energy management systems?"
)

# Keep LIMIT=1 for the next smoke run. Clear it only after inspecting that output.
LIMIT=""

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
require_file "${CODEBOOK_PATH}"
require_file "${PROMPT_PATH}"

# =========================
# Run folder and logging
# =========================
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="transcripts_energy_${STRATEGY}_deepseek_r1_distill_llama_70b"
RUN_ID="${RUN_NAME}_${SLURM_JOB_ID}_${TIMESTAMP}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

exec > >(tee -a "${RUN_DIR}/enrichment.log") 2> >(tee -a "${RUN_DIR}/enrichment.err" >&2)

echo "Run directory: ${RUN_DIR}"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
date

echo "Research questions:"
if [[ ${#RESEARCH_QUESTIONS[@]} -eq 0 ]]; then
    echo "  none supplied"
else
    for index in "${!RESEARCH_QUESTIONS[@]}"; do
        printf '  %d. %s\n' "$((index + 1))" "${RESEARCH_QUESTIONS[$index]}"
    done
fi

cat > "${RUN_DIR}/sbatch_config.txt" <<EOF
CONDA_ENV=${CONDA_ENV}
PROJECT_DIR=${PROJECT_DIR}
SCRATCH_DPO=${SCRATCH_DPO}
MODEL_PATH=${MODEL_PATH}
SEGMENTS_PATH=${SEGMENTS_PATH}
CODEBOOK_PATH=${CODEBOOK_PATH}
OUTPUT_ROOT=${OUTPUT_ROOT}
PROMPT_PATH=${PROMPT_PATH}
STRATEGY=${STRATEGY}
CONTEXT_SCOPE=${CONTEXT_SCOPE}
TEACHER_BACKEND=${TEACHER_BACKEND}
SELF_CONSISTENCY_SAMPLES=${SELF_CONSISTENCY_SAMPLES}
TEMPERATURE=${TEMPERATURE}
TOP_P=${TOP_P}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS}
TORCH_DTYPE=${TORCH_DTYPE}
DEVICE_MAP=${DEVICE_MAP}
LIMIT=${LIMIT}
RESEARCH_QUESTIONS=${RESEARCH_QUESTIONS[*]}
EOF

source ~/.bashrc
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

echo "Python: $(which python)"
python --version

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(f"gpu {index}:", torch.cuda.get_device_name(index))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

ENRICH_ARGS=(
    --segments-path "${SEGMENTS_PATH}"
    --output-dir "${RUN_DIR}"
    --codebook-path "${CODEBOOK_PATH}"
    --strategy "${STRATEGY}"
    --context-scope "${CONTEXT_SCOPE}"
    --prompt-path "${PROMPT_PATH}"
    --teacher-backend "${TEACHER_BACKEND}"
    --model-path "${MODEL_PATH}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --self-consistency-samples "${SELF_CONSISTENCY_SAMPLES}"
    --force-think-prefix
    --torch-dtype "${TORCH_DTYPE}"
    --device-map "${DEVICE_MAP}"
)

if [[ -n "${LIMIT}" ]]; then
    ENRICH_ARGS+=(--limit "${LIMIT}")
fi

for RESEARCH_QUESTION in "${RESEARCH_QUESTIONS[@]}"; do
    ENRICH_ARGS+=(--research-question "${RESEARCH_QUESTION}")
done

printf '%q ' python -m enrichment.cli "${ENRICH_ARGS[@]}" > "${RUN_DIR}/command.txt"
printf '\n' >> "${RUN_DIR}/command.txt"
cat "${RUN_DIR}/command.txt"

python -m enrichment.cli "${ENRICH_ARGS[@]}"

echo "Enrichment complete"
echo "Output directory: ${RUN_DIR}"
date
