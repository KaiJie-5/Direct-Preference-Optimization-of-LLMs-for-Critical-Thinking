#!/bin/bash
#SBATCH --job-name=dpo_reflect_q
#SBATCH --partition=quad_h200
#SBATCH --account=ecs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=2-12:00:00
#SBATCH --output=dpo_reflect_q_%j.out
#SBATCH --error=dpo_reflect_q_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kjl1a21@soton.ac.uk

set -eo pipefail
set +u

CONDA_ENV="dpo"
PROJECT_DIR="/iridisfs/home/kjl1a21/Direct-Preference-Optimization-of-LLMs-for-Critical-Thinking"
CONFIG_PATH="${PROJECT_DIR}/configs/reflective_questions_enrichment.json"
DEFAULT_RESUME_RUN_DIR="/iridisfs/scratch/kjl1a21/DPO/data/reflective_questions_enrichment/deepseek_r1_ranked_codes_reflective_questions_20260713_144547"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-${DEFAULT_RESUME_RUN_DIR}}"
MIGRATE_LABEL_NORMALIZATION="${MIGRATE_LABEL_NORMALIZATION:-true}"

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

require_directory "${PROJECT_DIR}"
require_file "${CONFIG_PATH}"
require_directory "${RESUME_RUN_DIR}"
require_file "${RESUME_RUN_DIR}/run_manifest.json"
RESUME_RUN_DIR="$(readlink -f "${RESUME_RUN_DIR}")"

case "${MIGRATE_LABEL_NORMALIZATION}" in
    true|false)
        ;;
    *)
        echo "MIGRATE_LABEL_NORMALIZATION must be 'true' or 'false'." >&2
        exit 2
        ;;
esac

source ~/.bashrc
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

echo "Starting stage-two reflective-question enrichment"
echo "Config: ${CONFIG_PATH}"
echo "Resume run: ${RESUME_RUN_DIR}"
echo "Migrate label normalization: ${MIGRATE_LABEL_NORMALIZATION}"
if [[ "${MIGRATE_LABEL_NORMALIZATION}" == "true" ]]; then
    echo "Migration creates a timestamped sibling backup, reparses recoverable attempts,"
    echo "and regenerates records that remain unresolved."
fi
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
date

ARGS=(
    --config "${CONFIG_PATH}"
    --resume "${RESUME_RUN_DIR}"
)
if [[ "${MIGRATE_LABEL_NORMALIZATION}" == "true" ]]; then
    ARGS+=(--migrate-label-normalization)
fi

printf 'Command: '
printf '%q ' python -m reflective_enrichment.cli "${ARGS[@]}"
printf '\n'

python -m reflective_enrichment.cli "${ARGS[@]}"

echo "Stage-two reflective-question enrichment complete"
date
