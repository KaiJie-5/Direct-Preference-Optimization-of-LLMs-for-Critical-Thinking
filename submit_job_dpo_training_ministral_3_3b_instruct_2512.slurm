#!/bin/bash
#SBATCH --job-name=dpo_ministral_3b_2512
#SBATCH --partition=quad_h200
#SBATCH --account=ecs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=2-12:00:00
#SBATCH --array=0-1%1
#SBATCH --output=dpo_ministral_3b_2512_%A_%a.out
#SBATCH --error=dpo_ministral_3b_2512_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kjl1a21@soton.ac.uk

set -eo pipefail
set +u

CONDA_ENV="dpo"
PROJECT_DIR="/iridisfs/home/kjl1a21/Direct-Preference-Optimization-of-LLMs-for-Critical-Thinking"
CONFIG_PATH="${PROJECT_DIR}/configs/dpo_training_ministral_3_3b_instruct_2512.json"
MODEL_LABEL="Ministral-3-3B-Instruct-2512"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"

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

case "${SLURM_ARRAY_TASK_ID:-}" in
    0)
        DATASET_VERSION="category_evidence"
        ;;
    1)
        DATASET_VERSION="question_only"
        ;;
    *)
        echo "SLURM_ARRAY_TASK_ID must be 0 or 1." >&2
        exit 2
        ;;
esac

case "${PREFLIGHT_ONLY}" in
    true|false)
        ;;
    *)
        echo "PREFLIGHT_ONLY must be true or false." >&2
        exit 2
        ;;
esac

require_directory "${PROJECT_DIR}"
require_file "${CONFIG_PATH}"

if [[ -n "${RESUME_RUN_DIR}" ]]; then
    if [[ "${PREFLIGHT_ONLY}" == "true" ]]; then
        echo "PREFLIGHT_ONLY=true cannot be combined with RESUME_RUN_DIR." >&2
        exit 2
    fi
    if [[ "${SLURM_ARRAY_TASK_COUNT:-0}" != "1" ]]; then
        echo "Resume requires one task selected with sbatch --array=<task-id>." >&2
        exit 2
    fi
    require_directory "${RESUME_RUN_DIR}"
    require_file "${RESUME_RUN_DIR}/run_manifest.json"
    RESUME_RUN_DIR="$(readlink -f "${RESUME_RUN_DIR}")"
fi

source ~/.bashrc
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export CONFIG_PATH
export DATASET_VERSION

echo "Starting ${MODEL_LABEL} DPO training"
echo "SLURM job id: ${SLURM_JOB_ID:-unset}"
echo "SLURM array job id: ${SLURM_ARRAY_JOB_ID:-unset}"
echo "SLURM array task id: ${SLURM_ARRAY_TASK_ID:-unset}"
echo "Node list: ${SLURM_JOB_NODELIST:-unset}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Project directory: ${PROJECT_DIR}"
echo "Config path: ${CONFIG_PATH}"
echo "Dataset version: ${DATASET_VERSION}"
echo "Preflight only: ${PREFLIGHT_ONLY}"
echo "Resume run directory: ${RESUME_RUN_DIR:-new run}"
date

echo "Python: $(which python)"
python --version
python - <<'PY'
import importlib.metadata
import os
from pathlib import Path

import torch

from dpo_training.config import MINISTRAL_TRAINING_PROFILE, load_training_config
from dpo_training.runner import _validate_training_dependencies

config = load_training_config(Path(os.environ["CONFIG_PATH"]))
if config.model.training_profile != MINISTRAL_TRAINING_PROFILE:
    raise SystemExit(
        "The dedicated Ministral launcher requires the Ministral training profile."
    )
for package in [
    "torch",
    "transformers",
    "trl",
    "datasets",
    "accelerate",
    "mistral-common",
]:
    try:
        print(f"{package}: {importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(f"Required training package is not installed: {package}")
_validate_training_dependencies(config)
print("training profile:", config.model.training_profile)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("This job requires exactly one visible CUDA GPU.")
print("gpu 0:", torch.cuda.get_device_name(0))
print("bf16 supported:", torch.cuda.is_bf16_supported())
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The allocated GPU does not support BF16.")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

ARGS=(
    --config "${CONFIG_PATH}"
    --dataset-version "${DATASET_VERSION}"
)
if [[ "${PREFLIGHT_ONLY}" == "true" ]]; then
    ARGS+=(--preflight-only)
fi
if [[ -n "${RESUME_RUN_DIR}" ]]; then
    ARGS+=(--resume "${RESUME_RUN_DIR}")
fi

printf 'Command: '
printf '%q ' python -m dpo_training.cli "${ARGS[@]}"
printf '\n'

python -m dpo_training.cli "${ARGS[@]}"

echo "${MODEL_LABEL} ${DATASET_VERSION} DPO job complete"
date
