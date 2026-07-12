#!/bin/bash
#SBATCH --job-name=dpo_reflect_q
#SBATCH --partition=quad_h200
#SBATCH --account=ecs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-12:00:00
#SBATCH --output=dpo_reflect_q_%j.out
#SBATCH --error=dpo_reflect_q_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kjl1a21@soton.ac.uk

set -euo pipefail

CONDA_ENV="dpo"
PROJECT_DIR="/iridisfs/home/kjl1a21/Direct-Preference-Optimization-of-LLMs-for-Critical-Thinking"
CONFIG_PATH="${PROJECT_DIR}/configs/reflective_questions_enrichment.json"
# Set at submission time to continue an existing run, for example:
# sbatch --export=ALL,RESUME_RUN_DIR=/path/to/existing/run submit_job_reflective_questions_enrichment.slurm
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Required config does not exist: ${CONFIG_PATH}" >&2
    exit 2
fi

source ~/.bashrc
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

echo "Starting stage-two reflective-question enrichment"
echo "Config: ${CONFIG_PATH}"
echo "Resume run: ${RESUME_RUN_DIR:-new run}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
date

ARGS=(--config "${CONFIG_PATH}")
if [[ -n "${RESUME_RUN_DIR}" ]]; then
    ARGS+=(--resume "${RESUME_RUN_DIR}")
fi

python -m reflective_enrichment.cli "${ARGS[@]}"

echo "Stage-two reflective-question enrichment complete"
date
