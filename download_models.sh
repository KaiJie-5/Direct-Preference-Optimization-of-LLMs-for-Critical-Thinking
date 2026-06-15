#!/bin/bash
set -e

PROJECT_SCRATCH="/iridisfs/scratch/kjl1a21/DPO/"

export HF_HOME="$PROJECT_SCRATCH/hf_cache"
export HF_HUB_CACHE="$PROJECT_SCRATCH/hf_cache/hub"
export HF_DATASETS_CACHE="$PROJECT_SCRATCH/hf_cache/datasets"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$PROJECT_SCRATCH/models/student"
mkdir -p "$PROJECT_SCRATCH/models/teacher"
mkdir -p "$PROJECT_SCRATCH/hf_cache"
mkdir -p "$PROJECT_SCRATCH/logs/downloads"

echo "Downloading SmolLM3-3B student model..."
hf download HuggingFaceTB/SmolLM3-3B \
  --local-dir "$PROJECT_SCRATCH/models/student/HuggingFaceTB__SmolLM3-3B"

echo "Downloading DeepSeek-R1-Distill-Llama-70B teacher model..."
hf download deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --local-dir "$PROJECT_SCRATCH/models/teacher/deepseek-ai__DeepSeek-R1-Distill-Llama-70B"

echo "Download completed."
echo "Models saved in:"
echo "$PROJECT_SCRATCH/models"