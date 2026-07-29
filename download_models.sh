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

# echo "Downloading Qwen3-4B-Instruct-2507 student model..."
# hf download Qwen/Qwen3-4B-Instruct-2507 \
#   --local-dir "$PROJECT_SCRATCH/models/student/Qwen__Qwen3-4B-Instruct-2507"

echo "Downloading Ministral-3-3B-Instruct-2512 student model..."
hf download mistralai/Ministral-3-3B-Instruct-2512 \
  --local-dir "$PROJECT_SCRATCH/models/student/mistralai__Ministral-3-3B-Instruct-2512"

# echo "Downloading Llama-3.2-3B-Instruct student model..."
# hf download meta-llama/Llama-3.2-3B-Instruct \
#   --local-dir "$PROJECT_SCRATCH/models/student/meta-llama__Llama-3.2-3B-Instruct"

# echo "Downloading SmolLM3-3B student model..."
# hf download HuggingFaceTB/SmolLM3-3B \
#   --local-dir "$PROJECT_SCRATCH/models/student/HuggingFaceTB__SmolLM3-3B"

# echo "Downloading DeepSeek-R1-Distill-Llama-70B teacher model..."
# hf download deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
#   --local-dir "$PROJECT_SCRATCH/models/teacher/deepseek-ai__DeepSeek-R1-Distill-Llama-70B"

# echo "Downloading Qwen2.5-32B-Instruct teacher model..."
# hf download Qwen/Qwen2.5-32B-Instruct \
#   --local-dir "$PROJECT_SCRATCH/models/teacher/Qwen__Qwen2.5-32B-Instruct"

# echo "Downloading Qwen2.5-72B-Instruct teacher/ranking-agent model..."
# hf download Qwen/Qwen2.5-72B-Instruct \
#   --local-dir "$PROJECT_SCRATCH/models/teacher/Qwen__Qwen2.5-72B-Instruct"

# echo "Downloading Llama-3.3-70B-Instruct teacher model..."
# hf download meta-llama/Llama-3.3-70B-Instruct \
#   --local-dir "$PROJECT_SCRATCH/models/teacher/Llama-3.3-70B-Instruct"

# echo "Downloading Qwen2.5-72B-Instruct-GPTQ-Int8 teacher/ranking-agent model..."
# hf download Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8 \
#   --local-dir "$PROJECT_SCRATCH/models/teacher/Qwen__Qwen2.5-72B-Instruct-GPTQ-Int8"

echo "Download completed."
echo "Models saved in:"
echo "$PROJECT_SCRATCH/models"
