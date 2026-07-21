#!/bin/bash
set -eo pipefail
set +u

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate dpo

echo "Active environment: ${CONDA_DEFAULT_ENV}"
echo "Python: $(which python)"
python --version

cd /iridisfs/home/kjl1a21/Direct-Preference-Optimization-of-LLMs-for-Critical-Thinking

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - <<'PY'
import json
import transformers
import tokenizers
from transformers import AutoTokenizer
from enrichment.teachers import (
    normalize_decoded_text,
    tokenizer_round_trip_diagnostic,
)

model_path = (
    "/iridisfs/scratch/kjl1a21/DPO/models/teacher/"
    "deepseek-ai__DeepSeek-R1-Distill-Llama-70B"
)
probe = "alpha beta\ngamma delta"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=False,
    local_files_only=True,
)
encoded = tokenizer(probe, add_special_tokens=False)
input_ids = encoded["input_ids"]
raw = tokenizer.decode(
    input_ids,
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False,
)
normalized = normalize_decoded_text(raw)

print("transformers_version:", transformers.__version__)
print("tokenizers_version:", tokenizers.__version__)
print("tokenizer_class:", type(tokenizer).__name__)
print("is_fast:", tokenizer.is_fast)
print("input_ids:", input_ids)
print("tokens:", tokenizer.convert_ids_to_tokens(input_ids))
print("raw_decoded_repr:", repr(raw))
print("raw_codepoints:", [f"U+{ord(c):04X}" for c in raw])
print("normalized_repr:", repr(normalized.text))
print(
    "diagnostic:",
    json.dumps(tokenizer_round_trip_diagnostic(tokenizer), indent=2),
)
print("backend_decoder:", getattr(tokenizer.backend_tokenizer, "decoder", None))
PY

sha256sum \
  /iridisfs/scratch/kjl1a21/DPO/models/teacher/deepseek-ai__DeepSeek-R1-Distill-Llama-70B/tokenizer.json \
  /iridisfs/scratch/kjl1a21/DPO/models/teacher/deepseek-ai__DeepSeek-R1-Distill-Llama-70B/tokenizer_config.json