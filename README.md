# Direct-Preference-Optimization-of-LLMs-for-Critical-Thinking

## Installation

This repository was developed and tested on a Linux-based HPC environment using Conda, Python 3.10, and NVIDIA GPUs. The project uses large open-source language models, so model files are stored on scratch storage and are not tracked by Git.

### Environment setup

Create and activate the Conda environment:

```bash
conda create -n dpo python=3.10 -y
conda activate dpo
```

Upgrade `pip` and install the Hugging Face command line tools:

```bash
python -m pip install -U pip
python -m pip install -U "huggingface_hub[cli]" hf_transfer
```

Check the Python version:

```bash
python --version
```

This project was tested with:

```text
Python 3.10
Conda
Hugging Face CLI
Linux-based HPC environment
NVIDIA GPU support
```

### Hugging Face access

Some model downloads may require a Hugging Face access token. Log in before running the download script:

```bash
hf auth login
```

If the download script uses a custom Hugging Face cache directory on scratch storage, also export the token:

```bash
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
```

Do not write the Hugging Face token directly inside the repository or commit it to GitHub.

### Download required models

The required models are downloaded using the provided bash script:

```bash
bash download_models.sh
```

The script downloads the models into scratch storage, for example:

```text
/iridisfs/scratch/kjl1a21/DPO/models/
```

Large files such as model weights, checkpoints, private datasets, and generated outputs should not be committed to GitHub.

After downloading, check that the model folders exist:

```bash
ls /iridisfs/scratch/kjl1a21/DPO/models/student
ls /iridisfs/scratch/kjl1a21/DPO/models/teacher
```

A successful setup should contain the student model and teacher model under the scratch `models/` directory.

## Enrichment phase

The enrichment phase code lives under `src/enrichment`.

It supports configurable teacher backends, prompt templates, HTML/JSONL/JSON/CSV/TXT inputs, self-consistency, self-refine, and detailed JSON/JSONL logs for manual review. See:

```text
docs/enrichment_phase.md
```

The preprocessing CLI also includes an isolated `rtf --profile ukda-4688`
workflow for the UKDA 4688 interview archive and enrichment supports centered
complete-turn context windows without changing the existing defaults.

## Multi-agent debate ranking

Start a debate ranking run with:

```bash
dpo-debate rank --config configs/multi_agent_debate_llama_qwen.json
```

If a job reaches its scheduler time limit, resume it in the original run
directory:

```bash
dpo-debate rank \
  --config configs/multi_agent_debate_llama_qwen.json \
  --resume /path/to/multi_agent_debate_rankings/existing_run_directory
```

Resume validates the supplied config and all saved trace files before loading
the models. Successful and failed saved review blocks are preserved, and only
missing blocks are generated. A block interrupted before its trace was saved
restarts from its first debate turn. Final JSONL, CSV, and failure outputs are
rebuilt from the complete set of checkpoints when the run finishes.
