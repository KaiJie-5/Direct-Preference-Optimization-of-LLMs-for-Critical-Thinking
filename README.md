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

Strict single-pass runs can be resumed in their original output directory:

```bash
dpo-enrich [the original arguments] \
  --output-dir /path/to/existing_run \
  --resume /path/to/existing_run
```

Resume validates the full input/configuration fingerprint and every saved
checkpoint before loading model weights. Only strictly valid successes are
skipped; failed, missing, or interrupted records are retried. Add
`--resume-validate-only` for a read-only preflight. Resume is intentionally not
available for the historical self-consistency or self-refine layouts.

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

## Stage-two reflective-question enrichment

Generate four reflective questions per segment from the top-ranked code in each
debate category:

```bash
dpo-reflective-enrich --config configs/reflective_questions_enrichment.json
```

Resume an interrupted scheduler job in its existing run directory:

```bash
dpo-reflective-enrich \
  --config configs/reflective_questions_enrichment.json \
  --resume /path/to/reflective_questions_enrichment/existing_run
```

Resume validates the ranking-to-sample mapping, selected code payloads, full
interview context, prompt, execution config, and saved segment checkpoints
before loading the teacher. Valid successful segments are skipped; failed or
interrupted segments are regenerated.

## DPO preference-pair construction

After reflective-question enrichment, construct the evidence-rich and
question-only conversational DPO datasets with:

```bash
dpo-build-preferences --config configs/dpo_preference_pairs.json
```

The constructor uses strict successful traces, reports failed and missing
records as skips, and writes line-aligned training and audit JSONL files under a
timestamped scratch run directory. See `docs/enrichment_phase.md` for the
formats, validation policy, and expected record totals.

To create the UKDA-to-unseen-domains experiment without resampling any rejected
response from the completed mixed-domain run, use:

```bash
dpo-build-domain-holdout --config configs/dpo_domain_holdout.json
```

This writes explicit train files containing all 6,304 accepted UKDA records
and explicit test files containing all 103 energy and 117 sexual-health
records. Copy the printed run directory into the training submission:

```bash
sbatch \
  --export=ALL,DPO_INPUT_RUN_DIR=/iridisfs/scratch/kjl1a21/DPO/data/dpo_preference_pairs/ukda4688_train_energy_sexual_health_test_TIMESTAMP \
  submit_job_dpo_training_domain_holdout_array.slurm
```

The `0-9%1` array runs category-evidence and question-only training for
SmolLM3, Qwen, Llama, Ministral, and Phi with at most one active task. Slurm
does not guarantee numeric task order.

## DPO training

Install the dedicated training dependencies into the existing `dpo`
environment:

```bash
python -m pip install -e ".[training]"
```

Run tokenizer-only validation and length profiling with:

```bash
dpo-train \
  --config configs/dpo_training_smollm3_3b.json \
  --dataset-version category_evidence \
  --preflight-only
```

The SmolLM3 evidence-rich and question-only experiments are submitted as
separate one-H200 jobs:

```bash
sbatch --export=ALL,DATASET_VERSION=category_evidence \
  submit_job_dpo_training.slurm

sbatch --export=ALL,DATASET_VERSION=question_only \
  submit_job_dpo_training.slurm
```

The workflow validates all source hashes and aligned audit rows, creates a
deterministic transcript-level 90/10 split, refuses truncation, evaluates
before and after DPO, and writes timestamped full-model runs under scratch
storage. See `docs/dpo_training.md` for the configuration, outputs, metrics,
and explicit resume command.
