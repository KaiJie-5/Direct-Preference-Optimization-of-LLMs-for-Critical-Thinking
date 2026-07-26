# DPO training

The `dpo-train` command trains a local causal language model with TRL's
`DPOTrainer` using one of the two conversational preference datasets produced
by `dpo-build-preferences`.

The checked-in SmolLM3 and Qwen3-4B-Instruct-2507 configurations use
full-parameter BF16 training on one H200. Model, data, split, chat-template,
optimizer, and checkpoint settings are configuration values shared by the
same Python workflow.

The Qwen checkpoint was downloaded locally to:

```text
X:\DPO\models\student\Qwen__Qwen3-4B-Instruct-2507
```

It is Hugging Face revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
The cluster configuration uses the corresponding path:

```text
/iridisfs/scratch/kjl1a21/DPO/models/student/Qwen__Qwen3-4B-Instruct-2507
```

## Environment

Activate the existing environment and install the training dependency group
once:

```bash
conda activate dpo
python -m pip install -e ".[training]"
```

The Slurm job validates the installed versions but deliberately does not
install or upgrade packages inside a GPU allocation.

## Data validation and split

The configured input run is:

```text
/iridisfs/scratch/kjl1a21/DPO/data/dpo_preference_pairs/reflective_question_dpo_pairs_20260725_110033
```

Before model weights are loaded, the command checks:

- the source run is complete;
- evidence-rich, question-only, and audit JSONL files are line-aligned;
- every row has exactly `prompt`, `chosen`, and `rejected`;
- row hashes match the audit and file hashes match the source manifest;
- pair, line, record, and transcript identities are valid;
- each source record contributes four category pairs;
- required model/tokenizer files and their checksums are present.

Seed 42 creates a deterministic 90/10 split grouped by dataset and complete
transcript. The selected test transcripts are optimized toward 10% of source
records separately for energy, sexual health, and UKDA. The expected result is:

| Split | Source records | Preference pairs |
| --- | ---: | ---: |
| Train | 5,871 | 23,484 |
| Test | 653 | 2,612 |

No interview or overlapping context from a transcript can cross the split.
The exact transcript, record, pair, and line identities are saved in a shared
split manifest and copied into every training run.

## Token preflight

SmolLM3's `/no_think` system message is inserted in memory and its native
current-date metadata remains enabled. Qwen3-4B-Instruct-2507 uses
`system_message: null` and `native_date_metadata: false`, so the original user
message stays first. The Qwen tokenizer's native ChatML template renders the
prompt through `<|im_start|>assistant\n`; each chosen or rejected completion
preserves its source text and ends with `<|im_end|>\n`.

The original preference JSONL files are never modified. Each new run renders
once and saves immutable, checksummed train/test snapshots. A resumed run
reuses those snapshots.

Run a CPU/tokenizer-only validation if desired:

```bash
dpo-train \
  --config configs/dpo_training_smollm3_3b.json \
  --dataset-version category_evidence \
  --preflight-only
```

For Qwen, select its dedicated configuration:

```bash
dpo-train \
  --config configs/dpo_training_qwen3_4b_instruct_2507.json \
  --dataset-version category_evidence \
  --preflight-only
```

The profile reports prompt, chosen-sequence, rejected-sequence, and maximum
sequence token lengths for each source dataset and overall, including minimum,
50th, 90th, 95th, 99th percentile, and maximum.

`max_length` is `null`; training never truncates an interview or completion.
The command fails before training if any rendered sequence exceeds the selected
model configuration's limit: 65,536 tokens for SmolLM3 or 262,144 tokens for
Qwen3-4B-Instruct-2507. This deliberately uses the model limit rather than a
tokenizer's larger advertised limit.

## Training

The selected recipe is standard sigmoid DPO:

- full-parameter BF16 policy and resident frozen reference model;
- beta 0.1 and learning rate `5e-7`;
- cosine schedule with 10% warmup;
- one epoch, device batch size 1, and gradient accumulation 8;
- gradient checkpointing, fused AdamW, and gradient clipping at 1.0;
- natural source frequencies and no external experiment tracker.

The trainer evaluates its built-in DPO metrics before the first update and
after the epoch. It saves checkpoints every 250 optimizer steps and retains the
latest two. The final directory includes the full model and tokenizer,
train/test rendered snapshots, token and split manifests, metrics, Trainer
history, dependency/hardware information, and checksums.

## Slurm submissions

Submit the two dataset versions for SmolLM3 as separate one-H200 jobs:

```bash
sbatch --export=ALL,DATASET_VERSION=category_evidence \
  submit_job_dpo_training.slurm

sbatch --export=ALL,DATASET_VERSION=question_only \
  submit_job_dpo_training.slurm
```

Use the dedicated Qwen launcher for Qwen3-4B-Instruct-2507:

```bash
sbatch --export=ALL,DATASET_VERSION=category_evidence \
  submit_job_dpo_training_qwen3_4b_instruct_2507.slurm

sbatch --export=ALL,DATASET_VERSION=question_only \
  submit_job_dpo_training_qwen3_4b_instruct_2507.slurm
```

To perform only validation, splitting, rendering, and token profiling in the
same Slurm environment:

```bash
sbatch --export=ALL,DATASET_VERSION=category_evidence,PREFLIGHT_ONLY=true \
  submit_job_dpo_training.slurm
```

Replace the script name with
`submit_job_dpo_training_qwen3_4b_instruct_2507.slurm` to run the same optional
preflight for Qwen. A full job already performs input validation, split
verification, template rendering, token profiling, and context-limit checks
before it loads model weights.

An interrupted training run resumes only when its directory is named
explicitly:

```bash
sbatch \
  --export=ALL,DATASET_VERSION=category_evidence,RESUME_RUN_DIR=/iridisfs/scratch/kjl1a21/DPO/models/student/dpo_runs/existing_run \
  submit_job_dpo_training.slurm
```

Resume a Qwen run with the same variables and the dedicated Qwen script.

Resume recomputes the model, source-data, configuration, template, and split
fingerprints, validates the rendered snapshots, and selects the latest numeric
`checkpoint-*` directory. It refuses completed or mismatched runs.

## Metric interpretation

TRL records loss, chosen/rejected logits, chosen/rejected log-probabilities,
implicit rewards, reward margins, reward accuracy, token accuracy, learning
rate, and gradient statistics. Since the initial policy and reference are
identical, the pre-training reference-normalized rewards are ties; the
post-training reward margin and reward accuracy show how the policy changed
relative to the frozen initial model.
