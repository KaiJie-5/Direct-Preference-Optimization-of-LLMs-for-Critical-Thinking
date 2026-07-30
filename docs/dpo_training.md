# DPO training

The `dpo-train` command trains a local causal language model with TRL's
`DPOTrainer` using one of the two conversational preference datasets produced
by `dpo-build-preferences`.

The checked-in SmolLM3, Qwen3-4B-Instruct-2507,
Llama-3.2-3B-Instruct, Ministral-3-3B-Instruct-2512, and
Phi-4-mini-instruct configurations use BF16 training on one H200. Model, data,
split, chat-template, optimizer, and checkpoint settings are configuration
values shared by the same Python workflow.

The Qwen checkpoint was downloaded locally to:

```text
X:\DPO\models\student\Qwen__Qwen3-4B-Instruct-2507
```

It is Hugging Face revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
The cluster configuration uses the corresponding path:

```text
/iridisfs/scratch/kjl1a21/DPO/models/student/Qwen__Qwen3-4B-Instruct-2507
```

The Llama checkpoint was downloaded locally to:

```text
X:\DPO\models\student\meta-llama__Llama-3.2-3B-Instruct
```

It is Hugging Face revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`.
The cluster configuration uses:

```text
/iridisfs/scratch/kjl1a21/DPO/models/student/meta-llama__Llama-3.2-3B-Instruct
```

The Ministral checkpoint was downloaded locally to:

```text
X:\DPO\models\student\mistralai__Ministral-3-3B-Instruct-2512
```

It is Hugging Face revision
`b35d4dfe56c142746f54dbd64f579faab2744308`. The cluster path is:

```text
/iridisfs/scratch/kjl1a21/DPO/models/student/mistralai__Ministral-3-3B-Instruct-2512
```

The Phi checkpoint was downloaded locally to:

```text
X:\DPO\models\student\microsoft__Phi-4-mini-instruct
```

It is Hugging Face revision
`cfbefacb99257ffa30c83adab238a50856ac3083`. The cluster path is:

```text
/iridisfs/scratch/kjl1a21/DPO/models/student/microsoft__Phi-4-mini-instruct
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

Ministral additionally requires its native tokenizer for strict prompt-token
verification. Install the dedicated dependency group before submitting its
array tasks:

```bash
conda activate dpo
python -m pip install -e ".[training-ministral]"
```

This accepts Transformers versions from 5.14.1 up to, but not including, 6
and installs `mistral-common>=1.8.6`. The audited environment uses
Transformers 5.14.1.

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

Llama-3.2-3B-Instruct also uses `system_message: null`, so no custom system
instruction or `/no_think` is inserted. Its native template still emits a
system header containing its December 2023 knowledge cutoff and the current
date. The prompt ends with
`<|start_header_id|>assistant<|end_header_id|>\n\n`; chosen and rejected
completions preserve their source text and end with `<|eot_id|>`.

Ministral uses its official `SYSTEM_PROMPT.txt` with the placeholders resolved
to `today=2026-07-29` and `yesterday=2026-07-28`. The fully resolved value is
stored in the configuration, and `native_date_metadata` is false because these
dates are explicit rather than generated dynamically. It does not receive
`/no_think`. Its text-only prompt is:

```text
<s>[SYSTEM_PROMPT]{resolved official system prompt}[/SYSTEM_PROMPT][INST]{original user text}[/INST]
```

Each completion is the unchanged source assistant text followed by `</s>`.
For every real prompt, chosen sequence, and rejected sequence, preflight
requires identical token IDs from direct Hugging Face template tokenization,
rendered-string tokenization, and `MistralCommonBackend`. The string renderer
is explicitly loaded as `TokenizersBackend` from `tokenizer.json` and must
load exactly the checkpoint's standalone `chat_template.jinja`. This avoids
Transformers 5 automatically selecting the native backend, whose
`tokenize=false` output is not used as DPO training text.

Native verification is mode-aware: prompt-only sequences use
`MistralCommonBackend(mode="test")`, while complete chosen and rejected
sequences use `mode="finetuning"`. This is necessary because mistral-common's
test mode expects a generation request ending in a user message, whereas its
finetuning mode requires an assistant message at the end. The run records the
renderer, template source and hash, both native modes and backend classes,
verification count, dependency versions, and aggregate token-ID hashes.

Phi uses `system_message: null`, `native_date_metadata: false`, and the
integrated Transformers implementation with `trust_remote_code=false`. It
does not receive `/no_think`. Its prompt is:

```text
<|user|>{original user text}<|end|><|assistant|>
```

Chosen and rejected completions preserve the source text and end with
`<|end|><|endoftext|>`.

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

For Llama, use:

```bash
dpo-train \
  --config configs/dpo_training_llama_3_2_3b_instruct.json \
  --dataset-version category_evidence \
  --preflight-only
```

The profile reports prompt, chosen-sequence, rejected-sequence, and maximum
sequence token lengths for each source dataset and overall, including minimum,
50th, 90th, 95th, 99th percentile, and maximum.

`max_length` is `null`; training never truncates an interview or completion.
The command fails before training if any rendered sequence exceeds the selected
model configuration's limit: 65,536 tokens for SmolLM3, 131,072 tokens for
Llama-3.2-3B-Instruct and Phi-4-mini-instruct, or 262,144 tokens for
Qwen3-4B-Instruct-2507 and Ministral-3-3B-Instruct-2512. Ministral's limit is
read from `text_config.max_position_embeddings`; the other models use the
top-level value. This deliberately uses the model limit rather than a
tokenizer's larger advertised limit.

## Training

The selected recipe is standard sigmoid DPO:

- BF16 policy with all selected language parameters trainable and a resident
  frozen reference model;
- beta 0.1 and learning rate `5e-7`;
- cosine schedule with 10% warmup;
- one epoch, device batch size 1, and gradient accumulation 8;
- gradient checkpointing, fused AdamW, and gradient clipping at 1.0;
- natural source frequencies and no external experiment tracker.

Ministral's checkpoint contains fine-grained FP8 language projection weights.
The model is loaded through
`FineGrainedFP8Config(dequantize=True)` so training uses BF16. The full
multimodal checkpoint remains resident, but `vision_tower` and
`multi_modal_projector` are frozen and the runner asserts that only the
language model and language-model head remain trainable. Phi and all earlier
models continue using the standard loader without this profile.

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

Use the dedicated Llama launcher for Llama-3.2-3B-Instruct:

```bash
sbatch --export=ALL,DATASET_VERSION=category_evidence \
  submit_job_dpo_training_llama_3_2_3b_instruct.slurm

sbatch --export=ALL,DATASET_VERSION=question_only \
  submit_job_dpo_training_llama_3_2_3b_instruct.slurm
```

Use the dedicated Ministral launcher for the two corrected
Ministral-3-3B-Instruct-2512 experiments. Its checked-in `%1` concurrency
limit runs one task at a time:

| Task ID | Dataset |
| ---: | --- |
| 0 | category-evidence |
| 1 | question-only |

Submit both fresh training runs:

```bash
sbatch submit_job_dpo_training_ministral_3_3b_instruct_2512.slurm
```

Run validation, rendering, three-way token verification, and token profiling
for both tasks without loading model weights:

```bash
sbatch --export=ALL,PREFLIGHT_ONLY=true \
  submit_job_dpo_training_ministral_3_3b_instruct_2512.slurm
```

Submit one dataset only:

```bash
# category-evidence
sbatch --array=0 \
  submit_job_dpo_training_ministral_3_3b_instruct_2512.slurm

# question-only
sbatch --array=1 \
  submit_job_dpo_training_ministral_3_3b_instruct_2512.slurm
```

Resume a future interrupted run only when it has a valid training checkpoint,
and select the matching single task:

```bash
sbatch --array=0 \
  --export=ALL,RESUME_RUN_DIR=/iridisfs/scratch/kjl1a21/DPO/models/student/dpo_runs/existing_ministral_run \
  submit_job_dpo_training_ministral_3_3b_instruct_2512.slurm
```

Do not resume either failed run from array job `1335887`. Those runs stopped
at tokenizer initialization and contain no rendered snapshots, token profile,
or training checkpoint; the corrected launcher must create fresh run
directories. `PREFLIGHT_ONLY=true` cannot be combined with resume, and
array-wide resume is rejected because one run directory cannot identify two
independent experiments.

The earlier combined Ministral/Phi launcher is retained as reproducibility
history for the already completed Phi experiments. Its mapping was:

| Task ID | Model | Dataset |
| ---: | --- | --- |
| 0 | Ministral-3-3B-Instruct-2512 | category-evidence |
| 1 | Ministral-3-3B-Instruct-2512 | question-only |
| 2 | Phi-4-mini-instruct | category-evidence |
| 3 | Phi-4-mini-instruct | question-only |

Historical combined submission:

```bash
sbatch submit_job_dpo_training_ministral_phi_array.slurm
```

Historical combined preflight:

```bash
sbatch --export=ALL,PREFLIGHT_ONLY=true \
  submit_job_dpo_training_ministral_phi_array.slurm
```

Historical single-task example for Phi category-evidence:

```bash
sbatch --array=2 submit_job_dpo_training_ministral_phi_array.slurm
```

The combined launcher is no longer the recommended way to start new
Ministral jobs.

To perform only validation, splitting, rendering, and token profiling in the
same Slurm environment:

```bash
sbatch --export=ALL,DATASET_VERSION=category_evidence,PREFLIGHT_ONLY=true \
  submit_job_dpo_training.slurm
```

Replace the script name with
`submit_job_dpo_training_qwen3_4b_instruct_2507.slurm` to run the same optional
preflight for Qwen, or
`submit_job_dpo_training_llama_3_2_3b_instruct.slurm` for Llama. A full job
already performs input validation, split verification, template rendering,
token profiling, and context-limit checks before it loads model weights.

An interrupted training run resumes only when its directory is named
explicitly:

```bash
sbatch \
  --export=ALL,DATASET_VERSION=category_evidence,RESUME_RUN_DIR=/iridisfs/scratch/kjl1a21/DPO/models/student/dpo_runs/existing_run \
  submit_job_dpo_training.slurm
```

Resume a Qwen or Llama run with the same variables and its dedicated script.

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
