# Segment-Level Enrichment Workflow

This phase is split into two steps:

1. Preprocess raw data into auditable HTML and segment-level JSONL.
2. Enrich each segment independently with a teacher model.

The teacher model should normally consume JSONL segment records, not full HTML interviews.

## Install

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[transformers]"
```

## Convert The Codebook

```bash
dpo-preprocess codebook \
  --input-xlsx "/path/to/ExampleCodes.xlsx" \
  --output-path data/codebooks/example_codes_v1.json \
  --codebook-id example_codes \
  --codebook-version v1 \
  --overwrite
```

The converter supports the current workbook layouts:

- `Contestable Camera Cars`: `Code`, `Quotes`, `Example Questions`.
- `Braun and Clarke`: `Quote`, `Codes`.

## Preprocess HTML

For one large HTML file containing multiple interviews:

```bash
dpo-preprocess html \
  --input-path /path/to/transcripts-energy.html \
  --raw-html-dir data/raw_html \
  --segments-dir data/segments_jsonl \
  --manifest-path data/preprocessing_manifest.json \
  --overwrite
```

For a directory where each HTML file is already one interview, pass the directory to `--input-path`.

Preprocessing writes:

- `data/raw_html/INT01.html`
- `data/segments_jsonl/INT01_segments.jsonl`
- `data/preprocessing_manifest.json`

Each JSONL line is one participant-turn segment with previous/next context,
participant characteristics extracted from the interview metadata table, and an
`interview_turns` array containing every interviewer and participant turn in
source order. Codebooks are selected later during enrichment so the same
segments can be reused with different codebook versions.

## Preprocess UKDA 4688 RTF Interviews

UKDA Study 4688 uses an archive-specific RTF profile. Activate the project
environment and install the project so `striprtf==0.0.32` is installed inside
that environment:

```bash
conda activate dpo
python -m pip install -e ".[dev]"
```

Run strict preprocessing on the HPC archive:

```bash
dpo-preprocess rtf \
  --profile ukda-4688 \
  --input-path /iridisfs/scratch/kjl1a21/DPO/interview_datasets/UKDA-4688-rtf \
  --output-dir /iridisfs/scratch/kjl1a21/DPO/data/UKDA-4688-rtf-preprocessed \
  --strict-inventory
```

The profile validates all 85 documented transcripts, preserves exact extracted
text under `source_text`, writes an auditable normalized HTML rendering, and
creates analytical question-led adult-response exchange records under
`segments_jsonl`. The target-selection policy is
`ukda-4688-analytical-evidence-v1`.

Interviewer backchannels such as `Right.` remain in normalized interview
context but do not create false exchange boundaries. Participant turns that
contain only acknowledgements, interview noise, or clear-uncertainty markers
are retained in context but pruned from target evidence. Targets containing
only a short fragment, a question echo, or a clear cut-off are rejected. Short
complete claims remain eligible, while labels and quantities such as
`State schools.` and `10 months.` do not. Runs of two or more transcription
question marks still become explicit `[unclear]` markers; normal question
punctuation is unchanged.

Every candidate decision is recorded without repeated full-interview context
in `target_filter_audit.jsonl`. `preprocessing_qa.json` reports candidate,
retained, rejected, and pruned-turn counts plus rejection reasons. Retained
segment JSONL continues to embed `interview_turns` for enrichment compatibility,
so the generated segment directory remains intentionally storage-heavy.

### Review Remaining UKDA Enrichment Targets

Generate a broad, human-reviewable queue from the compact target audit:

```bash
dpo-preprocess target-review \
  --profile ukda-4688 \
  --audit-path /iridisfs/scratch/kjl1a21/DPO/data/UKDA-4688-rtf-preprocessed/target_filter_audit.jsonl \
  --output-path /iridisfs/scratch/kjl1a21/DPO/data/UKDA-4688-rtf-preprocessed/enrichment_exclusion_review.jsonl
```

Each row contains the exact target, its interviewer question, suggested review
reasons, and `"decision": "review"`. Edit every decision to either `keep` or
`exclude`. Concise factual targets are deliberately included in this broad
queue and can be marked `keep`; no review suggestion is automatically active.

After resolving every row, compile the strict runtime list:

```bash
dpo-preprocess approve-exclusions \
  --review-path /iridisfs/scratch/kjl1a21/DPO/data/UKDA-4688-rtf-preprocessed/enrichment_exclusion_review.jsonl \
  --audit-path /iridisfs/scratch/kjl1a21/DPO/data/UKDA-4688-rtf-preprocessed/target_filter_audit.jsonl \
  --output-path /iridisfs/scratch/kjl1a21/DPO/data/UKDA-4688-rtf-preprocessed/enrichment_exclusions.jsonl
```

Compilation fails for unresolved decisions or stale target text. The approved
file contains only exact `record_id`/`text` pairs. Review and approved-list
manifests record source hashes and decision counts. The rich review file cannot
be used as the runtime list because enrichment rejects extra fields.

The UKDA SLURM template requires this approved file and passes it through
`--exclude-records-path`. Direct enrichment commands can use the same option;
without it, record loading is unchanged for existing datasets. Exclusions are
applied before `--limit`, and run manifests record the list hash and skip count.

The source archive is never modified. Use `run_preprocessing_ukda4688.sh` for
the same configured HPC workflow. To replace an existing derived output after
this policy change, run:

```bash
OVERWRITE=true bash run_preprocessing_ukda4688.sh
```

The script loads Conda directly instead of sourcing the user `.bashrc`. Set
`CONDA_BASE` explicitly only when Conda is not discoverable from `PATH`,
`~/miniconda3`, or `~/anaconda3`.

## Enrich Segments

Smoke test without loading a model:

```bash
dpo-enrich \
  --segments-path data/segments_jsonl \
  --output-dir outputs/enrichment \
  --codebook-path data/codebooks/example_codes_v1.json \
  --strategy single_pass \
  --prompt-path prompts/enrichment/self_consistency_four_codes.txt \
  --research-question "How do participants discuss energy efficiency?" \
  --research-question "How do participants describe smart technology use?" \
  --teacher-backend dry-run \
  --context-scope full_interview \
  --limit 1
```

Run with a local Hugging Face teacher model:

```bash
dpo-enrich \
  --segments-path data/segments_jsonl \
  --output-dir outputs/enrichment \
  --codebook-path data/codebooks/example_codes_v1.json \
  --strategy single_pass \
  --prompt-path prompts/enrichment/self_consistency_four_codes.txt \
  --research-question "How do participants discuss energy efficiency?" \
  --research-question "How do participants describe smart technology use?" \
  --teacher-backend transformers \
  --context-scope full_interview \
  --model-path /path/to/models/teacher/deepseek-ai__DeepSeek-R1-Distill-Llama-70B \
  --temperature 0.6 \
  --max-new-tokens 32768 \
  --force-think-prefix
```

The default `--max-new-tokens` is `32768`, matching the DeepSeek-R1 README's high generation-length setting. The Transformers backend still checks the model context window separately and clamps the effective generation budget when the prompt plus requested output would exceed that context.

`--context-scope immediate` is the backward-compatible default and renders the
existing previous/next context into `{analysis_context}`. The
`full_interview` scope requires newly preprocessed JSONL, renders every ordered
turn, and marks the current participant turn as the target. Full-interview mode
also requires `{analysis_context}` in every prompt used by the selected
strategy, and validates this before loading the teacher model.

`--context-scope turn_window` renders complete normalized turns around the
target exchange. Its defaults are 20 turns before and 20 turns after; customize
them with `--context-turns-before` and `--context-turns-after`. Interview
metadata, the leading interviewer question, boundary notices, and every target
response turn are included. `{context_scope}` is rendered directly into the v3
prompt contract, so the saved scope always matches the runtime selection.

For UKDA 4688, `submit_job_enrichment_self_consistency_ukda4688.slurm` uses the
20/20 window, one strict single-pass generation, and the existing DeepSeek teacher.
The codebook path and two UKDA-4688 research questions are defined directly in
the Slurm script. Each question is passed as a repeated `--research-question`
argument.
This output contains the four code-quality examples directly and must not be sent
to debate ranking, which requires at least two candidates.

```bash
sbatch submit_job_enrichment_self_consistency_ukda4688.slurm
```

If the UKDA job is interrupted, resume the exact run folder instead of creating
a new timestamped run:

```bash
sbatch --export=ALL,RESUME_DIR=/iridisfs/scratch/kjl1a21/DPO/data/UKDA-4688-rtf-enriched/existing_run \
  submit_job_enrichment_self_consistency_ukda4688.slurm
```

The original `command.txt` is preserved and each resubmission writes a separate
`resume_command_<job>_<timestamp>.txt`. Logs continue to append. Before using a
GPU allocation, the same command can be checked with `--resume-validate-only`;
this validates inputs, configuration, prompt hashes, and checkpoints without
changing the run or loading model weights.

Native resume applies only to `single_pass`. A root `run_manifest.json` freezes
the complete input, prompt, codebook, teacher/generation configuration, and
record fingerprints. Strictly valid successful checkpoints are skipped. Failed
and missing checkpoints are retried with attempt history preserved. Malformed
expected checkpoint files are archived under `resume_invalid_checkpoints/`
before regeneration, while identity or fingerprint mismatches abort the resume.
Historical single-pass runs without the root manifest receive a validated legacy
migration: all available interview manifests, source records, checkpoints, and
rendered prompt hashes are checked before the current complete input fingerprint
is frozen.

New enrichment generations use `segment_enrichment_sample_v3`. Historical v1
and v2 outputs remain readable. In `single_pass`, missing or malformed JSON,
schema violations, and missing closed `<think>...</think>` blocks are retained
with `final_parse_status: "invalid"` and detailed `validation_errors`. The segment
is marked failed, processing continues, and the final batch exits nonzero when
any segment fails. Historical multi-sample strategies retain their warning-based
collection behavior.

Outputs are grouped by interview:

```text
outputs/enrichment/
  run_manifest.json
  INT01_single_pass/
    run_manifest.json
    events.jsonl
    segments/
      INT01_SEG001.json
      INT01_SEG002.json
    failures.jsonl
```

Single-pass enrichment validates and logs one sample per segment and selects it
only when strict validation succeeds. Full rendered prompts and backend payloads
are retained in `events.jsonl`. Each per-segment JSON records the generation's
raw output, extracted reasoning and JSON text, model-parsed JSON, canonicalized
JSON, corrections, status, and validation issues. Bulky backend payloads remain
only in `events.jsonl`.

## Prompt Variables

Templates can use:

- `{record_id}`
- `{input_text}`
- `{analysis_context}` (selected by `--context-scope`)
- `{context_scope}` for every run; `{context_turns_before}` and
  `{context_turns_after}` in `turn_window` runs
- `{segment_json}`
- `{interview_id}`, `{segment_id}`, `{speaker}`
- `{previous_context}`, `{next_context}`
- `{codebook_id}`, `{codebook_version}`
- `{candidate_example_codes_json}`
- `{research_questions}`
- `{current_answer}`, `{feedback}`, and `{refinement_history}` inside Self-Refine prompts

Extra variables can be injected from the command line:

```bash
--prompt-var project_phase=enrichment --prompt-var teacher=deepseek_r1_distill
```

Research questions can be injected separately and repeated:

```bash
--research-question "How do participants discuss energy efficiency?" \
--research-question "How do participants describe smart technology use?"
```

## Stage-Two Reflective Questions From Ranked Codes

After debate ranking, run the dedicated second enrichment stage:

```bash
dpo-reflective-enrich --config configs/reflective_questions_enrichment.json
```

For every segment, it resolves the first-ranked candidate in each of the four
code-quality categories back to its original enrichment sample and generates
all four reflective questions in one teacher call. The target segment remains
the evidence unit; the full interview is supplied only as interpretive context.

Outputs contain an auditable per-segment trace, a rebuilt
`reflective_questions.jsonl`, `failures.jsonl`, and a run manifest. To resume,
pass the existing run folder:

```bash
dpo-reflective-enrich \
  --config configs/reflective_questions_enrichment.json \
  --resume /path/to/existing/run
```

Only strictly validated successful checkpoints are skipped. Failed, invalid,
and interrupted segments are retried, while prior saved attempts remain in the
segment audit trace.
