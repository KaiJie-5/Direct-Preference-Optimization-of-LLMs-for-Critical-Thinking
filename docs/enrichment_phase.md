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
  --strategy self_consistency \
  --prompt-path prompts/enrichment/self_consistency_placeholder.txt \
  --research-question "How do participants discuss energy efficiency?" \
  --research-question "How do participants describe smart technology use?" \
  --teacher-backend dry-run \
  --context-scope full_interview \
  --self-consistency-samples 5 \
  --limit 1
```

Run with a local Hugging Face teacher model:

```bash
dpo-enrich \
  --segments-path data/segments_jsonl \
  --output-dir outputs/enrichment \
  --codebook-path data/codebooks/example_codes_v1.json \
  --strategy self_consistency \
  --prompt-path prompts/enrichment/self_consistency_placeholder.txt \
  --research-question "How do participants discuss energy efficiency?" \
  --research-question "How do participants describe smart technology use?" \
  --teacher-backend transformers \
  --context-scope full_interview \
  --model-path /path/to/models/teacher/deepseek-ai__DeepSeek-R1-Distill-Llama-70B \
  --temperature 0.6 \
  --max-new-tokens 32768 \
  --self-consistency-samples 5 \
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
response turn are included. The existing self-consistency prompt is aligned at
runtime so its JSON contract reports `analysis_context_scope: "turn_window"`;
the checked-in full-interview prompt text and historical modes are unchanged.

For UKDA 4688, `submit_job_enrichment_self_consistency_ukda4688.slurm` uses the
20/20 window, five self-consistency samples, and the existing DeepSeek teacher.
Submit it only after setting `CODEBOOK_PATH` and `RESEARCH_QUESTIONS_FILE`; the
latter is a UTF-8 file containing one research question per non-comment line.

New Self-Consistency and Self-Refine generations use
`segment_enrichment_sample_v2`. Historical v1 outputs remain readable. Each
sample is generated exactly once. Missing or malformed JSON, schema violations,
and missing closed `<think>...</think>` blocks are retained with
`final_parse_status: "warning"` and detailed `validation_warnings`; they do not
fail the record. Teacher, filesystem, configuration, and other operational
exceptions still fail the record and make the final batch exit nonzero.

Outputs are grouped by interview:

```text
outputs/enrichment/
  INT01_self_consistency/
    run_manifest.json
    events.jsonl
    segments/
      INT01_SEG001.json
      INT01_SEG002.json
    failures.jsonl
```

Self-consistency currently validates and logs 5 samples per segment. Aggregation is intentionally marked as `not_implemented_yet`. Full rendered prompts and backend payloads are retained in `events.jsonl`. Each per-segment JSON records every generated sample and its single attempt, including raw output text, extracted reasoning and JSON text, model-parsed JSON, canonicalized JSON, corrections, status, and validation warnings. Bulky backend payloads remain only in `events.jsonl`.

## Prompt Variables

Templates can use:

- `{record_id}`
- `{input_text}`
- `{analysis_context}` (selected by `--context-scope`)
- `{context_scope}`, `{context_turns_before}`, and `{context_turns_after}` in
  `turn_window` runs
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
