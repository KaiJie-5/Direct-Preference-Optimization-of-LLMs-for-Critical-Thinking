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
  --codebook-path data/codebooks/example_codes_v1.json \
  --overwrite
```

For a directory where each HTML file is already one interview, pass the directory to `--input-path`.

Preprocessing writes:

- `data/raw_html/INT01.html`
- `data/segments_jsonl/INT01_segments.jsonl`
- `data/preprocessing_manifest.json`

Each JSONL line is one participant-turn segment with previous/next context, participant characteristics extracted from the interview metadata table, and all candidate example codes.

## Enrich Segments

Smoke test without loading a model:

```bash
dpo-enrich \
  --segments-path data/segments_jsonl \
  --output-dir outputs/enrichment \
  --strategy self_consistency \
  --prompt-path prompts/enrichment/self_consistency_placeholder.txt \
  --teacher-backend dry-run \
  --self-consistency-samples 5 \
  --json-retry-attempts 2 \
  --limit 1
```

Run with a local Hugging Face teacher model:

```bash
dpo-enrich \
  --segments-path data/segments_jsonl \
  --output-dir outputs/enrichment \
  --strategy self_consistency \
  --prompt-path prompts/enrichment/self_consistency_placeholder.txt \
  --teacher-backend transformers \
  --model-path /path/to/models/teacher/deepseek-ai__DeepSeek-R1-Distill-Llama-70B \
  --temperature 0.6 \
  --max-new-tokens 32768 \
  --self-consistency-samples 5 \
  --json-retry-attempts 2 \
  --force-think-prefix
```

The default `--max-new-tokens` is `32768`, matching the DeepSeek-R1 README's high generation-length setting. The Transformers backend still checks the model context window separately and clamps the effective generation budget when the prompt plus requested output would exceed that context.

Outputs are grouped by interview:

```text
outputs/enrichment/
  INT01_self_consistency/
    run_manifest.json
    events.jsonl
    enriched_records.jsonl
    failures.jsonl
```

Self-consistency currently validates and logs 5 samples per segment. Aggregation is intentionally marked as `not_implemented_yet`. Each teacher sample preserves the raw output and separately logs `reasoning_text`, `reasoning_block`, `json_text`, and `reasoning_parse_status` when the model returns a `<think>...</think>` block before the JSON.

## Prompt Variables

Templates can use:

- `{record_id}`
- `{input_text}`
- `{segment_json}`
- `{candidate_example_codes_json}`
- `{current_answer}`, `{feedback}`, and `{refinement_history}` inside Self-Refine prompts

Extra variables can be injected from the command line:

```bash
--prompt-var project_phase=enrichment --prompt-var teacher=deepseek_r1_distill
```
