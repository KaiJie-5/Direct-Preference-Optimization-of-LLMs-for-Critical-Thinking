# Enrichment Phase

This code covers only the first phase: teacher-model dataset enrichment with configurable prompting strategies.

It does not implement multi-agent debate, downstream evaluation, DPO training, or model comparison dashboards.

## Main Command

Install the package from the repository root:

```bash
python -m pip install -e .
```

Install optional dependencies only when needed:

```bash
python -m pip install -e ".[html]"
python -m pip install -e ".[transformers]"
```

Smoke-test the pipeline without loading a model:

```bash
python -m dpo_critical_thinking.enrichment.cli \
  --input-path /path/to/transcripts-energy.html \
  --input-format html \
  --output-dir outputs/enrichment/dry_run \
  --strategy self_consistency \
  --prompt-path prompts/enrichment/self_consistency_placeholder.txt \
  --teacher-backend dry-run \
  --self-consistency-samples 2 \
  --limit 1
```

Run self-consistency with a local Hugging Face teacher model:

```bash
python -m dpo_critical_thinking.enrichment.cli \
  --input-path /path/to/transcripts-energy.html \
  --input-format html \
  --output-dir outputs/enrichment/self_consistency_deepseek \
  --strategy self_consistency \
  --prompt-path prompts/enrichment/self_consistency_placeholder.txt \
  --teacher-backend transformers \
  --model-path /path/to/models/teacher/deepseek-ai__DeepSeek-R1-Distill-Llama-70B \
  --temperature 0.6 \
  --max-new-tokens 2048 \
  --self-consistency-samples 5 \
  --self-consistency-selection none \
  --force-think-prefix
```

Run self-refine:

```bash
python -m dpo_critical_thinking.enrichment.cli \
  --input-path /path/to/transcripts-energy.html \
  --input-format html \
  --output-dir outputs/enrichment/self_refine_deepseek \
  --strategy self_refine \
  --prompt-path prompts/enrichment/self_refine_initial_placeholder.txt \
  --refine-critique-prompt-path prompts/enrichment/self_refine_critique_placeholder.txt \
  --refine-revision-prompt-path prompts/enrichment/self_refine_revision_placeholder.txt \
  --teacher-backend transformers \
  --model-path /path/to/models/teacher/deepseek-ai__DeepSeek-R1-Distill-Llama-70B \
  --temperature 0.6 \
  --max-new-tokens 2048 \
  --refine-rounds 2 \
  --force-think-prefix
```

Use `--teacher-backend dry-run` for a smoke test that does not load any model.

## HTML Inputs

By default, an HTML file is treated as one record after stripping tags. If the transcript HTML has repeated interview blocks, pass CSS selectors:

```bash
--html-record-selector ".interview" --html-text-selector ".transcript" --html-id-attr id
```

Installing `beautifulsoup4` is required only when CSS selectors are used.

## Outputs

Each output directory contains:

- `run_manifest.json`: exact command arguments, environment, generation options, and teacher backend metadata.
- `events.jsonl`: every model call with rendered prompt, raw response, generation options, timing, and strategy step.
- `enriched_records.jsonl`: one record per input item with selected output plus full self-consistency samples or self-refine trace.
- `failures.jsonl`: per-record failures if `--continue-on-error` is enabled.

## Prompt Variables

Templates can use:

- `{record_id}`
- `{input_text}`
- `{record_json}`
- `{metadata_FIELDNAME}` for structured metadata fields

Extra variables can be injected from the command line:

```bash
--prompt-var project_phase=enrichment --prompt-var teacher=deepseek_r1_distill
```
