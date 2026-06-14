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
  --self-consistency-aggregation scaffold \
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
  --self-consistency-aggregation scaffold \
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
  --refine-stop-parser json \
  --refine-history-format text \
  --force-think-prefix
```

Use `--teacher-backend dry-run` for a smoke test that does not load any model.

## HTML Inputs

By default, an HTML file is split into one record per participant section using `h2` headings such as `P1`, `P2`, and so on. Demographic tables and role-labelled dialogue turns are preserved in record metadata.

To treat an entire HTML file as one record:

```bash
--html-split-mode whole
```

If another HTML file needs CSS selectors:

```bash
--html-split-mode css --html-record-selector ".interview" --html-text-selector ".transcript" --html-id-attr id
```

`beautifulsoup4` is a core dependency because participant splitting is part of this phase.

## Outputs

Each output directory contains:

- `run_manifest.json`: exact command arguments, environment, generation options, and teacher backend metadata.
- `events.jsonl`: every model call with rendered prompt, raw response, generation options, timing, and strategy step.
- `enriched_records.jsonl`: one record per input item with full self-consistency samples or self-refine trace. Self-consistency scaffold runs intentionally leave `selected_output` as `null`.
- `failures.jsonl`: per-record failures if `--continue-on-error` is enabled.

## Prompt Variables

Templates can use:

- `{record_id}`
- `{input_text}`
- `{record_json}`
- `{metadata_FIELDNAME}` for structured metadata fields
- `{current_answer}`, `{feedback}`, and `{refinement_history}` inside Self-Refine feedback/revision prompts

Extra variables can be injected from the command line:

```bash
--prompt-var project_phase=enrichment --prompt-var teacher=deepseek_r1_distill
```
