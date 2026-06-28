from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from typing import Any

from .data import DatasetRecord
from .logging import RunLogger
from .prompts import PromptTemplate
from .schema import (
    SAMPLE_SCHEMA_VERSION,
    canonicalize_source_fields,
    parse_json_object,
    split_response_sections,
    validate_segment_enrichment_sample_result,
)
from .teachers import GenerationOptions, Teacher


def run_self_consistency(
    *,
    record: DatasetRecord,
    teacher: Teacher,
    prompt: PromptTemplate,
    prompt_vars: dict[str, Any],
    generation_options: GenerationOptions,
    num_samples: int,
    aggregation: str,
    logger: RunLogger,
    codebook: dict[str, Any] | None = None,
    context_scope: str = "immediate",
) -> dict[str, Any]:
    if aggregation != "scaffold":
        raise ValueError(
            "Self-consistency aggregation is not implemented yet. "
            "Use --self-consistency-aggregation scaffold."
        )

    variables = {
        **record.to_prompt_vars(codebook, context_scope=context_scope),
        **prompt_vars,
    }
    variables["analysis_context"] = record.analysis_context(context_scope)
    rendered_prompt = prompt.render(variables)
    expected_codebook_version = _expected_codebook_version(record, codebook)
    samples: list[dict[str, Any]] = []

    for sample_index in range(1, num_samples + 1):
        sample_options = _options_with_sample_seed(generation_options, sample_index)
        sample_payload = _generate_validated_sample(
            record=record,
            teacher=teacher,
            prompt=rendered_prompt,
            generation_options=sample_options,
            expected_codebook_version=expected_codebook_version,
            context_scope=context_scope,
            logger=logger,
            strategy="self_consistency",
            sample_index=sample_index,
        )
        samples.append(sample_payload)

    return {
        "record_id": record.record_id,
        "input_text": record.text,
        "metadata": record.metadata,
        "source": record.source,
        "strategy": "self_consistency",
        "context_scope": context_scope,
        "prompt_path": str(prompt.path),
        "num_samples": num_samples,
        "aggregation": aggregation,
        "aggregation_status": "not_implemented_yet",
        "selected_sample_index": None,
        "selected_output": None,
        "samples": samples,
    }


def run_self_refine(
    *,
    record: DatasetRecord,
    teacher: Teacher,
    initial_prompt: PromptTemplate,
    critique_prompt: PromptTemplate,
    revision_prompt: PromptTemplate,
    prompt_vars: dict[str, Any],
    generation_options: GenerationOptions,
    refine_rounds: int,
    stop_parser: str,
    history_format: str,
    logger: RunLogger,
    codebook: dict[str, Any] | None = None,
    context_scope: str = "immediate",
) -> dict[str, Any]:
    variables = {
        **record.to_prompt_vars(codebook, context_scope=context_scope),
        **prompt_vars,
    }
    variables["analysis_context"] = record.analysis_context(context_scope)
    expected_codebook_version = _expected_codebook_version(record, codebook)
    initial_rendered = initial_prompt.render(variables)
    initial_sample = _generate_validated_sample(
        record=record,
        teacher=teacher,
        prompt=initial_rendered,
        generation_options=generation_options,
        expected_codebook_version=expected_codebook_version,
        context_scope=context_scope,
        logger=logger,
        strategy="self_refine",
        sample_index=0,
        step="initial",
    )
    current_answer = initial_sample["output_text"]
    current_parsed = initial_sample["parsed_output"]
    trace: list[dict[str, Any]] = [
        {
            "step": "initial",
            "round": 0,
            "prompt_path": str(initial_prompt.path),
            **initial_sample,
        }
    ]

    for round_index in range(1, refine_rounds + 1):
        refinement_history = _format_refinement_history(trace, history_format)
        feedback_vars = {
            **variables,
            "round_index": round_index,
            "current_answer": current_answer,
            "current_answer_json": current_parsed,
            "refinement_history": refinement_history,
        }
        critique_rendered = critique_prompt.render(feedback_vars)
        critique_result = teacher.generate(critique_rendered, generation_options)
        prompt_reference = logger.prompt_snapshot(
            record_id=record.record_id,
            strategy="self_refine",
            step="feedback",
            sample_index=round_index,
            rendered_prompt=critique_result.rendered_prompt,
        )
        critique_raw = dict(critique_result.raw)
        raw_decoded_text = critique_raw.pop("raw_decoded_text", None)
        if isinstance(raw_decoded_text, str):
            critique_raw.update(
                logger.decode_artifact(
                    record_id=record.record_id,
                    strategy="self_refine",
                    step="feedback",
                    sample_index=round_index,
                    attempt_index=1,
                    raw_text=raw_decoded_text,
                )
            )
        stop_decision = _parse_stop_decision(critique_result.text, stop_parser)
        feedback_payload = {
            "step": "feedback",
            "round": round_index,
            "prompt_path": str(critique_prompt.path),
            **prompt_reference,
            "output_text": critique_result.text,
            "raw_response": critique_raw,
            "elapsed_seconds": critique_result.elapsed_seconds,
            "parsed_stop_decision": stop_decision,
        }
        trace.append(feedback_payload)
        logger.event(
            {
                "event": "teacher_generation",
                "record_id": record.record_id,
                "strategy": "self_refine",
                **feedback_payload,
            }
        )
        logger.event(
            {
                "event": "self_refine_stop_decision",
                "record_id": record.record_id,
                "strategy": "self_refine",
                "round": round_index,
                **stop_decision,
            }
        )

        if stop_decision["should_stop"]:
            break

        revision_vars = {
            **feedback_vars,
            "critique": critique_result.text,
            "feedback": critique_result.text,
            "refinement_history": _format_refinement_history(trace, history_format),
        }
        revision_rendered = revision_prompt.render(revision_vars)
        revision_sample = _generate_validated_sample(
            record=record,
            teacher=teacher,
            prompt=revision_rendered,
            generation_options=generation_options,
            expected_codebook_version=expected_codebook_version,
            context_scope=context_scope,
            logger=logger,
            strategy="self_refine",
            sample_index=round_index,
            step="revision",
        )
        current_answer = revision_sample["output_text"]
        current_parsed = revision_sample["parsed_output"]
        trace.append(
            {
                "step": "revision",
                "round": round_index,
                "prompt_path": str(revision_prompt.path),
                **revision_sample,
            }
        )

    return {
        "record_id": record.record_id,
        "input_text": record.text,
        "metadata": record.metadata,
        "source": record.source,
        "strategy": "self_refine",
        "context_scope": context_scope,
        "prompt_paths": {
            "initial": str(initial_prompt.path),
            "critique": str(critique_prompt.path),
            "revision": str(revision_prompt.path),
        },
        "refine_rounds": refine_rounds,
        "selected_output": current_answer,
        "selected_json": current_parsed,
        "stop_parser": stop_parser,
        "history_format": history_format,
        "max_refine_rounds": refine_rounds,
        "completed_refinement_rounds": sum(
            1 for item in trace if item["step"] == "revision"
        ),
        "final_stop_decision": _latest_stop_decision(trace),
        "trace": trace,
    }


def _generate_validated_sample(
    *,
    record: DatasetRecord,
    teacher: Teacher,
    prompt: str,
    generation_options: GenerationOptions,
    expected_codebook_version: str | None,
    context_scope: str,
    logger: RunLogger,
    strategy: str,
    sample_index: int,
    step: str = "sample",
) -> dict[str, Any]:
    attempt_index = 1
    result = teacher.generate(prompt, generation_options)
    response_sections = split_response_sections(result.text)
    model_parsed, parse_error = parse_json_object(result.text)
    parsed, canonical_corrections = canonicalize_source_fields(
        model_parsed,
        record,
        expected_codebook_version=expected_codebook_version,
        expected_context_scope=context_scope,
    )
    validation_result = validate_segment_enrichment_sample_result(
        parsed,
        record,
        expected_codebook_version=expected_codebook_version,
        expected_schema_version=SAMPLE_SCHEMA_VERSION,
        expected_context_scope=context_scope,
        strict_prompt_schema=True,
        allow_target_text_mismatch=False,
    )
    issues = list(validation_result.errors)
    if response_sections["reasoning_parse_status"] != "found_closed_think_block":
        issues.insert(
            0,
            "Response must contain a closed <think>...</think> reasoning block.",
        )
    if parse_error and parsed is None:
        issues.insert(0, parse_error)
    validation_warnings = [*issues, *validation_result.warnings]
    parse_status = "warning" if validation_warnings else "valid"
    prompt_reference = logger.prompt_snapshot(
        record_id=record.record_id,
        strategy=strategy,
        step=step,
        sample_index=sample_index,
        rendered_prompt=result.rendered_prompt,
    )
    current_prompt_hash = sha256(result.rendered_prompt.encode("utf-8")).hexdigest()
    raw_response = dict(result.raw)
    raw_decoded_text = raw_response.pop("raw_decoded_text", None)
    if isinstance(raw_decoded_text, str):
        raw_response.update(
            logger.decode_artifact(
                record_id=record.record_id,
                strategy=strategy,
                step=step,
                sample_index=sample_index,
                attempt_index=attempt_index,
                raw_text=raw_decoded_text,
            )
        )
    attempt_payload = {
        "attempt_index": attempt_index,
        "generation_options": asdict(generation_options),
        **prompt_reference,
        "attempt_prompt_sha256": current_prompt_hash,
        "is_repair_prompt": False,
        "raw_output_text": result.text,
        **response_sections,
        "model_parsed_output": model_parsed,
        "parsed_output": parsed,
        "canonical_corrections": canonical_corrections,
        "parse_status": parse_status,
        "validation_errors": [],
        "validation_warnings": validation_warnings,
        "raw_response": raw_response,
        "elapsed_seconds": result.elapsed_seconds,
    }
    logger.event(
        {
            "event": "teacher_generation",
            "record_id": record.record_id,
            "strategy": strategy,
            "step": step,
            "sample_index": sample_index,
            **attempt_payload,
        }
    )

    return {
        "sample_index": sample_index,
        "step": step,
        "attempt_count": 1,
        "final_parse_status": parse_status,
        "validation_errors": [],
        "validation_warnings": validation_warnings,
        "canonical_corrections": canonical_corrections,
        "output_text": result.text,
        **response_sections,
        "parsed_output": parsed,
        "attempts": [attempt_payload],
    }


def _expected_codebook_version(
    record: DatasetRecord, codebook: dict[str, Any] | None
) -> str | None:
    if codebook is not None:
        version = codebook.get("codebook_version")
    else:
        version = record.metadata.get("codebook_version")
    return str(version) if version is not None else None


def _options_with_sample_seed(
    options: GenerationOptions, sample_index: int
) -> GenerationOptions:
    payload = asdict(options)
    if options.seed is not None:
        payload["seed"] = options.seed + sample_index - 1
    return GenerationOptions(**payload)


def _parse_stop_decision(text: str, parser: str) -> dict[str, Any]:
    if parser == "json":
        parsed, parse_error = parse_json_object(text)
        if parsed is None:
            return {
                "should_stop": False,
                "parser": parser,
                "parse_status": "json_not_found",
                "reason": parse_error or "No JSON object could be parsed from feedback.",
            }
        return _stop_decision_from_mapping(parsed, parser)
    if parser == "text":
        normalized = text.lower()
        stop_patterns = [
            "no further refinement",
            "no refinement needed",
            "does not need refinement",
            "stop: true",
            "needs_refinement: false",
        ]
        should_stop = any(pattern in normalized for pattern in stop_patterns)
        return {
            "should_stop": should_stop,
            "parser": parser,
            "parse_status": "text_patterns_checked",
            "reason": "Matched stop phrase." if should_stop else "No stop phrase matched.",
        }
    raise ValueError(f"Unsupported self-refine stop parser: {parser}")


def _stop_decision_from_mapping(payload: dict[str, Any], parser: str) -> dict[str, Any]:
    if "stop" in payload:
        should_stop = _as_bool(payload["stop"])
        source_field = "stop"
    elif "needs_refinement" in payload:
        should_stop = not _as_bool(payload["needs_refinement"])
        source_field = "needs_refinement"
    elif "continue_refinement" in payload:
        should_stop = not _as_bool(payload["continue_refinement"])
        source_field = "continue_refinement"
    else:
        should_stop = False
        source_field = None

    return {
        "should_stop": should_stop,
        "parser": parser,
        "parse_status": "parsed",
        "source_field": source_field,
        "reason": str(payload.get("reason", payload.get("rationale", ""))),
        "parsed_feedback": payload,
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "stop"}
    return bool(value)


def _format_refinement_history(trace: list[dict[str, Any]], history_format: str) -> str:
    if history_format == "json":
        compact = [
            {
                "step": item["step"],
                "round": item.get("round"),
                "output_text": item.get("output_text"),
                "parsed_output": item.get("parsed_output"),
                "validation_errors": item.get("validation_errors", []),
            }
            for item in trace
        ]
        import json

        return json.dumps(compact, ensure_ascii=False, indent=2)
    if history_format == "text":
        blocks = []
        for item in trace:
            label = f"Round {item.get('round')} {item['step']}"
            blocks.append(f"{label}:\n{item.get('output_text', '')}")
        return "\n\n".join(blocks)
    raise ValueError(f"Unsupported self-refine history format: {history_format}")


def _latest_stop_decision(trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in reversed(trace):
        if "parsed_stop_decision" in item:
            return item["parsed_stop_decision"]
    return None
