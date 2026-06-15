from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .data import DatasetRecord
from .logging import RunLogger
from .prompts import PromptTemplate
from .schema import (
    build_json_repair_prompt,
    parse_json_object,
    split_response_sections,
    validate_segment_enrichment_sample,
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
    json_retry_attempts: int,
    logger: RunLogger,
    codebook: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if aggregation != "scaffold":
        raise ValueError(
            "Self-consistency aggregation is not implemented yet. "
            "Use --self-consistency-aggregation scaffold."
        )

    variables = {**record.to_prompt_vars(codebook), **prompt_vars}
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
            json_retry_attempts=json_retry_attempts,
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
    json_retry_attempts: int,
    logger: RunLogger,
    codebook: dict[str, Any] | None = None,
) -> dict[str, Any]:
    variables = {**record.to_prompt_vars(codebook), **prompt_vars}
    expected_codebook_version = _expected_codebook_version(record, codebook)
    initial_rendered = initial_prompt.render(variables)
    initial_sample = _generate_validated_sample(
        record=record,
        teacher=teacher,
        prompt=initial_rendered,
        generation_options=generation_options,
        expected_codebook_version=expected_codebook_version,
        json_retry_attempts=json_retry_attempts,
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
        stop_decision = _parse_stop_decision(critique_result.text, stop_parser)
        feedback_payload = {
            "step": "feedback",
            "round": round_index,
            "prompt_path": str(critique_prompt.path),
            "rendered_prompt": critique_result.rendered_prompt,
            "output_text": critique_result.text,
            "raw_response": critique_result.raw,
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
            json_retry_attempts=json_retry_attempts,
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
        "json_retry_attempts": json_retry_attempts,
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
    json_retry_attempts: int,
    logger: RunLogger,
    strategy: str,
    sample_index: int,
    step: str = "sample",
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    current_prompt = prompt
    final_output = ""
    final_parsed: dict[str, Any] | None = None
    final_errors: list[str] = []
    final_sections = split_response_sections("")

    for attempt_index in range(1, json_retry_attempts + 2):
        attempt_options = _options_with_attempt_seed(generation_options, attempt_index)
        result = teacher.generate(current_prompt, attempt_options)
        response_sections = split_response_sections(result.text)
        parsed, parse_error = parse_json_object(result.text)
        validation_errors = validate_segment_enrichment_sample(
            parsed,
            record,
            expected_codebook_version=expected_codebook_version,
        )
        if parse_error and parsed is None:
            validation_errors = [parse_error, *validation_errors]
        parse_status = "valid" if not validation_errors else "invalid"
        attempt_payload = {
            "attempt_index": attempt_index,
            "generation_options": asdict(attempt_options),
            "rendered_prompt": result.rendered_prompt,
            "raw_output_text": result.text,
            **response_sections,
            "parsed_output": parsed,
            "parse_status": parse_status,
            "validation_errors": validation_errors,
            "raw_response": result.raw,
            "elapsed_seconds": result.elapsed_seconds,
        }
        attempts.append(attempt_payload)
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

        final_output = result.text
        final_parsed = parsed
        final_errors = validation_errors
        final_sections = response_sections
        if not validation_errors:
            break
        current_prompt = build_json_repair_prompt(
            original_prompt=prompt,
            invalid_output=result.text,
            errors=validation_errors,
        )

    return {
        "sample_index": sample_index,
        "step": step,
        "attempt_count": len(attempts),
        "final_parse_status": "valid" if not final_errors else "invalid",
        "validation_errors": final_errors,
        "output_text": final_output,
        **final_sections,
        "parsed_output": final_parsed,
        "attempts": attempts,
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


def _options_with_attempt_seed(
    options: GenerationOptions, attempt_index: int
) -> GenerationOptions:
    payload = asdict(options)
    if options.seed is not None:
        payload["seed"] = options.seed + ((attempt_index - 1) * 1000)
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
