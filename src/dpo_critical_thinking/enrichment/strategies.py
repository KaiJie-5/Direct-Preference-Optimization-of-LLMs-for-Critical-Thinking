from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from .data import DatasetRecord
from .logging import RunLogger
from .prompts import PromptTemplate
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
) -> dict[str, Any]:
    if aggregation != "scaffold":
        raise ValueError(
            "Self-consistency aggregation is scaffold-only until an open-text "
            "consistency metric is defined."
        )

    variables = {**record.to_prompt_vars(), **prompt_vars}
    rendered_prompt = prompt.render(variables)
    samples: list[dict[str, Any]] = []

    for sample_index in range(1, num_samples + 1):
        sample_options = _options_with_sample_seed(generation_options, sample_index)
        result = teacher.generate(rendered_prompt, sample_options)
        sample_payload = {
            "sample_index": sample_index,
            "generation_options": asdict(sample_options),
            "rendered_prompt": result.rendered_prompt,
            "output_text": result.text,
            "raw_response": result.raw,
            "elapsed_seconds": result.elapsed_seconds,
            "reasoning_path_status": "candidate_logged_without_aggregation",
        }
        samples.append(sample_payload)
        logger.event(
            {
                "event": "teacher_generation",
                "record_id": record.record_id,
                "strategy": "self_consistency",
                "aggregation": aggregation,
                **sample_payload,
            }
        )

    return {
        "record_id": record.record_id,
        "input_text": record.text,
        "metadata": record.metadata,
        "source": record.source,
        "strategy": "self_consistency",
        "prompt_path": str(prompt.path),
        "num_samples": num_samples,
        "aggregation": aggregation,
        "aggregation_status": "deferred_open_text_consistency_metric_required",
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
) -> dict[str, Any]:
    variables = {**record.to_prompt_vars(), **prompt_vars}
    initial_rendered = initial_prompt.render(variables)
    initial_result = teacher.generate(initial_rendered, generation_options)
    current_answer = initial_result.text
    trace: list[dict[str, Any]] = [
        {
            "step": "initial",
            "round": 0,
            "prompt_path": str(initial_prompt.path),
            "rendered_prompt": initial_result.rendered_prompt,
            "output_text": initial_result.text,
            "raw_response": initial_result.raw,
            "elapsed_seconds": initial_result.elapsed_seconds,
        }
    ]
    logger.event(
        {
            "event": "teacher_generation",
            "record_id": record.record_id,
            "strategy": "self_refine",
            **trace[-1],
        }
    )

    for round_index in range(1, refine_rounds + 1):
        refinement_history = _format_refinement_history(trace, history_format)
        round_vars = {
            **variables,
            "round_index": round_index,
            "current_answer": current_answer,
            "refinement_history": refinement_history,
        }
        critique_rendered = critique_prompt.render(round_vars)
        critique_result = teacher.generate(critique_rendered, generation_options)
        stop_decision = _parse_stop_decision(critique_result.text, stop_parser)
        trace.append(
            {
                "step": "feedback",
                "round": round_index,
                "prompt_path": str(critique_prompt.path),
                "rendered_prompt": critique_result.rendered_prompt,
                "output_text": critique_result.text,
                "raw_response": critique_result.raw,
                "elapsed_seconds": critique_result.elapsed_seconds,
                "parsed_stop_decision": stop_decision,
            }
        )
        logger.event(
            {
                "event": "teacher_generation",
                "record_id": record.record_id,
                "strategy": "self_refine",
                **trace[-1],
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
            **round_vars,
            "critique": critique_result.text,
            "feedback": critique_result.text,
            "refinement_history": _format_refinement_history(trace, history_format),
        }
        revision_rendered = revision_prompt.render(revision_vars)
        revision_result = teacher.generate(revision_rendered, generation_options)
        current_answer = revision_result.text
        trace.append(
            {
                "step": "revision",
                "round": round_index,
                "prompt_path": str(revision_prompt.path),
                "rendered_prompt": revision_result.rendered_prompt,
                "output_text": revision_result.text,
                "raw_response": revision_result.raw,
                "elapsed_seconds": revision_result.elapsed_seconds,
            }
        )
        logger.event(
            {
                "event": "teacher_generation",
                "record_id": record.record_id,
                "strategy": "self_refine",
                **trace[-1],
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
        "stop_parser": stop_parser,
        "history_format": history_format,
        "max_refine_rounds": refine_rounds,
        "completed_refinement_rounds": sum(
            1 for item in trace if item["step"] == "revision"
        ),
        "final_stop_decision": _latest_stop_decision(trace),
        "trace": trace,
    }


def _options_with_sample_seed(
    options: GenerationOptions, sample_index: int
) -> GenerationOptions:
    payload = asdict(options)
    if options.seed is not None:
        payload["seed"] = options.seed + sample_index - 1
    return GenerationOptions(**payload)


def _parse_stop_decision(text: str, parser: str) -> dict[str, Any]:
    if parser == "json":
        payload = _extract_json_object(text)
        if payload is None:
            return {
                "should_stop": False,
                "parser": parser,
                "parse_status": "json_not_found",
                "reason": "No JSON object could be parsed from feedback.",
            }
        return _stop_decision_from_mapping(payload, parser)
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


def _extract_json_object(text: str) -> dict[str, Any] | None:
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = [fenced_match.group(1)] if fenced_match else []

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


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
                "round": item["round"],
                "output_text": item["output_text"],
            }
            for item in trace
        ]
        return json.dumps(compact, ensure_ascii=False, indent=2)
    if history_format == "text":
        blocks = []
        for item in trace:
            label = f"Round {item['round']} {item['step']}"
            blocks.append(f"{label}:\n{item['output_text']}")
        return "\n\n".join(blocks)
    raise ValueError(f"Unsupported self-refine history format: {history_format}")


def _latest_stop_decision(trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in reversed(trace):
        if "parsed_stop_decision" in item:
            return item["parsed_stop_decision"]
    return None
