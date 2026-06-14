from __future__ import annotations

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
    selection: str,
    logger: RunLogger,
) -> dict[str, Any]:
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
        }
        samples.append(sample_payload)
        logger.event(
            {
                "event": "teacher_generation",
                "record_id": record.record_id,
                "strategy": "self_consistency",
                **sample_payload,
            }
        )

    selected_output, selected_sample_index = _select_self_consistency(samples, selection)
    return {
        "record_id": record.record_id,
        "input_text": record.text,
        "metadata": record.metadata,
        "source": record.source,
        "strategy": "self_consistency",
        "prompt_path": str(prompt.path),
        "num_samples": num_samples,
        "selection": selection,
        "selected_sample_index": selected_sample_index,
        "selected_output": selected_output,
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
        round_vars = {
            **variables,
            "round_index": round_index,
            "current_answer": current_answer,
        }
        critique_rendered = critique_prompt.render(round_vars)
        critique_result = teacher.generate(critique_rendered, generation_options)
        trace.append(
            {
                "step": "critique",
                "round": round_index,
                "prompt_path": str(critique_prompt.path),
                "rendered_prompt": critique_result.rendered_prompt,
                "output_text": critique_result.text,
                "raw_response": critique_result.raw,
                "elapsed_seconds": critique_result.elapsed_seconds,
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

        revision_vars = {
            **round_vars,
            "critique": critique_result.text,
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
        "trace": trace,
    }


def _options_with_sample_seed(
    options: GenerationOptions, sample_index: int
) -> GenerationOptions:
    payload = asdict(options)
    if options.seed is not None:
        payload["seed"] = options.seed + sample_index - 1
    return GenerationOptions(**payload)


def _select_self_consistency(
    samples: list[dict[str, Any]], selection: str
) -> tuple[str | None, int | None]:
    if selection == "none":
        return None, None
    if selection == "first":
        return samples[0]["output_text"], samples[0]["sample_index"]
    if selection == "longest":
        chosen = max(samples, key=lambda sample: len(sample["output_text"]))
        return chosen["output_text"], chosen["sample_index"]
    raise ValueError(
        f"Unsupported self-consistency selection {selection!r}. "
        "Use one of: none, first, longest."
    )
