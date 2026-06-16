from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dpo_critical_thinking.preprocessing.codebook import load_codebook

from .data import DatasetRecord, group_records_by_interview, load_segment_records
from .logging import RunLogger
from .prompts import PromptTemplate, parse_prompt_vars
from .strategies import run_self_consistency, run_self_refine
from .teachers import DEFAULT_MAX_NEW_TOKENS, GenerationOptions, Teacher, build_teacher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enrich preprocessed segment-level JSONL records."
    )

    io_group = parser.add_argument_group("input/output")
    io_group.add_argument("--segments-path", required=True, type=Path)
    io_group.add_argument("--output-dir", required=True, type=Path)
    io_group.add_argument(
        "--codebook-path",
        type=Path,
        help=(
            "Codebook JSON to use for candidate example codes. "
            "Overrides any codebook embedded in legacy segment files."
        ),
    )
    io_group.add_argument("--limit", type=int)
    io_group.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    strategy_group = parser.add_argument_group("prompting strategy")
    strategy_group.add_argument(
        "--strategy",
        required=True,
        choices=["self_consistency", "self_refine"],
    )
    strategy_group.add_argument("--prompt-path", required=True, type=Path)
    strategy_group.add_argument(
        "--prompt-var",
        action="append",
        default=[],
        help="Extra template variable in KEY=VALUE form. May be repeated.",
    )
    strategy_group.add_argument(
        "--research-question",
        action="append",
        default=[],
        help=(
            "Research question text to inject into self-consistency prompts. "
            "May be repeated; the same questions apply to all records in this run."
        ),
    )
    strategy_group.add_argument("--json-retry-attempts", type=int, default=2)
    strategy_group.add_argument("--self-consistency-samples", type=int, default=5)
    strategy_group.add_argument(
        "--self-consistency-aggregation",
        choices=["scaffold"],
        default="scaffold",
    )
    strategy_group.add_argument("--refine-rounds", type=int, default=2)
    strategy_group.add_argument("--refine-critique-prompt-path", type=Path)
    strategy_group.add_argument("--refine-revision-prompt-path", type=Path)
    strategy_group.add_argument(
        "--refine-stop-parser",
        choices=["json", "text"],
        default="json",
    )
    strategy_group.add_argument(
        "--refine-history-format",
        choices=["text", "json"],
        default="text",
    )

    teacher_group = parser.add_argument_group("teacher backend")
    teacher_group.add_argument(
        "--teacher-backend",
        choices=["dry-run", "transformers", "openai-compatible"],
        default="dry-run",
    )
    teacher_group.add_argument("--model-path")
    teacher_group.add_argument("--model-name")
    teacher_group.add_argument("--torch-dtype", default="auto")
    teacher_group.add_argument("--device-map", default="auto")
    teacher_group.add_argument("--trust-remote-code", action="store_true")
    teacher_group.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    teacher_group.add_argument(
        "--force-think-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    teacher_group.add_argument("--think-prefix", default="<think>\n")
    teacher_group.add_argument("--api-base")
    teacher_group.add_argument("--api-key-env")
    teacher_group.add_argument("--timeout-seconds", type=float, default=600.0)

    generation_group = parser.add_argument_group("generation")
    generation_group.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    generation_group.add_argument("--temperature", type=float, default=0.6)
    generation_group.add_argument("--top-p", type=float, default=0.95)
    generation_group.add_argument("--top-k", type=int)
    generation_group.add_argument("--repetition-penalty", type=float)
    generation_group.add_argument("--seed", type=int)
    generation_group.add_argument("--stop", action="append", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    teacher = build_teacher(args)
    records = load_segment_records(args.segments_path, limit=args.limit)
    codebook = _load_runtime_codebook(args.codebook_path, records)
    prompt_vars = {
        **_research_question_prompt_vars(args.research_question),
        **parse_prompt_vars(args.prompt_var),
    }
    generation_options = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        stop=args.stop,
    )

    prompt = PromptTemplate(args.prompt_path)
    critique_prompt = None
    revision_prompt = None
    if args.strategy == "self_refine":
        if args.refine_critique_prompt_path is None or args.refine_revision_prompt_path is None:
            raise ValueError(
                "Self-refine requires --refine-critique-prompt-path and "
                "--refine-revision-prompt-path."
            )
        critique_prompt = PromptTemplate(args.refine_critique_prompt_path)
        revision_prompt = PromptTemplate(args.refine_revision_prompt_path)

    grouped_records = group_records_by_interview(records)
    batch_summary = {
        "event": "batch_completed",
        "segments_path": str(args.segments_path),
        "output_dir": str(args.output_dir),
        "strategy": args.strategy,
        "codebook": _codebook_summary(codebook),
        "interview_count": len(grouped_records),
        "record_count": len(records),
        "interviews": [],
    }

    for interview_id, interview_records in grouped_records.items():
        interview_output_dir = args.output_dir / f"{interview_id}_{args.strategy}"
        summary = _run_interview(
            args=args,
            teacher=teacher,
            codebook=codebook,
            prompt=prompt,
            critique_prompt=critique_prompt,
            revision_prompt=revision_prompt,
            prompt_vars=prompt_vars,
            generation_options=generation_options,
            interview_id=interview_id,
            records=interview_records,
            output_dir=interview_output_dir,
        )
        batch_summary["interviews"].append(summary)

    batch_summary["success_count"] = sum(
        item["success_count"] for item in batch_summary["interviews"]
    )
    batch_summary["failure_count"] = sum(
        item["failure_count"] for item in batch_summary["interviews"]
    )
    print(json.dumps(batch_summary, indent=2))
    return 0 if batch_summary["failure_count"] == 0 else 1


def _run_interview(
    *,
    args: argparse.Namespace,
    teacher: Teacher,
    codebook: dict[str, Any] | None,
    prompt: PromptTemplate,
    critique_prompt: PromptTemplate | None,
    revision_prompt: PromptTemplate | None,
    prompt_vars: dict[str, Any],
    generation_options: GenerationOptions,
    interview_id: str,
    records: list[DatasetRecord],
    output_dir: Path,
) -> dict[str, Any]:
    logger = RunLogger(output_dir)
    manifest = _manifest(
        args=args,
        interview_id=interview_id,
        record_count=len(records),
        teacher_metadata=teacher.metadata(),
        codebook=codebook,
        generation_options=generation_options,
        output_dir=output_dir,
    )
    logger.write_manifest(manifest)
    logger.event({"event": "run_started", "manifest": manifest})

    success_count = 0
    failure_count = 0
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        print(
            "Enriching "
            f"dataset={args.segments_path} "
            f"interview={interview_id} "
            f"segment={record.metadata['segment_id']} "
            f"record={record.record_id} "
            f"index={index}/{len(records)}",
            flush=True,
        )
        try:
            logger.event(
                {
                    "event": "record_started",
                    "record_index": index,
                    "record_count": len(records),
                    "record_id": record.record_id,
                    "interview_id": interview_id,
                    "segment_id": record.metadata["segment_id"],
                    "input_char_count": len(record.text),
                }
            )
            enriched = _run_strategy(
                args=args,
                record=record,
                teacher=teacher,
                codebook=codebook,
                prompt=prompt,
                critique_prompt=critique_prompt,
                revision_prompt=revision_prompt,
                prompt_vars=prompt_vars,
                generation_options=generation_options,
                logger=logger,
            )
            if args.strategy == "self_consistency":
                segment_path = logger.enriched_segment(
                    record.record_id,
                    _focused_self_consistency_payload(enriched),
                )
            else:
                logger.enriched_record(enriched)
                segment_path = None
            logger.event({"event": "record_completed", "record_id": record.record_id})
            completed_message = (
                f"Completed record={record.record_id} status=success"
            )
            if segment_path is not None:
                completed_message += f" output={segment_path}"
            print(completed_message, flush=True)
            success_count += 1
        except Exception as exc:
            failure_count += 1
            logger.failure(
                {
                    "event": "record_failed",
                    "record_id": record.record_id,
                    "interview_id": interview_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(
                f"Completed record={record.record_id} status=failed "
                f"error_type={type(exc).__name__}",
                flush=True,
            )
            if not args.continue_on_error:
                raise

    summary = {
        "event": "run_completed",
        "interview_id": interview_id,
        "success_count": success_count,
        "failure_count": failure_count,
        "elapsed_seconds": time.perf_counter() - started,
        "output_dir": str(output_dir),
    }
    logger.event(summary)
    return summary


def _run_strategy(
    *,
    args: argparse.Namespace,
    record: DatasetRecord,
    teacher: Teacher,
    codebook: dict[str, Any] | None,
    prompt: PromptTemplate,
    critique_prompt: PromptTemplate | None,
    revision_prompt: PromptTemplate | None,
    prompt_vars: dict[str, Any],
    generation_options: GenerationOptions,
    logger: RunLogger,
) -> dict[str, Any]:
    if args.strategy == "self_consistency":
        return run_self_consistency(
            record=record,
            teacher=teacher,
            prompt=prompt,
            prompt_vars=prompt_vars,
            codebook=codebook,
            generation_options=generation_options,
            num_samples=args.self_consistency_samples,
            aggregation=args.self_consistency_aggregation,
            json_retry_attempts=args.json_retry_attempts,
            logger=logger,
        )

    if args.strategy == "self_refine":
        if critique_prompt is None or revision_prompt is None:
            raise ValueError("Self-refine prompt templates are not loaded.")
        return run_self_refine(
            record=record,
            teacher=teacher,
            initial_prompt=prompt,
            critique_prompt=critique_prompt,
            revision_prompt=revision_prompt,
            prompt_vars=prompt_vars,
            codebook=codebook,
            generation_options=generation_options,
            refine_rounds=args.refine_rounds,
            stop_parser=args.refine_stop_parser,
            history_format=args.refine_history_format,
            json_retry_attempts=args.json_retry_attempts,
            logger=logger,
        )

    raise ValueError(f"Unsupported strategy: {args.strategy}")


def _manifest(
    *,
    args: argparse.Namespace,
    interview_id: str,
    record_count: int,
    teacher_metadata: dict[str, Any],
    codebook: dict[str, Any] | None,
    generation_options: GenerationOptions,
    output_dir: Path,
) -> dict[str, Any]:
    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "interview_id": interview_id,
        "record_count": record_count,
        "output_dir": str(output_dir),
        "args": args_dict,
        "teacher": teacher_metadata,
        "codebook": _codebook_summary(codebook),
        "generation_options": asdict(generation_options),
    }


def _load_runtime_codebook(
    codebook_path: Path | None, records: list[DatasetRecord]
) -> dict[str, Any] | None:
    if codebook_path is not None:
        return load_codebook(codebook_path)
    if all("candidate_example_codes" in record.metadata for record in records):
        return None
    raise ValueError(
        "--codebook-path is required for segment files that do not contain "
        "legacy embedded candidate_example_codes."
    )


def _codebook_summary(codebook: dict[str, Any] | None) -> dict[str, Any]:
    if codebook is None:
        return {"source": "legacy_segment_metadata"}
    return {
        "source": "runtime_codebook",
        "codebook_id": codebook.get("codebook_id"),
        "codebook_version": codebook.get("codebook_version"),
        "code_count": len(codebook.get("codes", [])),
    }


def _research_question_prompt_vars(questions: list[str]) -> dict[str, str]:
    cleaned = [question.strip() for question in questions if question.strip()]
    if cleaned:
        text = "\n".join(
            f"{index}. {question}" for index, question in enumerate(cleaned, start=1)
        )
    else:
        text = "No explicit research questions were supplied for this run."
    return {"research_questions": text}


def _focused_self_consistency_payload(enriched: dict[str, Any]) -> dict[str, Any]:
    metadata = enriched["metadata"]
    return {
        "record_id": enriched["record_id"],
        "interview_id": metadata.get("interview_id"),
        "segment_id": metadata.get("segment_id"),
        "input_text": enriched["input_text"],
        "metadata": metadata,
        "source": enriched["source"],
        "strategy": enriched["strategy"],
        "prompt_path": enriched["prompt_path"],
        "num_samples": enriched["num_samples"],
        "aggregation": enriched["aggregation"],
        "aggregation_status": enriched["aggregation_status"],
        "selected_sample_index": enriched["selected_sample_index"],
        "selected_output": enriched["selected_output"],
        "samples": [
            {
                "sample_index": sample["sample_index"],
                "final_parse_status": sample["final_parse_status"],
                "validation_errors": sample["validation_errors"],
                "reasoning_text": sample["reasoning_text"],
                "parsed_output": sample["parsed_output"],
            }
            for sample in enriched["samples"]
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
