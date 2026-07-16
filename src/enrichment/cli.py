from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from preprocessing.codebook import load_codebook

from .data import (
    DatasetRecord,
    group_records_by_interview,
    load_segment_records_with_report,
)
from .logging import RunLogger
from .prompts import PromptTemplate, parse_prompt_vars
from .resume import (
    archive_corrupt_checkpoints,
    audit_resume,
    build_run_identity,
    checkpoint_metadata,
    mark_manifest_resumed,
    new_run_manifest,
    update_manifest_state,
    write_run_manifest,
)
from .schema import SAMPLE_SCHEMA_VERSION
from .strategies import run_self_consistency, run_self_refine, run_single_pass
from .teachers import (
    DEFAULT_MAX_NEW_TOKENS,
    GenerationOptions,
    Teacher,
    build_prompt_renderer,
    build_teacher,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enrich preprocessed segment-level JSONL records."
    )

    io_group = parser.add_argument_group("input/output")
    io_group.add_argument("--segments-path", required=True, type=Path)
    io_group.add_argument("--output-dir", required=True, type=Path)
    io_group.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIR",
        help="Resume a validated single-pass run in its existing output directory.",
    )
    io_group.add_argument(
        "--resume-validate-only",
        action="store_true",
        help="Validate a --resume run without changing files or loading model weights.",
    )
    io_group.add_argument(
        "--exclude-records-path",
        type=Path,
        help=(
            "Optional JSONL containing exact record_id/text pairs to skip before "
            "applying --limit."
        ),
    )
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
        "--context-scope",
        choices=["immediate", "full_interview", "turn_window"],
        default="immediate",
        help=(
            "Use immediate previous/next context, the complete ordered interview, "
            "or a centered complete-turn window. Full interview and turn-window "
            "modes require reprocessed segment JSONL and an {analysis_context} "
            "placeholder in every strategy prompt."
        ),
    )
    io_group.add_argument(
        "--context-turns-before",
        type=int,
        default=20,
        help="Complete normalized turns before the target in turn_window mode.",
    )
    io_group.add_argument(
        "--context-turns-after",
        type=int,
        default=20,
        help="Complete normalized turns after the target in turn_window mode.",
    )
    io_group.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    strategy_group = parser.add_argument_group("prompting strategy")
    strategy_group.add_argument(
        "--strategy",
        required=True,
        choices=["single_pass", "self_consistency", "self_refine"],
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
    _validate_resume_arguments(args)
    load_result = load_segment_records_with_report(
        args.segments_path,
        limit=args.limit,
        exclude_records_path=args.exclude_records_path,
    )
    records = load_result.records
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
    if args.strategy == "single_pass":
        _validate_single_pass_configuration(prompt, args.research_question)
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

    _validate_context_configuration(
        context_scope=args.context_scope,
        records=records,
        strategy=args.strategy,
        prompt=prompt,
        critique_prompt=critique_prompt,
        revision_prompt=revision_prompt,
        context_turns_before=args.context_turns_before,
        context_turns_after=args.context_turns_after,
    )

    checkpoints: dict[str, dict[str, Any]] = {}
    root_manifest: dict[str, Any] | None = None
    identity: dict[str, Any] | None = None
    is_resume = args.resume is not None
    if args.strategy == "single_pass":
        current_args = _jsonable_args(args)
        execution_config = {
            key: value
            for key, value in current_args.items()
            if key
            not in {
                "output_dir",
                "resume",
                "resume_validate_only",
                "continue_on_error",
            }
        }
        execution_config["exclusion_filter"] = {
            key: value
            for key, value in load_result.exclusion_summary.items()
            if key != "excluded_by_interview"
        }
        identity = build_run_identity(
            records=records,
            execution_config=execution_config,
            prompt=prompt,
            codebook=codebook,
        )
        if is_resume:
            prompt_renderer = build_prompt_renderer(args)
            audit = audit_resume(
                run_dir=args.resume,
                records=records,
                identity=identity,
                prompt=prompt,
                prompt_vars=prompt_vars,
                codebook=codebook,
                research_questions=args.research_question,
                context_scope=args.context_scope,
                context_turns_before=args.context_turns_before,
                context_turns_after=args.context_turns_after,
                generation_options=asdict(generation_options),
                current_args=current_args,
                exclusion_summary=load_result.exclusion_summary,
                model_prompt_renderer=prompt_renderer,
            )
            print(
                "Resume validation complete: "
                f"successful={audit.success_count} "
                f"failed={audit.failed_count} "
                f"missing={audit.missing_count} "
                f"corrupt={len(audit.corrupt)} "
                f"retry_or_missing={audit.retry_or_missing_count} "
                f"total={audit.total_count}",
                flush=True,
            )
            if args.resume_validate_only:
                return 0
            if audit.corrupt:
                archived = archive_corrupt_checkpoints(audit.run_dir, audit.corrupt)
                print(
                    f"Archived {len(audit.corrupt)} malformed checkpoint(s) and "
                    f"{len(archived) - len(audit.corrupt)} related artifact(s) "
                    "before retry.",
                    flush=True,
                )
            checkpoints = audit.checkpoints
            root_manifest = audit.manifest
            mark_manifest_resumed(
                root_manifest,
                legacy=audit.legacy_migration,
            )
            write_run_manifest(args.output_dir, root_manifest)
        else:
            if (args.output_dir / "run_manifest.json").exists():
                raise FileExistsError(
                    "Single-pass root run manifest already exists; use --resume to "
                    f"continue safely: {args.output_dir / 'run_manifest.json'}"
                )
            existing = sorted(args.output_dir.glob("*_single_pass/segments/*.json"))
            if existing:
                raise FileExistsError(
                    "Single-pass output checkpoints already exist; use --resume to "
                    f"continue safely: {existing[0]}"
                )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            root_manifest = new_run_manifest(
                identity=identity,
                output_dir=args.output_dir.resolve(),
                record_count=len(records),
            )
            write_run_manifest(args.output_dir, root_manifest)

    remaining = (
        len(records)
        if args.strategy != "single_pass"
        else sum(
            checkpoints.get(record.record_id, {}).get("status") != "success"
            for record in records
        )
    )
    teacher: Teacher | None = build_teacher(args) if remaining else None

    grouped_records = group_records_by_interview(records)
    batch_summary = {
        "event": "batch_completed",
        "segments_path": str(args.segments_path),
        "output_dir": str(args.output_dir),
        "strategy": args.strategy,
        "output_schema_version": SAMPLE_SCHEMA_VERSION,
        "codebook": _codebook_summary(codebook),
        "interview_count": len(grouped_records),
        "source_record_count": load_result.source_record_count,
        "record_count": len(records),
        "exclusion_filter": load_result.exclusion_summary,
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
            exclusion_summary=load_result.exclusion_summary,
            checkpoints=checkpoints,
            execution_fingerprint=(
                identity["execution_fingerprint"] if identity is not None else None
            ),
            is_resume=is_resume,
        )
        batch_summary["interviews"].append(summary)
        if root_manifest is not None:
            update_manifest_state(
                root_manifest,
                total_count=len(records),
                checkpoints=checkpoints,
                status="running",
            )
            write_run_manifest(args.output_dir, root_manifest)

    batch_summary["success_count"] = sum(
        item["success_count"] for item in batch_summary["interviews"]
    )
    batch_summary["failure_count"] = sum(
        item["failure_count"] for item in batch_summary["interviews"]
    )
    if root_manifest is not None:
        final_state = "complete" if batch_summary["failure_count"] == 0 else "incomplete"
        update_manifest_state(
            root_manifest,
            total_count=len(records),
            checkpoints=checkpoints,
            status=final_state,
        )
        write_run_manifest(args.output_dir, root_manifest)
    print(json.dumps(batch_summary, indent=2))
    return 0 if batch_summary["failure_count"] == 0 else 1


def _run_interview(
    *,
    args: argparse.Namespace,
    teacher: Teacher | None,
    codebook: dict[str, Any] | None,
    prompt: PromptTemplate,
    critique_prompt: PromptTemplate | None,
    revision_prompt: PromptTemplate | None,
    prompt_vars: dict[str, Any],
    generation_options: GenerationOptions,
    interview_id: str,
    records: list[DatasetRecord],
    output_dir: Path,
    exclusion_summary: dict[str, Any],
    checkpoints: dict[str, dict[str, Any]],
    execution_fingerprint: str | None,
    is_resume: bool,
) -> dict[str, Any]:
    logger = RunLogger(output_dir)
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        if teacher is None:
            raise RuntimeError("Teacher is required to start an unfinished interview.")
        manifest = _manifest(
            args=args,
            interview_id=interview_id,
            record_count=len(records),
            teacher_metadata=teacher.metadata(),
            codebook=codebook,
            generation_options=generation_options,
            output_dir=output_dir,
            exclusion_summary=exclusion_summary,
        )
        logger.write_manifest(manifest)
        logger.event({"event": "run_started", "manifest": manifest})
    elif is_resume:
        logger.event(
            {
                "event": "run_resumed",
                "interview_id": interview_id,
                "record_count": len(records),
            }
        )

    success_count = 0
    failure_count = 0
    operational_failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        saved = checkpoints.get(record.record_id)
        if args.strategy == "single_pass" and saved is not None:
            if saved.get("status") == "success":
                success_count += 1
                print(
                    f"Skipping validated checkpoint record={record.record_id} "
                    f"index={index}/{len(records)}",
                    flush=True,
                )
                continue
        if teacher is None:
            raise RuntimeError("Teacher was not loaded despite remaining records.")
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
                previous_attempts=_saved_attempts(saved),
            )
            if args.strategy == "self_consistency":
                segment_path = logger.enriched_segment(
                    record.record_id,
                    _focused_self_consistency_payload(enriched),
                )
            elif args.strategy == "single_pass":
                if execution_fingerprint is None:
                    raise RuntimeError("Single-pass execution fingerprint is missing.")
                focused = _focused_single_pass_payload(enriched)
                focused.update(
                    checkpoint_metadata(
                        execution_fingerprint=execution_fingerprint,
                        record=record,
                    )
                )
                segment_path = logger.enriched_segment(
                    record.record_id,
                    focused,
                )
                checkpoints[record.record_id] = focused
            else:
                logger.enriched_record(enriched)
                segment_path = None
            if args.strategy == "single_pass" and enriched["status"] == "failed":
                failure_count += 1
                validation_errors = enriched["samples"][0]["validation_errors"]
                logger.failure(
                    {
                        "event": "record_failed",
                        "record_id": record.record_id,
                        "interview_id": interview_id,
                        "error_type": "ValidationError",
                        "error": "Single-pass generation failed strict validation.",
                        "validation_errors": validation_errors,
                        "segment_path": str(segment_path),
                    }
                )
                print(
                    f"Completed record={record.record_id} status=failed "
                    f"error_type=ValidationError output={segment_path}",
                    flush=True,
                )
                continue
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
            failure_payload = {
                "event": "record_failed",
                "record_id": record.record_id,
                "interview_id": interview_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            operational_failures.append(failure_payload)
            logger.failure(failure_payload)
            print(
                f"Completed record={record.record_id} status=failed "
                f"error_type={type(exc).__name__}",
                flush=True,
            )
            if not args.continue_on_error:
                raise

    if args.strategy == "single_pass":
        operational_failure_ids = {
            payload["record_id"] for payload in operational_failures
        }
        current_failures = [
            {
                "event": "record_failed",
                "record_id": record.record_id,
                "interview_id": interview_id,
                "error_type": "ValidationError",
                "error": "Single-pass generation failed strict validation.",
                "validation_errors": checkpoints[record.record_id]["samples"][0][
                    "validation_errors"
                ],
                "segment_path": str(
                    output_dir / "segments" / f"{record.record_id}.json"
                ),
            }
            for record in records
            if checkpoints.get(record.record_id, {}).get("status") == "failed"
            and record.record_id not in operational_failure_ids
        ]
        logger.replace_failures([*current_failures, *operational_failures])

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
    previous_attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if args.strategy == "single_pass":
        return run_single_pass(
            record=record,
            teacher=teacher,
            prompt=prompt,
            prompt_vars=prompt_vars,
            codebook=codebook,
            research_questions=args.research_question,
            generation_options=generation_options,
            logger=logger,
            context_scope=args.context_scope,
            context_turns_before=args.context_turns_before,
            context_turns_after=args.context_turns_after,
            previous_attempts=previous_attempts,
        )

    if args.strategy == "self_consistency":
        return run_self_consistency(
            record=record,
            teacher=teacher,
            prompt=prompt,
            prompt_vars=prompt_vars,
            codebook=codebook,
            research_questions=args.research_question,
            generation_options=generation_options,
            num_samples=args.self_consistency_samples,
            aggregation=args.self_consistency_aggregation,
            logger=logger,
            context_scope=args.context_scope,
            context_turns_before=args.context_turns_before,
            context_turns_after=args.context_turns_after,
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
            research_questions=args.research_question,
            generation_options=generation_options,
            refine_rounds=args.refine_rounds,
            stop_parser=args.refine_stop_parser,
            history_format=args.refine_history_format,
            logger=logger,
            context_scope=args.context_scope,
            context_turns_before=args.context_turns_before,
            context_turns_after=args.context_turns_after,
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
    exclusion_summary: dict[str, Any],
) -> dict[str, Any]:
    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    if args.context_scope != "turn_window":
        args_dict.pop("context_turns_before", None)
        args_dict.pop("context_turns_after", None)
    manifest_exclusion_summary = {
        key: value
        for key, value in exclusion_summary.items()
        if key != "excluded_by_interview"
    }
    manifest_exclusion_summary["interview_excluded_count"] = exclusion_summary.get(
        "excluded_by_interview", {}
    ).get(interview_id, 0)
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "interview_id": interview_id,
        "record_count": record_count,
        "output_dir": str(output_dir),
        "args": args_dict,
        "exclusion_filter": manifest_exclusion_summary,
        "teacher": teacher_metadata,
        "codebook": _codebook_summary(codebook),
        "generation_options": asdict(generation_options),
        "output_schema_version": SAMPLE_SCHEMA_VERSION,
    }


def _validate_resume_arguments(args: argparse.Namespace) -> None:
    if args.resume_validate_only and args.resume is None:
        raise ValueError("--resume-validate-only requires --resume RUN_DIR.")
    if args.resume is None:
        return
    if args.strategy != "single_pass":
        raise ValueError("--resume is supported only with --strategy single_pass.")
    resume_dir = args.resume.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if resume_dir != output_dir:
        raise ValueError(
            "--output-dir and --resume must identify the same run directory: "
            f"output_dir={output_dir}, resume={resume_dir}."
        )
    args.resume = resume_dir
    args.output_dir = output_dir


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _saved_attempts(
    checkpoint: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if checkpoint is None or checkpoint.get("status") != "failed":
        return []
    samples = checkpoint.get("samples")
    if not isinstance(samples, list) or len(samples) != 1:
        return []
    attempts = samples[0].get("attempts")
    return list(attempts) if isinstance(attempts, list) else []


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
        text = json.dumps(cleaned, ensure_ascii=False, indent=2)
    else:
        text = "[]"
    return {"research_questions": text}


def _validate_single_pass_configuration(
    prompt: PromptTemplate, research_questions: list[str]
) -> None:
    if not any(question.strip() for question in research_questions):
        raise ValueError("single_pass requires at least one --research-question.")
    required_variables = {
        "record_id",
        "interview_id",
        "segment_id",
        "speaker",
        "codebook_version",
        "research_questions",
        "analysis_context",
        "context_scope",
        "input_text",
        "candidate_example_codes_json",
    }
    missing = sorted(
        variable
        for variable in required_variables
        if not prompt.uses_variable(variable)
    )
    if missing:
        raise ValueError(
            "single_pass prompt is missing required template variables: "
            f"{missing}."
        )


def _focused_single_pass_payload(enriched: dict[str, Any]) -> dict[str, Any]:
    metadata = enriched["metadata"]
    return {
        "record_id": enriched["record_id"],
        "interview_id": metadata.get("interview_id"),
        "segment_id": metadata.get("segment_id"),
        "input_text": enriched["input_text"],
        "metadata": metadata,
        "source": enriched["source"],
        "strategy": enriched["strategy"],
        "status": enriched["status"],
        "context_scope": enriched["context_scope"],
        **(
            {
                "context_turns_before": enriched["context_turns_before"],
                "context_turns_after": enriched["context_turns_after"],
            }
            if "context_turns_before" in enriched
            else {}
        ),
        "prompt_path": enriched["prompt_path"],
        "num_samples": enriched["num_samples"],
        "selected_sample_index": enriched["selected_sample_index"],
        "selected_output": enriched["selected_output"],
        "selected_json": enriched["selected_json"],
        "samples": _focused_samples(enriched["samples"]),
    }


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
        "context_scope": enriched["context_scope"],
        **(
            {
                "context_turns_before": enriched["context_turns_before"],
                "context_turns_after": enriched["context_turns_after"],
            }
            if "context_turns_before" in enriched
            else {}
        ),
        "prompt_path": enriched["prompt_path"],
        "num_samples": enriched["num_samples"],
        "aggregation": enriched["aggregation"],
        "aggregation_status": enriched["aggregation_status"],
        "selected_sample_index": enriched["selected_sample_index"],
        "selected_output": enriched["selected_output"],
        "samples": _focused_samples(enriched["samples"]),
    }


def _focused_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sample_index": sample["sample_index"],
            "attempt_count": sample["attempt_count"],
            "final_parse_status": sample["final_parse_status"],
            "validation_errors": sample["validation_errors"],
            "validation_warnings": sample.get("validation_warnings", []),
            "canonical_corrections": sample.get("canonical_corrections", []),
            "output_text": sample["output_text"],
            "reasoning_block": sample["reasoning_block"],
            "reasoning_text": sample["reasoning_text"],
            "json_text": sample["json_text"],
            "reasoning_parse_status": sample["reasoning_parse_status"],
            "json_extraction": sample.get("json_extraction", {}),
            "parsed_output": sample["parsed_output"],
            "attempts": [
                _focused_attempt_payload(attempt) for attempt in sample["attempts"]
            ],
        }
        for sample in samples
    ]


def _focused_attempt_payload(attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_index": attempt["attempt_index"],
        "generation_options": attempt["generation_options"],
        "prompt_id": attempt["prompt_id"],
        "prompt_sha256": attempt["prompt_sha256"],
        "attempt_prompt_sha256": attempt["attempt_prompt_sha256"],
        "is_repair_prompt": attempt["is_repair_prompt"],
        "raw_output_text": attempt["raw_output_text"],
        "reasoning_block": attempt["reasoning_block"],
        "reasoning_text": attempt["reasoning_text"],
        "json_text": attempt["json_text"],
        "reasoning_parse_status": attempt["reasoning_parse_status"],
        "json_extraction": attempt.get("json_extraction", {}),
        "model_parsed_output": attempt["model_parsed_output"],
        "parsed_output": attempt["parsed_output"],
        "canonical_corrections": attempt["canonical_corrections"],
        "parse_status": attempt["parse_status"],
        "validation_errors": attempt["validation_errors"],
        "validation_warnings": attempt["validation_warnings"],
        "elapsed_seconds": attempt["elapsed_seconds"],
    }


def _validate_context_configuration(
    *,
    context_scope: str,
    records: list[DatasetRecord],
    strategy: str,
    prompt: PromptTemplate,
    critique_prompt: PromptTemplate | None,
    revision_prompt: PromptTemplate | None,
    context_turns_before: int,
    context_turns_after: int,
) -> None:
    if context_scope == "immediate":
        return

    required_prompts = [("main", prompt)]
    if strategy == "self_refine":
        if critique_prompt is None or revision_prompt is None:
            raise ValueError("Self-refine prompt templates are not loaded.")
        required_prompts.extend(
            [
                ("critique", critique_prompt),
                ("revision", revision_prompt),
            ]
        )
    for label, template in required_prompts:
        if not template.uses_variable("analysis_context"):
            raise ValueError(
                f"context_scope={context_scope!r} requires {{analysis_context}} "
                f"in the {label} prompt: {template.path}"
            )

    for record in records:
        record.analysis_context(
            context_scope,
            context_turns_before=context_turns_before,
            context_turns_after=context_turns_after,
        )


if __name__ == "__main__":
    raise SystemExit(main())
