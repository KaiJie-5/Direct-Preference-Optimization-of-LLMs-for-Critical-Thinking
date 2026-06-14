from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .data import load_records
from .logging import RunLogger
from .prompts import PromptTemplate, parse_prompt_vars
from .strategies import run_self_consistency, run_self_refine
from .teachers import GenerationOptions, build_teacher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enrich an input dataset with teacher-model reasoning outputs."
    )

    io_group = parser.add_argument_group("input/output")
    io_group.add_argument("--input-path", required=True, type=Path)
    io_group.add_argument("--input-format", default="auto", choices=["auto", "html", "jsonl", "json", "csv", "txt"])
    io_group.add_argument("--output-dir", required=True, type=Path)
    io_group.add_argument("--limit", type=int)
    io_group.add_argument("--text-field", default="text")
    io_group.add_argument("--record-id-field")
    io_group.add_argument("--html-split-mode", choices=["participant", "whole", "css"], default="participant")
    io_group.add_argument("--html-record-selector")
    io_group.add_argument("--html-text-selector")
    io_group.add_argument("--html-id-attr")
    io_group.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)

    strategy_group = parser.add_argument_group("prompting strategy")
    strategy_group.add_argument("--strategy", required=True, choices=["self_consistency", "self_refine"])
    strategy_group.add_argument("--prompt-path", required=True, type=Path)
    strategy_group.add_argument("--prompt-var", action="append", default=[], help="Extra template variable in KEY=VALUE form. May be repeated.")
    strategy_group.add_argument("--self-consistency-samples", type=int, default=5)
    strategy_group.add_argument("--self-consistency-aggregation", choices=["scaffold"], default="scaffold")
    strategy_group.add_argument("--refine-rounds", type=int, default=2)
    strategy_group.add_argument("--refine-critique-prompt-path", type=Path)
    strategy_group.add_argument("--refine-revision-prompt-path", type=Path)
    strategy_group.add_argument("--refine-stop-parser", choices=["json", "text"], default="json")
    strategy_group.add_argument("--refine-history-format", choices=["text", "json"], default="text")

    teacher_group = parser.add_argument_group("teacher backend")
    teacher_group.add_argument("--teacher-backend", choices=["dry-run", "transformers", "openai-compatible"], default="dry-run")
    teacher_group.add_argument("--model-path")
    teacher_group.add_argument("--model-name")
    teacher_group.add_argument("--torch-dtype", default="auto")
    teacher_group.add_argument("--device-map", default="auto")
    teacher_group.add_argument("--trust-remote-code", action="store_true")
    teacher_group.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    teacher_group.add_argument("--force-think-prefix", action=argparse.BooleanOptionalAction, default=True)
    teacher_group.add_argument("--think-prefix", default="<think>\n")
    teacher_group.add_argument("--api-base")
    teacher_group.add_argument("--api-key-env")
    teacher_group.add_argument("--timeout-seconds", type=float, default=600.0)

    generation_group = parser.add_argument_group("generation")
    generation_group.add_argument("--max-new-tokens", type=int, default=2048)
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

    logger = RunLogger(args.output_dir)
    prompt_vars = parse_prompt_vars(args.prompt_var)
    generation_options = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        stop=args.stop,
    )

    teacher = build_teacher(args)
    records = load_records(
        args.input_path,
        input_format=args.input_format,
        text_field=args.text_field,
        record_id_field=args.record_id_field,
        html_split_mode=args.html_split_mode,
        html_record_selector=args.html_record_selector,
        html_text_selector=args.html_text_selector,
        html_id_attr=args.html_id_attr,
        limit=args.limit,
    )

    manifest = _manifest(args, teacher.metadata(), generation_options, len(records))
    logger.write_manifest(manifest)
    logger.event({"event": "run_started", "manifest": manifest})

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

    success_count = 0
    failure_count = 0
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        try:
            logger.event(
                {
                    "event": "record_started",
                    "record_index": index,
                    "record_count": len(records),
                    "record_id": record.record_id,
                    "input_char_count": len(record.text),
                }
            )
            enriched = _run_strategy(
                args=args,
                record=record,
                teacher=teacher,
                prompt=prompt,
                critique_prompt=critique_prompt,
                revision_prompt=revision_prompt,
                prompt_vars=prompt_vars,
                generation_options=generation_options,
                logger=logger,
            )
            logger.enriched_record(enriched)
            logger.event({"event": "record_completed", "record_id": record.record_id})
            success_count += 1
        except Exception as exc:
            failure_count += 1
            logger.failure(
                {
                    "event": "record_failed",
                    "record_id": record.record_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if not args.continue_on_error:
                raise

    summary = {
        "event": "run_completed",
        "success_count": success_count,
        "failure_count": failure_count,
        "elapsed_seconds": time.perf_counter() - started,
        "output_dir": str(args.output_dir),
    }
    logger.event(summary)
    print(json.dumps(summary, indent=2))
    return 0 if failure_count == 0 else 1


def _run_strategy(
    *,
    args: argparse.Namespace,
    record: Any,
    teacher: Any,
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
            generation_options=generation_options,
            num_samples=args.self_consistency_samples,
            aggregation=args.self_consistency_aggregation,
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
            generation_options=generation_options,
            refine_rounds=args.refine_rounds,
            stop_parser=args.refine_stop_parser,
            history_format=args.refine_history_format,
            logger=logger,
        )

    raise ValueError(f"Unsupported strategy: {args.strategy}")


def _manifest(
    args: argparse.Namespace,
    teacher_metadata: dict[str, Any],
    generation_options: GenerationOptions,
    record_count: int,
) -> dict[str, Any]:
    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "record_count": record_count,
        "args": args_dict,
        "teacher": teacher_metadata,
        "generation_options": asdict(generation_options),
    }


if __name__ == "__main__":
    raise SystemExit(main())
