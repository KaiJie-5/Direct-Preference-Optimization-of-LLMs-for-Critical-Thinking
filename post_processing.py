from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from enrichment.data import DatasetRecord
from enrichment.schema import (
    parse_json_object,
    validate_segment_enrichment_sample_result,
)


AGGREGATE_SCHEMA_VERSION = "failed_enrichment_outputs_v1"
VALID_SCOPES = ("invalid-samples", "no-valid-segments", "both")
RAW_ATTEMPT_FIELDS = (
    "attempt_index",
    "parse_status",
    "validation_errors",
    "reasoning_parse_status",
    "reasoning_text",
    "reasoning_block",
    "json_text",
    "raw_output_text",
    "generation_options",
    "elapsed_seconds",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process enrichment outputs. Use aggregate to collect failed "
            "parsed samples; repair is reserved for a later replacement workflow."
        )
    )
    parser.add_argument(
        "--process",
        required=True,
        choices=["aggregate", "repair"],
        help="Top-level post-processing action to run.",
    )
    parser.add_argument(
        "--enriched-dir",
        type=Path,
        help=(
            "Enriched output directory for --process aggregate. This may be a "
            "single run folder, an interview output folder, or a parent folder "
            "containing run folders."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "Where to write the aggregate JSON. Defaults to "
            "<enriched-dir>/failed_parsed_outputs.json."
        ),
    )
    parser.add_argument(
        "--scope",
        choices=VALID_SCOPES,
        default="invalid-samples",
        help=(
            "invalid-samples collects every invalid sample; no-valid-segments "
            "collects segment summaries where all samples are invalid; both writes both."
        ),
    )
    parser.add_argument(
        "--include-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach matching final-attempt raw output from sibling events.jsonl files.",
    )
    parser.add_argument(
        "--strategy",
        default="self_consistency",
        help="Only process segment JSON files whose strategy field matches this value.",
    )
    parser.add_argument(
        "--segment-glob",
        default="**/segments/*.json",
        help="Glob, relative to --enriched-dir, used to discover segment JSON files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing --output-path.",
    )
    parser.add_argument(
        "--failed-path",
        type=Path,
        help="Failed aggregate JSON for --process repair.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help=(
            "Repair report path. Defaults to "
            "<failed-path stem>_repair_report.json."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="For --process repair, report planned changes without writing segment files.",
    )
    return parser


def collect_failed_outputs(
    *,
    enriched_dir: Path,
    output_path: Path | None = None,
    scope: str = "invalid-samples",
    include_raw: bool = True,
    strategy: str = "self_consistency",
    segment_glob: str = "**/segments/*.json",
) -> dict[str, Any]:
    if scope not in VALID_SCOPES:
        raise ValueError(f"scope must be one of {VALID_SCOPES}, got {scope!r}")

    enriched_dir = enriched_dir.resolve()
    if not enriched_dir.is_dir():
        raise FileNotFoundError(f"Enriched directory does not exist: {enriched_dir}")

    output_path = (
        output_path.resolve()
        if output_path is not None
        else (enriched_dir / "failed_parsed_outputs.json").resolve()
    )

    warnings: list[str] = []
    raw_event_cache: dict[Path, dict[tuple[str, int, str], dict[str, Any]]] = {}
    failed_samples: list[dict[str, Any]] = []
    no_valid_segments: list[dict[str, Any]] = []

    counts = {
        "segment_files_found": 0,
        "segments_scanned": 0,
        "segments_skipped_for_strategy": 0,
        "samples_scanned": 0,
        "invalid_samples": 0,
        "segments_with_no_valid_samples": 0,
        "raw_attempts_matched": 0,
        "raw_attempts_missing": 0,
        "event_files_read": 0,
        "event_files_missing": 0,
        "event_lines_invalid": 0,
    }

    for segment_path in sorted(enriched_dir.glob(segment_glob)):
        counts["segment_files_found"] += 1
        try:
            segment = _read_json(segment_path)
        except Exception as exc:
            warnings.append(f"Could not read segment JSON {segment_path}: {exc}")
            continue

        if not isinstance(segment, dict):
            warnings.append(f"Skipping non-object segment JSON: {segment_path}")
            continue

        if segment.get("strategy") != strategy:
            counts["segments_skipped_for_strategy"] += 1
            continue

        samples = segment.get("samples", [])
        if not isinstance(samples, list):
            warnings.append(f"Skipping segment with non-list samples: {segment_path}")
            continue

        counts["segments_scanned"] += 1
        counts["samples_scanned"] += len(samples)

        valid_samples = [
            sample
            for sample in samples
            if isinstance(sample, dict) and sample.get("final_parse_status") == "valid"
        ]
        _warn_on_duplicate_sample_indexes(
            samples=samples,
            segment=segment,
            segment_path=segment_path,
            warnings=warnings,
        )
        invalid_samples = _invalid_samples_with_list_indexes(samples)
        counts["invalid_samples"] += len(invalid_samples)

        if not valid_samples and samples:
            counts["segments_with_no_valid_samples"] += 1
            if scope in {"no-valid-segments", "both"}:
                no_valid_segments.append(
                    _segment_summary(
                        segment=segment,
                        segment_path=segment_path,
                        enriched_dir=enriched_dir,
                        invalid_sample_count=len(invalid_samples),
                        total_sample_count=len(samples),
                    )
                )

        if scope in {"invalid-samples", "both"}:
            for sample_list_index, sample in invalid_samples:
                item = _failed_sample_item(
                    segment=segment,
                    sample=sample,
                    sample_list_index=sample_list_index,
                    segment_path=segment_path,
                    enriched_dir=enriched_dir,
                    warnings=warnings,
                )
                if include_raw:
                    raw_event, event_state = _find_raw_event(
                        segment_path=segment_path,
                        record_id=str(segment.get("record_id", "")),
                        sample_index=_as_int(sample.get("sample_index")),
                        strategy=str(segment.get("strategy", "")),
                        raw_event_cache=raw_event_cache,
                    )
                    _merge_event_state_counts(counts, event_state)
                    if raw_event is None:
                        counts["raw_attempts_missing"] += 1
                        warnings.append(
                            "No matching final raw attempt found for "
                            f"{segment.get('record_id')} sample "
                            f"{sample.get('sample_index')} in {segment_path}"
                        )
                    else:
                        counts["raw_attempts_matched"] += 1
                        item["raw_final_attempt"] = _compact_raw_attempt(raw_event)
                failed_samples.append(item)

    result: dict[str, Any] = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "enriched_dir": str(enriched_dir),
        "output_path": str(output_path),
        "settings": {
            "scope": scope,
            "include_raw": include_raw,
            "strategy": strategy,
            "segment_glob": segment_glob,
        },
        "counts": counts,
        "warnings": warnings,
    }

    if scope in {"invalid-samples", "both"}:
        result["failed_samples"] = failed_samples
    if scope in {"no-valid-segments", "both"}:
        result["segments_with_no_valid_samples"] = no_valid_segments

    return result


def write_aggregate(
    payload: dict[str, Any],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def repair_failed_outputs(
    *,
    failed_path: Path,
    report_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    failed_path = failed_path.resolve()
    failed_payload = _read_json(failed_path)
    if not isinstance(failed_payload, dict):
        raise ValueError(f"Failed aggregate must be a JSON object: {failed_path}")

    failed_samples = failed_payload.get("failed_samples", [])
    if not isinstance(failed_samples, list):
        raise ValueError("Failed aggregate must contain a failed_samples list.")

    report_path = (
        report_path.resolve()
        if report_path is not None
        else failed_path.with_name(f"{failed_path.stem}_repair_report.json").resolve()
    )
    enriched_dir = Path(str(failed_payload.get("enriched_dir", "")))
    segment_cache: dict[Path, dict[str, Any]] = {}
    touched_paths: set[Path] = set()
    outcomes: list[dict[str, Any]] = []
    counts = {
        "failed_samples": len(failed_samples),
        "repaired": 0,
        "skipped": 0,
        "unrepairable": 0,
        "files_touched": 0,
        "files_written": 0,
    }

    for failed_sample in failed_samples:
        if not isinstance(failed_sample, dict):
            outcomes.append(
                {
                    "status": "skipped",
                    "reason": "failed_samples item is not an object",
                }
            )
            counts["skipped"] += 1
            continue

        target_path = _resolve_repair_target_path(failed_sample, enriched_dir)
        target = failed_sample.get("repair_target", {})
        outcome_base = {
            "record_id": failed_sample.get("record_id"),
            "sample_index": failed_sample.get("sample_index"),
            "segment_path": str(target_path) if target_path is not None else None,
        }
        if target_path is None or not target_path.exists():
            outcomes.append(
                {
                    **outcome_base,
                    "status": "skipped",
                    "reason": "repair target segment file does not exist",
                }
            )
            counts["skipped"] += 1
            continue

        segment = segment_cache.get(target_path)
        if segment is None:
            loaded = _read_json(target_path)
            if not isinstance(loaded, dict):
                outcomes.append(
                    {
                        **outcome_base,
                        "status": "skipped",
                        "reason": "repair target segment JSON is not an object",
                    }
                )
                counts["skipped"] += 1
                continue
            segment = loaded
            segment_cache[target_path] = segment

        samples = segment.get("samples")
        sample_list_index = _as_int(target.get("sample_list_index"))
        if not isinstance(samples, list) or sample_list_index is None:
            outcomes.append(
                {
                    **outcome_base,
                    "status": "skipped",
                    "reason": "segment samples or sample_list_index is invalid",
                }
            )
            counts["skipped"] += 1
            continue
        if sample_list_index < 0 or sample_list_index >= len(samples):
            outcomes.append(
                {
                    **outcome_base,
                    "status": "skipped",
                    "reason": "sample_list_index is outside the segment samples list",
                }
            )
            counts["skipped"] += 1
            continue

        current_sample = samples[sample_list_index]
        guard_error = _repair_guard_error(
            segment=segment,
            current_sample=current_sample,
            failed_sample=failed_sample,
        )
        if guard_error is not None:
            outcomes.append(
                {
                    **outcome_base,
                    "status": "skipped",
                    "reason": guard_error,
                }
            )
            counts["skipped"] += 1
            continue

        repair_result = _build_repaired_sample(
            failed_sample=failed_sample,
            current_sample=current_sample,
            failed_path=failed_path,
        )
        if repair_result["status"] != "repaired":
            outcomes.append({**outcome_base, **repair_result})
            counts["unrepairable"] += 1
            continue

        samples[sample_list_index] = repair_result["sample"]
        touched_paths.add(target_path)
        outcomes.append(
            {
                **outcome_base,
                "status": "repaired",
                "method": repair_result["method"],
                "validation_warnings": repair_result["validation_warnings"],
            }
        )
        counts["repaired"] += 1

    counts["files_touched"] = len(touched_paths)
    if not dry_run:
        for path in sorted(touched_paths):
            path.write_text(
                json.dumps(segment_cache[path], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        counts["files_written"] = len(touched_paths)

    report = {
        "schema_version": "failed_enrichment_repair_report_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "failed_path": str(failed_path),
        "report_path": str(report_path),
        "dry_run": dry_run,
        "counts": counts,
        "outcomes": outcomes,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.process == "repair":
        if args.failed_path is None:
            print(
                "post_processing.py: error: --failed-path is required for "
                "--process repair",
                file=sys.stderr,
            )
            return 1
        try:
            report = repair_failed_outputs(
                failed_path=args.failed_path,
                report_path=args.report_path,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            print(f"post_processing.py: error: {exc}", file=sys.stderr)
            return 1
        counts = report["counts"]
        print(
            "Repair complete: "
            f"{counts['repaired']} repaired, "
            f"{counts['skipped']} skipped, "
            f"{counts['unrepairable']} unrepairable. "
            f"Report: {report['report_path']}"
        )
        return 0

    if args.enriched_dir is None:
        print(
            "post_processing.py: error: --enriched-dir is required for "
            "--process aggregate",
            file=sys.stderr,
        )
        return 1

    output_path = (
        args.output_path
        if args.output_path is not None
        else args.enriched_dir / "failed_parsed_outputs.json"
    )

    try:
        payload = collect_failed_outputs(
            enriched_dir=args.enriched_dir,
            output_path=output_path,
            scope=args.scope,
            include_raw=args.include_raw,
            strategy=args.strategy,
            segment_glob=args.segment_glob,
        )
        write_aggregate(payload, output_path, overwrite=args.overwrite)
    except Exception as exc:
        print(f"post_processing.py: error: {exc}", file=sys.stderr)
        return 1

    counts = payload["counts"]
    print(
        "Wrote "
        f"{output_path} with {counts['invalid_samples']} invalid samples "
        f"from {counts['segments_scanned']} scanned segments."
    )
    if payload["warnings"]:
        print(f"Warnings: {len(payload['warnings'])}", file=sys.stderr)
    return 0


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    invalid_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if isinstance(payload, dict):
                records.append(payload)
            else:
                invalid_lines += 1
    return records, invalid_lines


def _failed_sample_item(
    *,
    segment: dict[str, Any],
    sample: dict[str, Any],
    sample_list_index: int,
    segment_path: Path,
    enriched_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    relative_segment_path = _relative_or_absolute(segment_path, enriched_dir)
    repair_target = _repair_target(
        segment=segment,
        sample=sample,
        sample_list_index=sample_list_index,
        segment_path=segment_path,
        relative_segment_path=relative_segment_path,
        warnings=warnings,
    )
    return {
        "record_id": segment.get("record_id"),
        "interview_id": segment.get("interview_id"),
        "segment_id": segment.get("segment_id"),
        "strategy": segment.get("strategy"),
        "sample_index": sample.get("sample_index"),
        "segment_path": relative_segment_path,
        "repair_target": repair_target,
        "input_text": segment.get("input_text"),
        "metadata": segment.get("metadata"),
        "source": segment.get("source"),
        "final_parse_status": sample.get("final_parse_status"),
        "validation_errors": sample.get("validation_errors", []),
        "reasoning_text": sample.get("reasoning_text", ""),
        "parsed_output": sample.get("parsed_output"),
    }


def _segment_summary(
    *,
    segment: dict[str, Any],
    segment_path: Path,
    enriched_dir: Path,
    invalid_sample_count: int,
    total_sample_count: int,
) -> dict[str, Any]:
    return {
        "record_id": segment.get("record_id"),
        "interview_id": segment.get("interview_id"),
        "segment_id": segment.get("segment_id"),
        "strategy": segment.get("strategy"),
        "segment_path": _relative_or_absolute(segment_path, enriched_dir),
        "input_text": segment.get("input_text"),
        "metadata": segment.get("metadata"),
        "invalid_sample_count": invalid_sample_count,
        "total_sample_count": total_sample_count,
        "invalid_sample_indexes": [
            sample.get("sample_index")
            for sample in segment.get("samples", [])
            if isinstance(sample, dict)
            and sample.get("final_parse_status") == "invalid"
        ],
    }


def _find_raw_event(
    *,
    segment_path: Path,
    record_id: str,
    sample_index: int | None,
    strategy: str,
    raw_event_cache: dict[Path, dict[tuple[str, int, str], dict[str, Any]]],
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    event_state = {
        "event_files_read": 0,
        "event_files_missing": 0,
        "event_lines_invalid": 0,
    }
    if sample_index is None:
        return None, event_state

    event_path = segment_path.parent.parent / "events.jsonl"
    if event_path not in raw_event_cache:
        if not event_path.exists():
            raw_event_cache[event_path] = {}
            event_state["event_files_missing"] = 1
        else:
            events, invalid_lines = _read_jsonl(event_path)
            raw_event_cache[event_path] = _index_generation_events(events)
            event_state["event_files_read"] = 1
            event_state["event_lines_invalid"] = invalid_lines

    key = (record_id, sample_index, strategy)
    return raw_event_cache[event_path].get(key), event_state


def _resolve_repair_target_path(
    failed_sample: dict[str, Any],
    enriched_dir: Path,
) -> Path | None:
    target = failed_sample.get("repair_target")
    if not isinstance(target, dict):
        return None

    absolute = target.get("absolute_segment_path")
    if isinstance(absolute, str) and absolute.strip():
        absolute_path = Path(absolute)
        if absolute_path.exists():
            return absolute_path.resolve()

    relative = target.get("relative_segment_path")
    if isinstance(relative, str) and relative.strip():
        return (enriched_dir / relative).resolve()
    return None


def _repair_guard_error(
    *,
    segment: dict[str, Any],
    current_sample: Any,
    failed_sample: dict[str, Any],
) -> str | None:
    if not isinstance(current_sample, dict):
        return "target sample is not an object"

    target = failed_sample.get("repair_target")
    if not isinstance(target, dict):
        return "failed sample is missing repair_target"

    if segment.get("record_id") != target.get("record_id"):
        return "record_id does not match repair target"
    if current_sample.get("sample_index") != target.get("sample_index"):
        return "sample_index does not match repair target"

    expected_fingerprint = target.get("original_sample_fingerprint")
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        return "repair target is missing original_sample_fingerprint"
    if _sample_fingerprint(current_sample) != expected_fingerprint:
        return "current sample fingerprint does not match repair target"
    return None


def _build_repaired_sample(
    *,
    failed_sample: dict[str, Any],
    current_sample: dict[str, Any],
    failed_path: Path,
) -> dict[str, Any]:
    parsed = failed_sample.get("parsed_output")
    method = "relaxed_target_text_validation"

    if not isinstance(parsed, dict):
        if not _has_extra_data_error(failed_sample):
            return {
                "status": "unrepairable",
                "reason": "failed sample has no parsed_output and is not an Extra data parse case",
            }
        raw_text = _raw_repair_text(failed_sample)
        if not raw_text:
            return {
                "status": "unrepairable",
                "reason": "Extra data parse case has no raw output text",
            }
        parsed, parse_error = parse_json_object(raw_text)
        if parsed is None:
            return {
                "status": "unrepairable",
                "reason": parse_error or "raw output could not be parsed",
            }
        method = "trailing_extra_brace_parse_recovery"

    validation = _validate_failed_sample_payload(parsed, failed_sample)
    if validation.errors:
        return {
            "status": "unrepairable",
            "reason": "validation errors remain after deterministic repair",
            "validation_errors": validation.errors,
        }

    repaired = dict(current_sample)
    repaired["final_parse_status"] = "valid"
    repaired["validation_errors"] = []
    repaired["validation_warnings"] = validation.warnings
    repaired["parsed_output"] = parsed
    repaired["reasoning_text"] = failed_sample.get(
        "reasoning_text",
        repaired.get("reasoning_text", ""),
    )
    repaired["repair_metadata"] = {
        "repaired_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": method,
        "source_failed_path": str(failed_path),
        "original_sample_fingerprint": failed_sample.get("repair_target", {}).get(
            "original_sample_fingerprint"
        ),
        "original_validation_errors": failed_sample.get("validation_errors", []),
    }
    return {
        "status": "repaired",
        "method": method,
        "sample": repaired,
        "validation_warnings": validation.warnings,
    }


def _validate_failed_sample_payload(
    parsed: dict[str, Any],
    failed_sample: dict[str, Any],
) -> Any:
    metadata = failed_sample.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {
            "interview_id": failed_sample.get("interview_id"),
            "segment_id": failed_sample.get("segment_id"),
            "speaker": "participant",
        }
    source = failed_sample.get("source") if isinstance(failed_sample.get("source"), dict) else {}
    record = DatasetRecord(
        record_id=str(failed_sample.get("record_id", "")),
        text=str(failed_sample.get("input_text", "")),
        metadata=metadata,
        source=source,
    )
    return validate_segment_enrichment_sample_result(
        parsed,
        record,
        expected_codebook_version=parsed.get("codebook_version"),
        strict_prompt_schema=failed_sample.get("strategy") == "self_consistency",
        allow_target_text_mismatch=True,
    )


def _has_extra_data_error(failed_sample: dict[str, Any]) -> bool:
    errors = failed_sample.get("validation_errors", [])
    return any(isinstance(error, str) and error.startswith("Extra data:") for error in errors)


def _raw_repair_text(failed_sample: dict[str, Any]) -> str:
    raw_attempt = failed_sample.get("raw_final_attempt")
    if not isinstance(raw_attempt, dict):
        return ""
    raw_output = raw_attempt.get("raw_output_text")
    if isinstance(raw_output, str) and raw_output.strip():
        return raw_output
    json_text = raw_attempt.get("json_text")
    if isinstance(json_text, str):
        return json_text
    return ""


def _invalid_samples_with_list_indexes(
    samples: list[Any],
) -> list[tuple[int, dict[str, Any]]]:
    return [
        (sample_list_index, sample)
        for sample_list_index, sample in enumerate(samples)
        if isinstance(sample, dict) and sample.get("final_parse_status") == "invalid"
    ]


def _repair_target(
    *,
    segment: dict[str, Any],
    sample: dict[str, Any],
    sample_list_index: int,
    segment_path: Path,
    relative_segment_path: str,
    warnings: list[str],
) -> dict[str, Any]:
    record_id = segment.get("record_id")
    sample_index = sample.get("sample_index")
    strategy = segment.get("strategy")
    missing_fields = []

    if not record_id:
        missing_fields.append("record_id")
    if sample_index is None:
        missing_fields.append("sample_index")
        warnings.append(
            "Failed sample is missing sample_index for "
            f"{record_id or segment_path} at samples[{sample_list_index}]"
        )
    if not strategy:
        missing_fields.append("strategy")

    fingerprint = _sample_fingerprint(sample)
    if not fingerprint:
        missing_fields.append("original_sample_fingerprint")

    if missing_fields:
        warnings.append(
            "Repair target may be unsafe for "
            f"{record_id or segment_path} at samples[{sample_list_index}]; "
            f"missing: {missing_fields}"
        )

    return {
        "absolute_segment_path": str(segment_path.resolve()),
        "relative_segment_path": relative_segment_path,
        "record_id": record_id,
        "sample_index": sample_index,
        "sample_list_index": sample_list_index,
        "strategy": strategy,
        "original_sample_fingerprint": fingerprint,
    }


def _sample_fingerprint(sample: dict[str, Any]) -> str:
    serialized = json.dumps(
        sample,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _warn_on_duplicate_sample_indexes(
    *,
    samples: list[Any],
    segment: dict[str, Any],
    segment_path: Path,
    warnings: list[str],
) -> None:
    seen: dict[str, int] = {}
    duplicates: set[str] = set()
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_index = sample.get("sample_index")
        if sample_index is None:
            continue
        duplicate_key = _json_identity(sample_index)
        seen[duplicate_key] = seen.get(duplicate_key, 0) + 1
        if seen[duplicate_key] > 1:
            duplicates.add(duplicate_key)

    if duplicates:
        warnings.append(
            "Duplicate sample_index values in "
            f"{segment.get('record_id') or segment_path}: "
            f"{sorted(duplicates)}"
        )


def _json_identity(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return repr(value)


def _index_generation_events(
    events: list[dict[str, Any]],
) -> dict[tuple[str, int, str], dict[str, Any]]:
    indexed: dict[tuple[str, int, str], dict[str, Any]] = {}
    for event in events:
        if event.get("event") != "teacher_generation":
            continue
        sample_index = _as_int(event.get("sample_index"))
        if sample_index is None:
            continue
        key = (str(event.get("record_id", "")), sample_index, str(event.get("strategy", "")))
        previous = indexed.get(key)
        if previous is None or _as_int(event.get("attempt_index"), default=0) > _as_int(
            previous.get("attempt_index"), default=0
        ):
            indexed[key] = event
    return indexed


def _compact_raw_attempt(event: dict[str, Any]) -> dict[str, Any]:
    return {field: event.get(field) for field in RAW_ATTEMPT_FIELDS if field in event}


def _merge_event_state_counts(counts: dict[str, int], event_state: dict[str, int]) -> None:
    for key, value in event_state.items():
        counts[key] += value


def _as_int(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
