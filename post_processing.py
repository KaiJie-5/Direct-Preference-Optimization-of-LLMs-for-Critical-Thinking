from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.process == "repair":
        print(
            "post_processing.py: repair process is not implemented yet.",
            file=sys.stderr,
        )
        return 1

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
