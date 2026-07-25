from __future__ import annotations

import json
import os
import platform
import secrets
import sys
import time
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from debate.schema import REVIEW_BLOCK_BY_ID
from reflective_enrichment.schema import CATEGORY_ORDER

from .config import PreferencePairConfig, SourceConfig, config_to_jsonable


CATEGORY_EVIDENCE_FILENAME = "preference_pairs_category_evidence.jsonl"
QUESTION_ONLY_FILENAME = "preference_pairs_question_only.jsonl"
AUDIT_FILENAME = "preference_pair_audit.jsonl"
MANIFEST_FILENAME = "run_manifest.json"

NegativeSelector = Callable[[tuple[int, ...]], int]


def build_preference_pairs(
    config: PreferencePairConfig,
    *,
    negative_selector: NegativeSelector | None = None,
) -> Path:
    """Build both line-aligned DPO datasets and their provenance audit."""

    selector = negative_selector or _system_random_choice
    inventories = [_inventory_source(source) for source in config.sources]
    run_dir = _new_run_dir(config.output_root, config.run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / MANIFEST_FILENAME
    manifest = _initial_manifest(config, inventories)
    _write_json(manifest_path, manifest)

    final_paths = {
        "category_evidence": run_dir / CATEGORY_EVIDENCE_FILENAME,
        "question_only": run_dir / QUESTION_ONLY_FILENAME,
        "audit": run_dir / AUDIT_FILENAME,
    }
    temporary_paths = {
        name: path.with_name(f".{path.name}.tmp") for name, path in final_paths.items()
    }

    source_summaries = [_source_summary(inventory) for inventory in inventories]
    dataset_counts = _initial_dataset_counts(config)
    rejection_matrix = {
        category: {other: 0 for other in CATEGORY_ORDER if other != category}
        for category in CATEGORY_ORDER
    }
    seen_identities: set[tuple[str, str]] = set()
    pair_count = 0

    try:
        with ExitStack() as stack:
            handles = {
                name: stack.enter_context(
                    path.open("w", encoding="utf-8", newline="\n")
                )
                for name, path in temporary_paths.items()
            }
            for source_index, inventory in enumerate(inventories):
                source = inventory["source"]
                summary = source_summaries[source_index]
                for trace_path in inventory["trace_paths"]:
                    payload = _read_json_object(trace_path, "reflective trace")
                    dataset = _required_string(payload, "dataset", trace_path)
                    record_id = _required_string(payload, "record_id", trace_path)
                    if dataset not in source.datasets:
                        raise ValueError(
                            f"Trace dataset {dataset!r} is not configured for "
                            f"source {source.name!r}: {trace_path}"
                        )
                    identity = (dataset, record_id)
                    if identity in seen_identities:
                        raise ValueError(
                            f"Duplicate trace identity {dataset} {record_id}: {trace_path}"
                        )
                    seen_identities.add(identity)
                    dataset_counts[dataset]["trace_count"] += 1
                    summary["trace_count_by_dataset"][dataset] += 1

                    status = payload.get("status")
                    if status == "failed":
                        _record_skipped_trace(
                            summary=summary,
                            dataset_counts=dataset_counts,
                            payload=payload,
                            trace_path=trace_path,
                            dataset=dataset,
                            record_id=record_id,
                        )
                        continue
                    if status != "success":
                        raise ValueError(
                            f"Trace status must be 'success' or 'failed': {trace_path}"
                        )

                    record = _validate_successful_trace(
                        payload=payload,
                        trace_path=trace_path,
                    )
                    summary["accepted_count"] += 1
                    summary["accepted_count_by_dataset"][dataset] += 1
                    dataset_counts[dataset]["accepted_record_count"] += 1

                    for target_index, target_category in enumerate(CATEGORY_ORDER):
                        candidate_indexes = tuple(
                            index
                            for index in range(len(CATEGORY_ORDER))
                            if index != target_index
                        )
                        rejected_index = selector(candidate_indexes)
                        if rejected_index not in candidate_indexes:
                            raise ValueError(
                                "Negative selector returned the target or an unknown "
                                f"index for {dataset} {record_id} {target_category}."
                            )
                        rejected_category = CATEGORY_ORDER[rejected_index]
                        rejection_matrix[target_category][rejected_category] += 1

                        target_code = record["codes"][target_index]
                        rejected_code = record["codes"][rejected_index]
                        target_question = record["questions"][target_index]["question"]
                        rejected_question = record["questions"][rejected_index]["question"]
                        evidence_prompt = _render_prompt(
                            record=record,
                            source=source,
                            version="category_evidence",
                            code_label=target_code["code_label"],
                        )
                        question_prompt = _render_prompt(
                            record=record,
                            source=source,
                            version="question_only",
                            code_label=target_code["code_label"],
                        )
                        evidence_chosen = _render_evidence_response(
                            target_category, target_code, target_question
                        )
                        evidence_rejected = _render_evidence_response(
                            rejected_category, rejected_code, rejected_question
                        )
                        evidence_row = _conversation_row(
                            evidence_prompt, evidence_chosen, evidence_rejected
                        )
                        question_row = _conversation_row(
                            question_prompt, target_question, rejected_question
                        )

                        pair_count += 1
                        dataset_counts[dataset]["pair_count"] += 1
                        _write_jsonl_row(handles["category_evidence"], evidence_row)
                        _write_jsonl_row(handles["question_only"], question_row)
                        audit_row = _audit_row(
                            line_number=pair_count,
                            source=source,
                            trace_path=trace_path,
                            record=record,
                            target_category=target_category,
                            target_code=target_code,
                            target_question=target_question,
                            rejected_category=rejected_category,
                            rejected_code=rejected_code,
                            rejected_question=rejected_question,
                            evidence_row=evidence_row,
                            question_row=question_row,
                        )
                        _write_jsonl_row(handles["audit"], audit_row)

        for name, temporary_path in temporary_paths.items():
            os.replace(temporary_path, final_paths[name])

        completed_at = _timestamp()
        manifest.update(
            {
                "source_summaries": source_summaries,
                "dataset_counts": dataset_counts,
                "counts": {
                    "accepted_record_count": sum(
                        item["accepted_record_count"]
                        for item in dataset_counts.values()
                    ),
                    "skipped_failed_trace_count": sum(
                        item["skipped_failed_trace_count"]
                        for item in dataset_counts.values()
                    ),
                    "missing_trace_count": sum(
                        item["missing_trace_count"] for item in source_summaries
                    ),
                    "pair_count_per_version": pair_count,
                    "category_evidence_row_count": pair_count,
                    "question_only_row_count": pair_count,
                    "audit_row_count": pair_count,
                },
                "rejection_category_matrix": rejection_matrix,
                "audit_alignment": {
                    "is_line_aligned": True,
                    "category_evidence_rows": pair_count,
                    "question_only_rows": pair_count,
                    "audit_rows": pair_count,
                },
                "output_files": {
                    name: _file_metadata(path, pair_count)
                    for name, path in final_paths.items()
                },
            }
        )
        manifest["run_state"].update(
            {
                "status": "complete",
                "updated_at_utc": completed_at,
                "completed_at_utc": completed_at,
            }
        )
        _write_json(manifest_path, manifest)
    except Exception as exc:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
        manifest["source_summaries"] = source_summaries
        manifest["dataset_counts"] = dataset_counts
        manifest["run_state"].update(
            {
                "status": "failed",
                "updated_at_utc": _timestamp(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json(manifest_path, manifest)
        raise

    _print_summary(run_dir, manifest)
    return run_dir


def _inventory_source(source: SourceConfig) -> dict[str, Any]:
    if not source.trace_root.is_dir():
        raise FileNotFoundError(
            f"Preference-pair trace root does not exist: {source.trace_root}"
        )
    manifest = _read_json_object(source.manifest_path, "reflective run manifest")
    record_count = manifest.get("record_count")
    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count < 0:
        raise ValueError(
            f"Manifest record_count must be a non-negative integer: "
            f"{source.manifest_path}"
        )
    run_state = manifest.get("run_state")
    if not isinstance(run_state, dict):
        raise ValueError(f"Manifest run_state must be an object: {source.manifest_path}")
    trace_paths = tuple(
        sorted(source.trace_root.rglob("*.json"), key=lambda path: path.as_posix())
    )
    if len(trace_paths) > record_count:
        raise ValueError(
            f"Source {source.name!r} contains more traces ({len(trace_paths)}) "
            f"than its manifest record_count ({record_count})."
        )
    return {
        "source": source,
        "manifest": manifest,
        "trace_paths": trace_paths,
        "manifest_sha256": _file_sha256(source.manifest_path),
        "missing_trace_count": record_count - len(trace_paths),
    }


def _source_summary(inventory: dict[str, Any]) -> dict[str, Any]:
    source: SourceConfig = inventory["source"]
    manifest = inventory["manifest"]
    state = manifest["run_state"]
    return {
        "name": source.name,
        "trace_root": str(source.trace_root),
        "manifest_path": str(source.manifest_path),
        "manifest_sha256": inventory["manifest_sha256"],
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_status": state.get("status"),
        "manifest_record_count": manifest["record_count"],
        "manifest_success_count": state.get("success_count"),
        "manifest_failure_count": state.get("failure_count"),
        "trace_count": len(inventory["trace_paths"]),
        "missing_trace_count": inventory["missing_trace_count"],
        "accepted_count": 0,
        "skipped_failed_trace_count": 0,
        "trace_count_by_dataset": {dataset: 0 for dataset in source.datasets},
        "accepted_count_by_dataset": {dataset: 0 for dataset in source.datasets},
        "skipped_records": [],
    }


def _initial_dataset_counts(config: PreferencePairConfig) -> dict[str, dict[str, int]]:
    expected = config.expected_counts()
    return {
        dataset: {
            "expected_record_count": expected.get(dataset, 0),
            "trace_count": 0,
            "accepted_record_count": 0,
            "skipped_failed_trace_count": 0,
            "pair_count": 0,
        }
        for source in config.sources
        for dataset in source.datasets
    }


def _record_skipped_trace(
    *,
    summary: dict[str, Any],
    dataset_counts: dict[str, dict[str, int]],
    payload: dict[str, Any],
    trace_path: Path,
    dataset: str,
    record_id: str,
) -> None:
    errors = payload.get("validation_errors")
    normalized_errors = (
        [str(error) for error in errors]
        if isinstance(errors, list)
        else ["validation_errors was unavailable or malformed"]
    )
    summary["skipped_failed_trace_count"] += 1
    summary["skipped_records"].append(
        {
            "dataset": dataset,
            "record_id": record_id,
            "status": "failed",
            "validation_errors": normalized_errors,
            "trace_path": str(trace_path),
        }
    )
    dataset_counts[dataset]["skipped_failed_trace_count"] += 1


def _validate_successful_trace(
    *, payload: dict[str, Any], trace_path: Path
) -> dict[str, Any]:
    if payload.get("validation_errors") != []:
        raise ValueError(
            f"Successful trace must have empty validation_errors: {trace_path}"
        )
    dataset = _required_string(payload, "dataset", trace_path)
    record_id = _required_string(payload, "record_id", trace_path)
    transcript_id = _required_string(payload, "transcript_id", trace_path)
    segment_id = _required_string(payload, "segment_id", trace_path)
    target_segment = _required_string(payload, "target_segment", trace_path)
    context = _required_string(payload, "full_interview_context", trace_path)
    research_questions = payload.get("research_questions")
    if (
        not isinstance(research_questions, list)
        or not research_questions
        or not all(
            isinstance(question, str) and question.strip()
            for question in research_questions
        )
    ):
        raise ValueError(
            f"Successful trace research_questions must be a non-empty string list: "
            f"{trace_path}"
        )

    selected_codes = payload.get("selected_codes")
    if not isinstance(selected_codes, list) or len(selected_codes) != len(CATEGORY_ORDER):
        raise ValueError(
            f"Successful trace must contain four selected_codes: {trace_path}"
        )
    codes: list[dict[str, str]] = []
    for index, category in enumerate(CATEGORY_ORDER):
        item = selected_codes[index]
        if not isinstance(item, dict) or item.get("hint") != category:
            raise ValueError(
                f"selected_codes[{index}] must have hint {category!r}: {trace_path}"
            )
        code = item.get("code")
        expected_fields = REVIEW_BLOCK_BY_ID[category].fields
        if not isinstance(code, dict) or set(code) != set(expected_fields):
            raise ValueError(
                f"selected_codes[{index}].code fields do not match {category}: "
                f"{trace_path}"
            )
        if not all(
            isinstance(code.get(field), str) and code[field].strip()
            for field in expected_fields
        ):
            raise ValueError(
                f"selected_codes[{index}].code contains an empty/non-string field: "
                f"{trace_path}"
            )
        codes.append({field: code[field] for field in expected_fields})

    parsed_output = payload.get("parsed_output")
    top_level_questions = payload.get("reflective_questions")
    if not isinstance(parsed_output, dict):
        raise ValueError(f"Successful trace lacks parsed_output: {trace_path}")
    parsed_questions = parsed_output.get("reflective_questions")
    if (
        not isinstance(top_level_questions, list)
        or top_level_questions != parsed_questions
    ):
        raise ValueError(
            f"Successful trace top-level/parsed reflective questions differ: "
            f"{trace_path}"
        )
    if len(top_level_questions) != len(CATEGORY_ORDER):
        raise ValueError(
            f"Successful trace must contain four reflective questions: {trace_path}"
        )
    normalized_questions: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    for index, category in enumerate(CATEGORY_ORDER):
        question_item = top_level_questions[index]
        if not isinstance(question_item, dict) or set(question_item) != {
            "code",
            "hint",
            "question",
        }:
            raise ValueError(
                f"reflective_questions[{index}] has invalid fields: {trace_path}"
            )
        if question_item.get("hint") != category:
            raise ValueError(
                f"reflective_questions[{index}].hint must be {category!r}: "
                f"{trace_path}"
            )
        if question_item.get("code") != codes[index]["code_label"]:
            raise ValueError(
                f"reflective_questions[{index}].code does not match its selected "
                f"code label: {trace_path}"
            )
        question = question_item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"reflective_questions[{index}].question must be non-empty: "
                f"{trace_path}"
            )
        normalized = " ".join(question.casefold().split())
        if normalized in seen_questions:
            raise ValueError(
                f"Successful trace contains duplicate reflective questions: "
                f"{trace_path}"
            )
        seen_questions.add(normalized)
        normalized_questions.append(
            {"code": codes[index]["code_label"], "hint": category, "question": question}
        )

    return {
        "dataset": dataset,
        "record_id": record_id,
        "transcript_id": transcript_id,
        "segment_id": segment_id,
        "target_segment": target_segment,
        "research_questions": list(research_questions),
        "context": context,
        "codes": codes,
        "questions": normalized_questions,
    }


def _render_prompt(
    *,
    record: dict[str, Any],
    source: SourceConfig,
    version: str,
    code_label: str,
) -> str:
    if version == "category_evidence":
        task = (
            "Assess the supplied code. Identify its code-quality category, explain "
            "all evidence and rationale for that classification, and formulate one "
            "specific open-ended reflective question that helps a qualitative "
            "researcher critically examine the coding decision."
        )
    elif version == "question_only":
        task = (
            "Formulate one specific open-ended reflective question that helps a "
            "qualitative researcher critically examine the supplied coding decision. "
            "Return only the reflective question."
        )
    else:
        raise ValueError(f"Unknown preference-pair prompt version: {version!r}")

    if source.context_scope == "full_interview":
        context_heading = "Full interview context"
    else:
        context_heading = (
            "Interview context window "
            f"(up to {source.context_turns_before} turns before and "
            f"{source.context_turns_after} turns after the target)"
        )
    questions = "\n".join(
        f"{index}. {question}"
        for index, question in enumerate(record["research_questions"], 1)
    )
    return (
        "You are assisting a reflexive qualitative researcher using reflexive "
        "thematic analysis.\n\n"
        "The target segment is the primary evidence for the coding decision. Use "
        "the interview context only to clarify language, sequence, or ambiguity; "
        "do not treat material from other turns as evidence for the target "
        "segment.\n\n"
        f"Task:\n{task}\n\n"
        f"Research questions:\n{questions}\n\n"
        f"{context_heading}:\n{record['context']}\n\n"
        f"Target segment:\n{record['target_segment']}\n\n"
        f"Code label:\n{code_label}"
    )


def _render_evidence_response(
    category: str, code: dict[str, str], question: str
) -> str:
    block = REVIEW_BLOCK_BY_ID[category]
    lines = [f"Code category: {block.title}"]
    lines.extend(
        f"{_readable_field_name(field)}: {code[field]}" for field in block.fields
    )
    lines.append(f"Reflective question: {question}")
    return "\n".join(lines)


def _readable_field_name(field: str) -> str:
    return field.replace("_", " ").capitalize()


def _conversation_row(prompt: str, chosen: str, rejected: str) -> dict[str, Any]:
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def _audit_row(
    *,
    line_number: int,
    source: SourceConfig,
    trace_path: Path,
    record: dict[str, Any],
    target_category: str,
    target_code: dict[str, str],
    target_question: str,
    rejected_category: str,
    rejected_code: dict[str, str],
    rejected_question: str,
    evidence_row: dict[str, Any],
    question_row: dict[str, Any],
) -> dict[str, Any]:
    identity = (
        f"{record['dataset']}\0{record['record_id']}\0{target_category}"
    ).encode("utf-8")
    return {
        "schema_version": "preference_pair_audit_v1",
        "line_number": line_number,
        "pair_id": sha256(identity).hexdigest(),
        "source_name": source.name,
        "source_trace_path": str(trace_path),
        "dataset": record["dataset"],
        "record_id": record["record_id"],
        "transcript_id": record["transcript_id"],
        "segment_id": record["segment_id"],
        "context_scope": source.context_scope,
        "context_turns_before": source.context_turns_before,
        "context_turns_after": source.context_turns_after,
        "target_category": target_category,
        "target_code_label": target_code["code_label"],
        "chosen_question": target_question,
        "rejected_source_category": rejected_category,
        "rejected_code_label": rejected_code["code_label"],
        "rejected_question": rejected_question,
        "category_evidence_row_sha256": _json_sha256(evidence_row),
        "question_only_row_sha256": _json_sha256(question_row),
    }


def _initial_manifest(
    config: PreferencePairConfig, inventories: list[dict[str, Any]]
) -> dict[str, Any]:
    now = _timestamp()
    return {
        "schema_version": "reflective_question_preference_pair_run_v1",
        "created_at_utc": now,
        "python": sys.version,
        "platform": platform.platform(),
        "config": config_to_jsonable(config),
        "random_negative_sampling": {
            "strategy": "secrets.choice",
            "seed": None,
            "reproducible_across_runs": False,
            "same_selection_used_for_both_versions": True,
            "candidate_pool": "the other three aligned same-segment bundles",
        },
        "input_inventory": [
            {
                "name": inventory["source"].name,
                "trace_root": str(inventory["source"].trace_root),
                "manifest_path": str(inventory["source"].manifest_path),
                "manifest_sha256": inventory["manifest_sha256"],
                "manifest_record_count": inventory["manifest"]["record_count"],
                "trace_count": len(inventory["trace_paths"]),
                "missing_trace_count": inventory["missing_trace_count"],
            }
            for inventory in inventories
        ],
        "output_layout": {
            "category_evidence": CATEGORY_EVIDENCE_FILENAME,
            "question_only": QUESTION_ONLY_FILENAME,
            "audit": AUDIT_FILENAME,
        },
        "run_state": {
            "status": "running",
            "started_at_utc": now,
            "updated_at_utc": now,
        },
    }


def _print_summary(run_dir: Path, manifest: dict[str, Any]) -> None:
    print("Preference-pair construction summary", flush=True)
    for source in manifest["source_summaries"]:
        print(
            f"  source={source['name']} manifest_status={source['manifest_status']} "
            f"traces={source['trace_count']} accepted={source['accepted_count']} "
            f"failed_skipped={source['skipped_failed_trace_count']} "
            f"missing={source['missing_trace_count']}",
            flush=True,
        )
    for dataset, counts in manifest["dataset_counts"].items():
        expected = counts["expected_record_count"]
        expected_text = str(expected) if expected else "not-configured"
        print(
            f"  dataset={dataset} expected_records={expected_text} "
            f"traces={counts['trace_count']} "
            f"accepted={counts['accepted_record_count']} "
            f"failed_skipped={counts['skipped_failed_trace_count']} "
            f"pairs_per_version={counts['pair_count']}",
            flush=True,
        )
    counts = manifest["counts"]
    print(
        f"  total accepted_records={counts['accepted_record_count']} "
        f"failed_skipped={counts['skipped_failed_trace_count']} "
        f"missing={counts['missing_trace_count']} "
        f"pairs_per_version={counts['pair_count_per_version']}",
        flush=True,
    )
    print("  rejected-category sampling matrix:", flush=True)
    for category, rejected in manifest["rejection_category_matrix"].items():
        rendered = ", ".join(f"{key}={value}" for key, value in rejected.items())
        print(f"    {category}: {rendered}", flush=True)
    for name, metadata in manifest["output_files"].items():
        print(
            f"  output={name} rows={metadata['row_count']} "
            f"sha256={metadata['sha256']} path={metadata['path']}",
            flush=True,
        )
    print(f"  run_dir={run_dir}", flush=True)


def _system_random_choice(candidates: tuple[int, ...]) -> int:
    return secrets.choice(candidates)


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description.capitalize()} must be a JSON object: {path}")
    return payload


def _required_string(payload: dict[str, Any], field: str, path: Path) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Trace field {field!r} must be non-empty: {path}")
    return value


def _write_jsonl_row(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_sha256(payload: dict[str, Any]) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: Path, row_count: int) -> dict[str, Any]:
    return {
        "path": str(path),
        "row_count": row_count,
        "byte_count": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _new_run_dir(output_root: Path, run_name: str) -> Path:
    base = output_root / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
    if not base.exists():
        return base
    suffix = 1
    while True:
        candidate = base.with_name(f"{base.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
