from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from debate.schema import REVIEW_BLOCK_BY_ID
from enrichment.data import DatasetRecord
from enrichment.schema import (
    SAMPLE_SCHEMA_VERSION,
    normalize_code_label,
    validate_segment_enrichment_sample_result,
)

from .schema import CATEGORY_ORDER


@dataclass(frozen=True, slots=True)
class ReflectiveInput:
    dataset: str
    record_id: str
    transcript_id: str
    segment_id: str
    target_segment: str
    research_questions: tuple[str, ...]
    full_interview_context: str
    selected_codes: tuple[dict[str, Any], ...]
    source_segment_path: str

    def jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LoadedInputs:
    records: tuple[ReflectiveInput, ...]
    fingerprint: str
    input_snapshot: dict[str, Any]


def load_reflective_inputs(
    *,
    ranking_run_dir: Path | None = None,
    review_pack_path: Path | None = None,
    single_pass_run_dir: Path | None = None,
    input_mode: str = "ranked",
    input_status_policy: str = "successful_only",
    context_scope: str = "full_interview",
    context_turns_before: int = 20,
    context_turns_after: int = 20,
    snapshot_record_ids: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> LoadedInputs:
    if input_mode == "single_pass":
        if single_pass_run_dir is None:
            raise ValueError("single_pass_run_dir is required for single-pass input.")
        return _load_single_pass_inputs(
            run_dir=single_pass_run_dir,
            input_status_policy=input_status_policy,
            context_scope=context_scope,
            context_turns_before=context_turns_before,
            context_turns_after=context_turns_after,
            snapshot_record_ids=snapshot_record_ids,
            limit=limit,
        )
    if input_mode != "ranked":
        raise ValueError(f"Unsupported reflective input_mode: {input_mode!r}")
    if ranking_run_dir is None or review_pack_path is None:
        raise ValueError(
            "ranking_run_dir and review_pack_path are required for ranked input."
        )
    if snapshot_record_ids is not None:
        raise ValueError("Frozen snapshot IDs are supported only for single-pass input.")
    return _load_ranked_inputs(
        ranking_run_dir=ranking_run_dir,
        review_pack_path=review_pack_path,
        limit=limit,
    )


def _load_ranked_inputs(
    *, ranking_run_dir: Path, review_pack_path: Path, limit: int | None = None
) -> LoadedInputs:
    manifest_path = ranking_run_dir / "run_manifest.json"
    rankings_path = ranking_run_dir / "final_rankings.jsonl"
    mappings_path = review_pack_path / "internal_candidate_mapping.csv"
    manifest = _read_json_object(manifest_path, "ranking manifest")
    run_state = manifest.get("run_state")
    if not isinstance(run_state, dict) or run_state.get("status") != "complete":
        raise ValueError("Debate ranking run_manifest.json must have run_state.status='complete'.")
    questions_by_dataset = _research_questions_by_dataset(manifest)
    ranking_rows = _read_jsonl(rankings_path)
    mapping_rows = _read_csv(mappings_path)
    mappings = _mapping_index(mapping_rows)
    records: list[ReflectiveInput] = []
    seen: set[tuple[str, str]] = set()
    for row in ranking_rows:
        dataset = _required_string(row, "dataset", "ranking row")
        record_id = _required_string(row, "record_id", "ranking row")
        key = (dataset, record_id)
        if key in seen:
            raise ValueError(f"Duplicate ranking row for {dataset} {record_id}.")
        seen.add(key)
        if dataset not in questions_by_dataset:
            raise ValueError(f"No research questions configured for dataset {dataset!r}.")
        _validate_ranking_row(row, dataset=dataset, record_id=record_id)

        trace_path = ranking_run_dir / "debate_traces" / dataset / f"{record_id}.json"
        trace = _read_json_object(trace_path, "debate trace")
        _require_equal(trace.get("dataset"), dataset, trace_path, "dataset")
        _require_equal(trace.get("record_id"), record_id, trace_path, "record_id")
        source_path = Path(_required_string(trace, "segment_json_path", str(trace_path)))
        source = _read_json_object(source_path, "source segment")
        _require_equal(source.get("record_id"), record_id, source_path, "record_id")
        target_segment = _required_string(source, "input_text", str(source_path))
        metadata = source.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Source segment metadata must be an object: {source_path}")
        transcript_id = str(trace.get("transcript_id") or metadata.get("interview_id") or "")
        segment_id = str(trace.get("segment_id") or metadata.get("segment_id") or "")
        if not transcript_id or not segment_id:
            raise ValueError(f"Missing transcript/segment identity in {source_path}.")
        full_context = _render_full_interview(metadata, target_segment, source_path)

        selected: list[dict[str, Any]] = []
        rankings = row["rankings"]
        for category in CATEGORY_ORDER:
            top_label = rankings[category][0]
            mapping_key = (dataset, record_id, category, top_label)
            matches = mappings.get(mapping_key, [])
            if len(matches) != 1:
                raise ValueError(
                    "Expected exactly one candidate mapping for "
                    f"{mapping_key}, got {len(matches)}."
                )
            mapping = matches[0]
            sample_index = int(mapping["original_sample_index"])
            code = _selected_code(source, sample_index, category, source_path)
            selected_item: dict[str, Any] = {
                "hint": category,
                "selected_candidate_label": top_label,
                "original_sample_index": sample_index,
                "code": dict(code),
            }
            model_label = selected_item["code"].get("code_label")
            if isinstance(model_label, str):
                canonical_label = normalize_code_label(model_label)
                if canonical_label != model_label:
                    selected_item["code"]["code_label"] = canonical_label
                    selected_item["canonical_corrections"] = [
                        {
                            "path": "code.code_label",
                            "was_present": True,
                            "model_value": model_label,
                            "canonical_value": canonical_label,
                            "correction_type": "underscore_to_space_code_label",
                        }
                    ]
            selected.append(selected_item)
        records.append(
            ReflectiveInput(
                dataset=dataset,
                record_id=record_id,
                transcript_id=transcript_id,
                segment_id=segment_id,
                target_segment=target_segment,
                research_questions=questions_by_dataset[dataset],
                full_interview_context=full_context,
                selected_codes=tuple(selected),
                source_segment_path=str(source_path),
            )
        )
    if limit is not None:
        records = records[:limit]
    fingerprint_payload = {
        "ranking_manifest_sha256": _file_sha256(manifest_path),
        "final_rankings_sha256": _file_sha256(rankings_path),
        "candidate_mapping_sha256": _file_sha256(mappings_path),
        "resolved_records": [record.jsonable() for record in records],
    }
    fingerprint = sha256(
        json.dumps(
            fingerprint_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return LoadedInputs(
        records=tuple(records),
        fingerprint=fingerprint,
        input_snapshot={
            "input_mode": "ranked",
            "accepted_count": len(records),
            "skipped_count": 0,
        },
    )


def _load_single_pass_inputs(
    *,
    run_dir: Path,
    input_status_policy: str,
    context_scope: str,
    context_turns_before: int,
    context_turns_after: int,
    snapshot_record_ids: tuple[str, ...] | None,
    limit: int | None,
) -> LoadedInputs:
    if input_status_policy != "successful_only":
        raise ValueError("Single-pass reflective input requires successful_only status policy.")
    if context_scope != "turn_window":
        raise ValueError("Single-pass reflective input requires turn_window context.")
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json_object(manifest_path, "single-pass run manifest")
    if manifest.get("schema_version") != "single_pass_enrichment_run_v1":
        raise ValueError(
            "Single-pass source manifest schema_version must be "
            "'single_pass_enrichment_run_v1'."
        )
    execution_fingerprint = _required_string(
        manifest, "execution_fingerprint", str(manifest_path)
    )
    execution_config = manifest.get("execution_config")
    if not isinstance(execution_config, dict):
        raise ValueError("Single-pass source manifest must contain execution_config.")
    if execution_config.get("strategy") != "single_pass":
        raise ValueError("Single-pass source execution_config.strategy must be 'single_pass'.")
    questions = execution_config.get("research_question")
    if not isinstance(questions, list) or not questions or not all(
        isinstance(item, str) and item.strip() for item in questions
    ):
        raise ValueError("Single-pass source must contain non-empty research questions.")

    run_state = manifest.get("run_state")
    if not isinstance(run_state, dict):
        raise ValueError("Single-pass source manifest must contain run_state.")
    counts = {
        name: _required_nonnegative_int(run_state, name, manifest_path)
        for name in ("success_count", "failure_count", "missing_count")
    }
    record_count = _required_nonnegative_int(manifest, "record_count", manifest_path)
    if counts["missing_count"] != 0:
        raise ValueError(
            "Single-pass source is still interrupted: run_state.missing_count must be 0."
        )
    if sum(counts.values()) != record_count:
        raise ValueError(
            "Single-pass source manifest counts do not add up to record_count."
        )

    checkpoint_paths = sorted(run_dir.glob("*_single_pass/segments/*.json"), key=str)
    if len(checkpoint_paths) != record_count:
        raise ValueError(
            "Single-pass checkpoint count does not match manifest record_count: "
            f"expected {record_count}, got {len(checkpoint_paths)}."
        )

    frozen_ids = set(snapshot_record_ids or ())
    if snapshot_record_ids is not None and len(frozen_ids) != len(snapshot_record_ids):
        raise ValueError("Frozen single-pass snapshot contains duplicate record IDs.")
    records: list[ReflectiveInput] = []
    accepted_entries: list[dict[str, str]] = []
    skipped_records: list[dict[str, Any]] = []
    observed_statuses = {"success": 0, "failed": 0}
    seen_ids: set[str] = set()
    for checkpoint_path in checkpoint_paths:
        source = _read_json_object(checkpoint_path, "single-pass checkpoint")
        record_id = _required_string(source, "record_id", str(checkpoint_path))
        if record_id in seen_ids:
            raise ValueError(f"Duplicate single-pass record_id {record_id!r}.")
        seen_ids.add(record_id)
        status = source.get("status")
        if status not in observed_statuses:
            raise ValueError(f"Invalid single-pass status in {checkpoint_path}: {status!r}.")
        observed_statuses[status] += 1

        should_accept = (
            record_id in frozen_ids if snapshot_record_ids is not None else status == "success"
        )
        if should_accept:
            if status != "success":
                raise ValueError(
                    f"Frozen accepted record is no longer successful: {record_id}."
                )
            record = _single_pass_record(
                source=source,
                source_path=checkpoint_path,
                research_questions=tuple(questions),
                context_scope=context_scope,
                context_turns_before=context_turns_before,
                context_turns_after=context_turns_after,
            )
            records.append(record)
            accepted_entries.append(
                {
                    "dataset": record.dataset,
                    "record_id": record.record_id,
                    "record_input_sha256": _record_sha256(record),
                }
            )
        elif snapshot_record_ids is None and status == "failed":
            skipped_records.append(_skipped_record(source, checkpoint_path))

    if observed_statuses["success"] != counts["success_count"] or (
        observed_statuses["failed"] != counts["failure_count"]
    ):
        raise ValueError(
            "Single-pass checkpoint statuses do not match source manifest run_state counts."
        )
    if snapshot_record_ids is not None:
        missing_frozen = sorted(frozen_ids - seen_ids)
        if missing_frozen:
            raise ValueError(
                f"Frozen accepted record is missing from single-pass source: {missing_frozen[0]}."
            )
    elif limit is not None:
        records = records[:limit]
        accepted_entries = accepted_entries[:limit]
    if not records:
        raise ValueError("Single-pass source contains no accepted successful records.")

    fingerprint_payload = {
        "input_mode": "single_pass",
        "source_execution_fingerprint": execution_fingerprint,
        "context_scope": context_scope,
        "context_turns_before": context_turns_before,
        "context_turns_after": context_turns_after,
        "resolved_records": [record.jsonable() for record in records],
    }
    fingerprint = sha256(
        json.dumps(
            fingerprint_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return LoadedInputs(
        records=tuple(records),
        fingerprint=fingerprint,
        input_snapshot={
            "input_mode": "single_pass",
            "source_run_dir": str(run_dir),
            "source_execution_fingerprint": execution_fingerprint,
            "source_manifest_sha256": _file_sha256(manifest_path),
            "source_record_count": record_count,
            "source_success_count": counts["success_count"],
            "source_failure_count": counts["failure_count"],
            "accepted_count": len(records),
            "skipped_count": record_count - len(records),
            "accepted_records": accepted_entries,
            "skipped_records": skipped_records,
        },
    )


def _single_pass_record(
    *,
    source: dict[str, Any],
    source_path: Path,
    research_questions: tuple[str, ...],
    context_scope: str,
    context_turns_before: int,
    context_turns_after: int,
) -> ReflectiveInput:
    if source.get("strategy") != "single_pass":
        raise ValueError(f"Checkpoint strategy must be single_pass: {source_path}")
    record_id = _required_string(source, "record_id", str(source_path))
    target_segment = _required_string(source, "input_text", str(source_path))
    metadata = source.get("metadata")
    source_metadata = source.get("source")
    if not isinstance(metadata, dict) or not isinstance(source_metadata, dict):
        raise ValueError(f"Checkpoint metadata/source must be objects: {source_path}")
    dataset_record = DatasetRecord(
        record_id=record_id,
        text=target_segment,
        metadata=dict(metadata),
        source=dict(source_metadata),
    )
    samples = source.get("samples")
    if not isinstance(samples, list) or len(samples) != 1:
        raise ValueError(f"Single-pass checkpoint must contain exactly one sample: {source_path}")
    selected_index = source.get("selected_sample_index")
    selected_matches = [
        sample
        for sample in samples
        if isinstance(sample, dict) and sample.get("sample_index") == selected_index
    ]
    if len(selected_matches) != 1:
        raise ValueError(f"Single-pass selected sample is missing or ambiguous: {source_path}")
    sample = selected_matches[0]
    selected_json = source.get("selected_json")
    if sample.get("final_parse_status") != "valid" or not isinstance(selected_json, dict):
        raise ValueError(f"Single-pass selected sample is not strictly valid: {source_path}")
    if sample.get("parsed_output") != selected_json:
        raise ValueError(f"Single-pass selected/sample JSON mismatch: {source_path}")
    if source.get("selected_output") != sample.get("output_text"):
        raise ValueError(f"Single-pass selected/sample output mismatch: {source_path}")
    if source.get("context_scope") != context_scope:
        raise ValueError(
            f"Single-pass checkpoint context_scope must be {context_scope!r}: {source_path}"
        )
    codebook_version = _required_string(
        selected_json, "codebook_version", str(source_path)
    )
    validation = validate_segment_enrichment_sample_result(
        selected_json,
        dataset_record,
        expected_codebook_version=codebook_version,
        expected_schema_version=SAMPLE_SCHEMA_VERSION,
        expected_context_scope=context_scope,
        expected_research_questions=research_questions,
        strict_prompt_schema=True,
        allow_target_text_mismatch=False,
    )
    if validation.errors:
        raise ValueError(f"Invalid successful single-pass checkpoint {source_path}: {validation.errors}")

    examples = selected_json.get("code_quality_examples")
    if not isinstance(examples, dict) or set(examples) != set(CATEGORY_ORDER):
        raise ValueError(
            f"Single-pass code categories must exactly match the required set: {source_path}"
        )
    selected: list[dict[str, Any]] = []
    normalized_labels: list[str] = []
    for category in CATEGORY_ORDER:
        code = examples.get(category)
        if not isinstance(code, dict):
            raise ValueError(f"Missing code_quality_examples.{category}: {source_path}")
        block = REVIEW_BLOCK_BY_ID[category]
        missing = [
            field
            for field in block.fields
            if not isinstance(code.get(field), str) or not code[field].strip()
        ]
        if missing:
            raise ValueError(
                f"Selected {category} code in {source_path} has invalid fields: {missing}."
            )
        canonical_code = dict(code)
        label = canonical_code["code_label"]
        canonical_label = normalize_code_label(label)
        canonical_code["code_label"] = canonical_label
        normalized_labels.append(" ".join(canonical_label.split()).casefold())
        item: dict[str, Any] = {
            "hint": category,
            "source_strategy": "single_pass",
            "original_sample_index": selected_index,
            "code": canonical_code,
        }
        if canonical_label != label:
            item["canonical_corrections"] = [
                {
                    "path": "code.code_label",
                    "was_present": True,
                    "model_value": label,
                    "canonical_value": canonical_label,
                    "correction_type": "underscore_to_space_code_label",
                }
            ]
        selected.append(item)
    if len(set(normalized_labels)) != len(CATEGORY_ORDER):
        raise ValueError(f"Single-pass checkpoint has duplicate normalized code labels: {source_path}")

    dataset = _required_string(metadata, "dataset_id", str(source_path))
    transcript_id = _required_string(metadata, "interview_id", str(source_path))
    segment_id = _required_string(metadata, "segment_id", str(source_path))
    _require_equal(source.get("interview_id"), transcript_id, source_path, "interview_id")
    _require_equal(source.get("segment_id"), segment_id, source_path, "segment_id")
    _require_equal(source_path.stem, record_id, source_path, "filename record_id")
    expected_interview_dir = f"{transcript_id}_single_pass"
    _require_equal(
        source_path.parent.parent.name,
        expected_interview_dir,
        source_path,
        "interview directory",
    )
    context = dataset_record.analysis_context(
        context_scope,
        context_turns_before=context_turns_before,
        context_turns_after=context_turns_after,
    )
    return ReflectiveInput(
        dataset=dataset,
        record_id=record_id,
        transcript_id=transcript_id,
        segment_id=segment_id,
        target_segment=target_segment,
        research_questions=research_questions,
        full_interview_context=context,
        selected_codes=tuple(selected),
        source_segment_path=str(source_path),
    )


def _skipped_record(source: dict[str, Any], source_path: Path) -> dict[str, Any]:
    samples = source.get("samples")
    errors: list[str] = []
    if isinstance(samples, list) and samples and isinstance(samples[-1], dict):
        candidate = samples[-1].get("validation_errors")
        if isinstance(candidate, list):
            errors = [str(item) for item in candidate]
    return {
        "record_id": str(source.get("record_id", source_path.stem)),
        "interview_id": str(source.get("interview_id", source_path.parent.parent.name)),
        "status": str(source.get("status", "failed")),
        "validation_errors": errors,
        "source_segment_path": str(source_path),
    }


def _record_sha256(record: ReflectiveInput) -> str:
    return sha256(
        json.dumps(
            record.jsonable(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _required_nonnegative_int(
    payload: dict[str, Any], field: str, path: Path
) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} field {field!r} must be a non-negative integer.")
    return value


def _validate_ranking_row(row: dict[str, Any], *, dataset: str, record_id: str) -> None:
    statuses = row.get("block_status")
    rankings = row.get("rankings")
    if not isinstance(statuses, dict) or not isinstance(rankings, dict):
        raise ValueError(f"Malformed rankings for {dataset} {record_id}.")
    if set(statuses) != set(CATEGORY_ORDER) or set(rankings) != set(CATEGORY_ORDER):
        raise ValueError(
            f"Ranking categories for {dataset} {record_id} must be {list(CATEGORY_ORDER)}."
        )
    candidate_labels = row.get("candidate_labels")
    if (
        not isinstance(candidate_labels, list)
        or not 2 <= len(candidate_labels) <= 5
        or candidate_labels != list("ABCDE"[: len(candidate_labels)])
    ):
        raise ValueError(f"Invalid candidate_labels for {dataset} {record_id}.")
    for category in CATEGORY_ORDER:
        if statuses[category] != "success":
            raise ValueError(f"Ranking {dataset} {record_id} {category} is not successful.")
        ranking = rankings[category]
        if (
            not isinstance(ranking, list)
            or ranking != list(dict.fromkeys(ranking))
            or set(ranking) != set(candidate_labels)
        ):
            raise ValueError(
                f"Ranking {dataset} {record_id} {category} is not a complete permutation."
            )


def _selected_code(
    source: dict[str, Any], sample_index: int, category: str, source_path: Path
) -> dict[str, Any]:
    samples = source.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"Source samples must be a list: {source_path}")
    matches = [sample for sample in samples if sample.get("sample_index") == sample_index]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one sample_index={sample_index} in {source_path}, got {len(matches)}."
        )
    parsed = matches[0].get("parsed_output")
    examples = parsed.get("code_quality_examples") if isinstance(parsed, dict) else None
    code = examples.get(category) if isinstance(examples, dict) else None
    if not isinstance(code, dict):
        raise ValueError(f"Missing code_quality_examples.{category} in {source_path}.")
    block = REVIEW_BLOCK_BY_ID[category]
    missing = [field for field in block.fields if not isinstance(code.get(field), str) or not code[field].strip()]
    if missing:
        raise ValueError(
            f"Selected {category} code in {source_path} has invalid fields: {missing}."
        )
    return dict(code)


def _render_full_interview(
    metadata: dict[str, Any], target_segment: str, source_path: Path
) -> str:
    turns = metadata.get("interview_turns")
    target_index = metadata.get("turn_index")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"Full interview turns are unavailable in {source_path}.")
    lines: list[str] = []
    target_matches = 0
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError(f"Malformed interview turn in {source_path}.")
        index = turn.get("turn_index")
        speaker = str(turn.get("speaker", "unknown"))
        text = str(turn.get("text", ""))
        is_target = index == target_index
        if is_target:
            target_matches += 1
            if text != target_segment:
                raise ValueError(
                    f"Target turn text does not match input_text in {source_path}."
                )
        marker = " [TARGET SEGMENT]" if is_target else ""
        lines.append(f"Turn {index} | {speaker}{marker}: {text}")
    if target_matches != 1:
        raise ValueError(
            f"Expected exactly one target turn in {source_path}, got {target_matches}."
        )
    return "\n".join(lines)


def _research_questions_by_dataset(manifest: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    config = manifest.get("config")
    datasets = config.get("datasets") if isinstance(config, dict) else None
    if not isinstance(datasets, list):
        raise ValueError("Ranking manifest must contain config.datasets.")
    result: dict[str, tuple[str, ...]] = {}
    for item in datasets:
        if not isinstance(item, dict):
            raise ValueError("Ranking manifest dataset entries must be objects.")
        dataset = _required_string(item, "dataset", "ranking manifest dataset")
        questions = item.get("research_questions")
        if not isinstance(questions, list) or not all(
            isinstance(question, str) and question.strip() for question in questions
        ):
            raise ValueError(f"Invalid research questions for dataset {dataset!r}.")
        if dataset in result:
            raise ValueError(f"Duplicate dataset config {dataset!r}.")
        result[dataset] = tuple(questions)
    return result


def _mapping_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    result: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["dataset"], row["record_id"], row["review_block"], row["candidate_label"])
        result.setdefault(key, []).append(row)
    return result


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required {description} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {description}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description.capitalize()} must be a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required ranking JSONL does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}.")
        rows.append(row)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required candidate mapping does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "dataset",
            "record_id",
            "review_block",
            "candidate_label",
            "original_sample_index",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Candidate mapping CSV is missing required columns: {path}"
            )
        return list(reader)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_string(payload: dict[str, Any], field: str, location: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} field {field!r} must be a non-empty string.")
    return value


def _require_equal(actual: Any, expected: Any, path: Path, field: str) -> None:
    if actual != expected:
        raise ValueError(
            f"Identity mismatch in {path} at {field}: expected {expected!r}, got {actual!r}."
        )
