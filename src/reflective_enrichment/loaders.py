from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from debate.schema import REVIEW_BLOCK_BY_ID
from enrichment.schema import normalize_code_label

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


def load_reflective_inputs(
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
    return LoadedInputs(records=tuple(records), fingerprint=fingerprint)


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
