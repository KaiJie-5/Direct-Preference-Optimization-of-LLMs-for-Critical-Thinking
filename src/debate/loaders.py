from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DatasetConfig
from .schema import CANDIDATE_LABELS, ReviewBlock


@dataclass(frozen=True, slots=True)
class ReviewSegment:
    dataset: str
    transcript_id: str
    segment_id: str
    record_id: str
    candidate_count: int | None
    run_name: str
    segment_json_relative_to_run: str


@dataclass(frozen=True, slots=True)
class CandidateMapping:
    dataset: str
    transcript_id: str
    segment_id: str
    record_id: str
    review_block: str
    candidate_label: str
    original_sample_index: int
    candidate_count: int | None
    run_name: str
    segment_json_relative_to_run: str


@dataclass(frozen=True, slots=True)
class DebateBlockInput:
    segment: ReviewSegment
    review_block: ReviewBlock
    participant_segment_text: str
    previous_context: str
    next_context: str
    research_questions: tuple[str, ...]
    candidate_count: int
    candidate_labels: tuple[str, ...]
    candidate_table: list[dict[str, Any]]
    candidate_mapping: list[dict[str, Any]]
    segment_json_path: Path


def load_review_segments(review_pack_path: Path) -> list[ReviewSegment]:
    rows = _read_csv_dicts(review_pack_path / "review_segments.csv")
    return [
        ReviewSegment(
            dataset=row["dataset"],
            transcript_id=row["transcript_id"],
            segment_id=row["segment_id"],
            record_id=row["record_id"],
            candidate_count=_optional_int(row.get("candidate_count")),
            run_name=row["run_name"],
            segment_json_relative_to_run=row["segment_json_relative_to_run"],
        )
        for row in rows
    ]


def load_candidate_mappings(review_pack_path: Path) -> list[CandidateMapping]:
    rows = _read_csv_dicts(review_pack_path / "internal_candidate_mapping.csv")
    return [
        CandidateMapping(
            dataset=row["dataset"],
            transcript_id=row["transcript_id"],
            segment_id=row["segment_id"],
            record_id=row["record_id"],
            review_block=row["review_block"],
            candidate_label=row["candidate_label"],
            original_sample_index=int(row["original_sample_index"]),
            candidate_count=_optional_int(row.get("candidate_count")),
            run_name=row["run_name"],
            segment_json_relative_to_run=row["segment_json_relative_to_run"],
        )
        for row in rows
    ]


def build_block_inputs(
    *,
    review_pack_path: Path,
    dataset_configs: tuple[DatasetConfig, ...],
    review_blocks: tuple[ReviewBlock, ...],
    limit: int | None = None,
) -> list[DebateBlockInput]:
    segments = load_review_segments(review_pack_path)
    mappings = load_candidate_mappings(review_pack_path)
    dataset_config_by_name = {item.dataset: item for item in dataset_configs}
    review_block_by_id = {item.id: item for item in review_blocks}

    allowed_datasets = set(dataset_config_by_name)
    selected_segments = [
        segment for segment in segments if segment.dataset in allowed_datasets
    ]
    if limit is not None:
        selected_segments = selected_segments[:limit]

    mappings_by_key: dict[tuple[str, str, str], list[CandidateMapping]] = {}
    for mapping in mappings:
        if mapping.dataset not in allowed_datasets:
            continue
        if mapping.review_block not in review_block_by_id:
            continue
        key = (mapping.dataset, mapping.record_id, mapping.review_block)
        mappings_by_key.setdefault(key, []).append(mapping)

    inputs: list[DebateBlockInput] = []
    segment_cache: dict[Path, dict[str, Any]] = {}
    for segment in selected_segments:
        dataset_config = dataset_config_by_name[segment.dataset]
        segment_json_path = resolve_segment_json_path(segment, dataset_config)
        payload = segment_cache.get(segment_json_path)
        if payload is None:
            payload = json.loads(segment_json_path.read_text(encoding="utf-8"))
            segment_cache[segment_json_path] = payload

        segment_candidate_labels: tuple[str, ...] | None = None
        for block in review_blocks:
            key = (segment.dataset, segment.record_id, block.id)
            raw_mappings = mappings_by_key.get(key, [])
            unknown_labels = sorted(
                {
                    mapping.candidate_label
                    for mapping in raw_mappings
                    if mapping.candidate_label not in CANDIDATE_LABELS
                }
            )
            if unknown_labels:
                raise ValueError(
                    f"Unknown candidate labels {unknown_labels} for "
                    f"{segment.dataset} {segment.record_id} {block.id}."
                )
            block_mappings = sorted(
                raw_mappings,
                key=lambda item: CANDIDATE_LABELS.index(item.candidate_label),
            )
            candidate_count = len(block_mappings)
            if candidate_count < 2 or candidate_count > len(CANDIDATE_LABELS):
                raise ValueError(
                    "Expected between two and five candidate mappings for "
                    f"{segment.dataset} {segment.record_id} {block.id}, "
                    f"got {candidate_count}."
                )
            candidate_labels = tuple(
                mapping.candidate_label for mapping in block_mappings
            )
            expected_labels = CANDIDATE_LABELS[:candidate_count]
            if candidate_labels != expected_labels:
                raise ValueError(
                    f"Candidate labels for {segment.dataset} {segment.record_id} "
                    f"{block.id} must be the contiguous prefix {list(expected_labels)}, "
                    f"got {list(candidate_labels)}."
                )
            if segment_candidate_labels is None:
                segment_candidate_labels = candidate_labels
            elif candidate_labels != segment_candidate_labels:
                raise ValueError(
                    f"Candidate labels must be segment-wide for {segment.dataset} "
                    f"{segment.record_id}: expected {list(segment_candidate_labels)}, "
                    f"got {list(candidate_labels)} for {block.id}."
                )
            if (
                segment.candidate_count is not None
                and segment.candidate_count != candidate_count
            ):
                raise ValueError(
                    f"review_segments.csv declares candidate_count="
                    f"{segment.candidate_count} for {segment.dataset} "
                    f"{segment.record_id}, but {block.id} has {candidate_count} mappings."
                )
            declared_mapping_counts = {
                mapping.candidate_count
                for mapping in block_mappings
                if mapping.candidate_count is not None
            }
            if declared_mapping_counts and declared_mapping_counts != {candidate_count}:
                raise ValueError(
                    f"internal_candidate_mapping.csv candidate_count values "
                    f"{sorted(declared_mapping_counts)} do not match {candidate_count} "
                    f"for {segment.dataset} {segment.record_id} {block.id}."
                )
            inconsistent_paths = [
                mapping.candidate_label
                for mapping in block_mappings
                if mapping.run_name != segment.run_name
                or mapping.segment_json_relative_to_run
                != segment.segment_json_relative_to_run
            ]
            if inconsistent_paths:
                raise ValueError(
                    f"Candidate mappings {inconsistent_paths} disagree with the review "
                    f"segment path for {segment.dataset} {segment.record_id} {block.id}."
                )
            inputs.append(
                DebateBlockInput(
                    segment=segment,
                    review_block=block,
                    participant_segment_text=str(payload.get("input_text", "")),
                    previous_context=_metadata_text(payload, "previous_context"),
                    next_context=_metadata_text(payload, "next_context"),
                    research_questions=dataset_config.research_questions,
                    candidate_count=candidate_count,
                    candidate_labels=candidate_labels,
                    candidate_table=[
                        _candidate_payload(payload, mapping, block)
                        for mapping in block_mappings
                    ],
                    candidate_mapping=[
                        {
                            "candidate_label": mapping.candidate_label,
                            "original_sample_index": mapping.original_sample_index,
                            "run_name": mapping.run_name,
                            "segment_json_relative_to_run": (
                                mapping.segment_json_relative_to_run
                            ),
                        }
                        for mapping in block_mappings
                    ],
                    segment_json_path=segment_json_path,
                )
            )
    return inputs


def resolve_segment_json_path(
    segment: ReviewSegment | CandidateMapping,
    dataset_config: DatasetConfig,
) -> Path:
    relative = str(segment.segment_json_relative_to_run).lstrip("\\/")
    relative_path = Path(*relative.replace("\\", "/").split("/"))
    if dataset_config.enriched_run_path is not None:
        return dataset_config.enriched_run_path / relative_path
    if dataset_config.enriched_parent_path is None:
        raise ValueError(
            f"Dataset {dataset_config.dataset!r} needs enriched_parent_path "
            "or enriched_run_path."
        )
    return dataset_config.enriched_parent_path / segment.run_name / relative_path


def _candidate_payload(
    segment_payload: dict[str, Any],
    mapping: CandidateMapping,
    block: ReviewBlock,
) -> dict[str, Any]:
    sample = _sample_by_index(segment_payload, mapping.original_sample_index)
    parsed = sample.get("parsed_output") or {}
    block_payload = _review_block_payload(parsed, block)
    return {
        "candidate_label": mapping.candidate_label,
        "original_sample_index": mapping.original_sample_index,
        "review_block": block.id,
        "fields": {
            field: block_payload.get(field, "")
            for field in block.fields
            if isinstance(block_payload, dict)
        },
    }


def _metadata_text(segment_payload: dict[str, Any], key: str) -> str:
    metadata = segment_payload.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    value = metadata.get(key, "")
    return "" if value is None else str(value)


def _sample_by_index(segment_payload: dict[str, Any], sample_index: int) -> dict[str, Any]:
    for sample in segment_payload.get("samples", []):
        if int(sample.get("sample_index", -1)) == sample_index:
            return sample
    raise ValueError(
        f"Could not find sample_index={sample_index} in {segment_payload.get('record_id')}"
    )


def _review_block_payload(parsed_output: dict[str, Any], block: ReviewBlock) -> dict[str, Any]:
    if block.kind != "code":
        raise ValueError(f"Unsupported review block kind: {block.kind}")
    examples = parsed_output.get("code_quality_examples") or {}
    value = examples.get(block.source_name or "")
    return value if isinstance(value, dict) else {}


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _optional_int(value: str | None) -> int | None:
    return int(value) if value is not None and value.strip() else None
