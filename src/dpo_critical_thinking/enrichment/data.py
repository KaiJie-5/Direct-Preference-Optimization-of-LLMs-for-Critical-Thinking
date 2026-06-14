from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SEGMENT_REQUIRED_FIELDS = {
    "record_id",
    "text",
    "interview_id",
    "segment_id",
    "speaker",
    "turn_index",
    "previous_context",
    "next_context",
    "research_focus",
    "codebook_id",
    "codebook_version",
    "candidate_example_codes",
    "source_html_path",
}


@dataclass(slots=True)
class DatasetRecord:
    """A single segment-level enrichment input item."""

    record_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def interview_id(self) -> str:
        return str(self.metadata["interview_id"])

    def to_prompt_vars(self) -> dict[str, Any]:
        payload = {
            "record_id": self.record_id,
            "text": self.text,
            **self.metadata,
            "source": self.source,
        }
        variables = {
            "record_id": self.record_id,
            "input_text": self.text,
            "segment_json": json.dumps(payload, ensure_ascii=False, indent=2),
            "candidate_example_codes_json": json.dumps(
                self.metadata.get("candidate_example_codes", []),
                ensure_ascii=False,
                indent=2,
            ),
            "record_json": json.dumps(payload, ensure_ascii=False, indent=2),
        }
        for key, value in self.metadata.items():
            variables[key] = (
                json.dumps(value, ensure_ascii=False, indent=2)
                if isinstance(value, (dict, list))
                else value
            )
        return variables


def load_segment_records(
    segments_path: Path,
    *,
    limit: int | None = None,
) -> list[DatasetRecord]:
    """Load preprocessed segment-level JSONL records from one file or a directory."""

    records: list[DatasetRecord] = []
    for path in iter_segment_files(segments_path):
        records.extend(_load_segment_jsonl(path))
        if limit is not None and len(records) >= limit:
            return records[:limit]
    return records


def iter_segment_files(segments_path: Path) -> list[Path]:
    if segments_path.is_file():
        if segments_path.suffix.lower() != ".jsonl":
            raise ValueError(f"Segment input file must be .jsonl: {segments_path}")
        return [segments_path]

    if not segments_path.is_dir():
        raise FileNotFoundError(f"Segment path does not exist: {segments_path}")

    files = sorted(segments_path.glob("*_segments.jsonl"))
    if not files:
        files = sorted(segments_path.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No segment JSONL files found in {segments_path}")
    return files


def group_records_by_interview(
    records: list[DatasetRecord],
) -> dict[str, list[DatasetRecord]]:
    grouped: dict[str, list[DatasetRecord]] = {}
    for record in records:
        grouped.setdefault(record.interview_id, []).append(record)
    return grouped


def _load_segment_jsonl(path: Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(_record_from_segment_payload(payload, path, line_index))
    return records


def _record_from_segment_payload(
    payload: Any, path: Path, line_index: int
) -> DatasetRecord:
    if not isinstance(payload, dict):
        raise ValueError(f"{path}:{line_index} must contain a JSON object.")

    missing = sorted(SEGMENT_REQUIRED_FIELDS - set(payload))
    if missing:
        raise ValueError(
            f"{path}:{line_index} is missing required segment fields: {missing}"
        )

    if payload["speaker"] != "participant":
        raise ValueError(
            f"{path}:{line_index} has speaker={payload['speaker']!r}; "
            "segment-level enrichment expects participant turns."
        )

    metadata = dict(payload)
    text = str(metadata.pop("text"))
    record_id = str(metadata.pop("record_id"))
    source = {
        "segments_path": str(path),
        "segments_line": line_index,
        "source_html_path": payload["source_html_path"],
    }
    return DatasetRecord(
        record_id=record_id,
        text=text,
        metadata=metadata,
        source=source,
    )
