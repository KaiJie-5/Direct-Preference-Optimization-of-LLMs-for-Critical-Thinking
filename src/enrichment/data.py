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
    "source_html_path",
}

SEGMENT_PROMPT_EXCLUDED_FIELDS = {
    "candidate_example_codes",
    "codebook_id",
    "codebook_version",
    "interview_turns",
}

CONTEXT_SCOPES = {"immediate", "full_interview"}
INTERVIEW_TURN_REQUIRED_FIELDS = {
    "turn_index",
    "speaker",
    "text",
    "paragraph_index",
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

    def to_prompt_vars(
        self,
        codebook: dict[str, Any] | None = None,
        *,
        context_scope: str = "immediate",
    ) -> dict[str, Any]:
        payload = {
            "record_id": self.record_id,
            "text": self.text,
            **{
                key: value
                for key, value in self.metadata.items()
                if key not in SEGMENT_PROMPT_EXCLUDED_FIELDS
            },
            "source": self.source,
        }
        candidate_codes = (
            codebook.get("codes", [])
            if codebook is not None
            else self.metadata.get("candidate_example_codes", [])
        )
        variables = {
            "record_id": self.record_id,
            "input_text": self.text,
            "analysis_context": self.analysis_context(context_scope),
            "segment_json": json.dumps(payload, ensure_ascii=False, indent=2),
            "candidate_example_codes_json": json.dumps(
                candidate_codes, ensure_ascii=False, indent=2
            ),
            "record_json": json.dumps(payload, ensure_ascii=False, indent=2),
        }
        if codebook is not None:
            variables["codebook_id"] = codebook.get("codebook_id", "")
            variables["codebook_version"] = codebook.get("codebook_version", "")
        else:
            variables["codebook_id"] = self.metadata.get("codebook_id", "")
            variables["codebook_version"] = self.metadata.get("codebook_version", "")
        for key, value in self.metadata.items():
            if key in SEGMENT_PROMPT_EXCLUDED_FIELDS:
                continue
            variables[key] = (
                json.dumps(value, ensure_ascii=False, indent=2)
                if isinstance(value, (dict, list))
                else value
            )
        return variables

    def analysis_context(self, context_scope: str = "immediate") -> str:
        if context_scope not in CONTEXT_SCOPES:
            options = ", ".join(sorted(CONTEXT_SCOPES))
            raise ValueError(
                f"context_scope must be one of {options}; got {context_scope!r}."
            )
        if context_scope == "immediate":
            previous = str(self.metadata.get("previous_context", ""))
            following = str(self.metadata.get("next_context", ""))
            return (
                f"Previous context:\n{previous}\n\n"
                f"Next context:\n{following}"
            )
        return self._full_interview_context()

    def _full_interview_context(self) -> str:
        turns = self.metadata.get("interview_turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(
                f"Record {self.record_id} cannot use context_scope='full_interview': "
                "interview_turns is missing or empty. Reprocess the source HTML first."
            )

        validated_turns: list[dict[str, Any]] = []
        seen_indexes: set[int] = set()
        previous_index = 0
        for position, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                raise ValueError(
                    f"Record {self.record_id} interview_turns[{position - 1}] "
                    "must be an object."
                )
            missing = sorted(INTERVIEW_TURN_REQUIRED_FIELDS - set(turn))
            if missing:
                raise ValueError(
                    f"Record {self.record_id} interview_turns[{position - 1}] "
                    f"is missing required fields: {missing}."
                )

            turn_index = turn["turn_index"]
            paragraph_index = turn["paragraph_index"]
            speaker = turn["speaker"]
            text = turn["text"]
            if isinstance(turn_index, bool) or not isinstance(turn_index, int):
                raise ValueError(
                    f"Record {self.record_id} interview turn_index must be an integer."
                )
            if turn_index <= previous_index or turn_index in seen_indexes:
                raise ValueError(
                    f"Record {self.record_id} interview_turns must have unique, "
                    "strictly increasing turn_index values."
                )
            if speaker not in {"interviewer", "participant"}:
                raise ValueError(
                    f"Record {self.record_id} interview turn {turn_index} has "
                    f"unsupported speaker {speaker!r}."
                )
            if not isinstance(text, str):
                raise ValueError(
                    f"Record {self.record_id} interview turn {turn_index} text "
                    "must be a string."
                )
            if isinstance(paragraph_index, bool) or not isinstance(paragraph_index, int):
                raise ValueError(
                    f"Record {self.record_id} interview turn {turn_index} "
                    "paragraph_index must be an integer."
                )
            seen_indexes.add(turn_index)
            previous_index = turn_index
            validated_turns.append(turn)

        target_turn_index = self.metadata.get("turn_index")
        target_turns = [
            turn for turn in validated_turns
            if turn["turn_index"] == target_turn_index
        ]
        if len(target_turns) != 1:
            raise ValueError(
                f"Record {self.record_id} full interview must contain exactly one "
                f"target turn with turn_index={target_turn_index!r}; "
                f"found {len(target_turns)}."
            )
        target_turn = target_turns[0]
        if target_turn["speaker"] != "participant" or target_turn["text"] != self.text:
            raise ValueError(
                f"Record {self.record_id} target interview turn must be the same "
                "participant text as the segment."
            )

        segment_id = str(self.metadata.get("segment_id", ""))
        lines = []
        for turn in validated_turns:
            speaker_label = str(turn["speaker"]).capitalize()
            target_label = (
                f" [TARGET SEGMENT {segment_id}]"
                if turn["turn_index"] == target_turn_index
                else ""
            )
            lines.append(
                f"Turn {turn['turn_index']} | {speaker_label}{target_label}: "
                f"{turn['text']}"
            )
        return "\n".join(lines)


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
