from __future__ import annotations

import hashlib
import html
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


UKDA_4688_PROFILE = "ukda-4688"
UKDA_4688_DATASET_ID = "ukda-4688"
UKDA_4688_DOMAIN = "working families, work-life balance, and urban life"
UKDA_4688_EXPECTED_TRANSCRIPT_COUNT = 85
UKDA_4688_NORMALIZATION = "ukda-4688-conservative-v1"

_SPEAKER_PATTERN = re.compile(
    r"(?im)^[ \t]*\*?"
    r"(MALEALE|FEMAIL|FEMALE|MALE|MASLE|IE1|IE2|YO1|IV|IE|Q|I|M|F)"
    r"[ \t]*[:;][ \t]*"
)
_GENERIC_SPEAKER_PATTERN = re.compile(
    r"(?im)^[ \t]*\*?([A-Z][A-Z0-9]{0,15})[ \t]*[:;][ \t]*"
)
_UNCERTAINTY_PAIR = re.compile(
    r"(?<!\w)\?{2,}\s*(\S(?:.*?\S)?)\s*\?{2,}(?!\w)"
)
_UNCERTAINTY_RUN = re.compile(r"\?{2,}")
_UNCLEAR_TAG = re.compile(r"\[unclear\]", re.IGNORECASE)
_FILLER_TOKEN = re.compile(
    r"(?:u+h+|huh|m+|h+m+|erm+|um+|ah+|eh+|yeah+|yea+h*|yep|okay|ok)",
    re.IGNORECASE,
)
_FILLER_SEPARATORS = re.compile(r"[\s,.;!?…'\"()\-]+")

_CANONICAL_SPEAKERS: dict[str, tuple[str, str, bool]] = {
    "Q": ("interviewer", "Interviewer", False),
    "I": ("interviewer", "Interviewer", False),
    "IV": ("interviewer", "Interviewer", False),
    "IE": ("interviewer", "Interviewer", False),
    "MALE": ("participant", "Male", True),
    "M": ("participant", "Male", True),
    "MALEALE": ("participant", "Male", True),
    "MASLE": ("participant", "Male", True),
    "FEMALE": ("participant", "Female", True),
    "F": ("participant", "Female", True),
    "FEMAIL": ("participant", "Female", True),
    "IE1": ("participant", "Incidental", False),
    "IE2": ("participant", "Incidental", False),
    "YO1": ("participant", "Incidental", False),
}

_KNOWN_ARCHIVE_WARNINGS: dict[str, list[str]] = {
    "int001t": ["Interview contains only the female adult participant."],
    "int026t": ["Adult participants were interviewed separately with a brief overlap."],
    "int048t": ["The archive notes that the first part of the interview is missing."],
    "int051t": ["Interview contains only the female adult participant."],
    "int056t": ["Transcript begins mid-discussion and appears truncated."],
    "int069t": ["Telephone interview with the female adult participant only."],
    "int073t": ["The archive notes poor tape quality and missing sound at the start."],
    "int094t": ["Interview contains only the female adult participant."],
    "int095t": ["Telephone interview with the female adult participant only."],
}


@dataclass(slots=True)
class RtfTurn:
    role: str
    speaker_label: str
    raw_speaker_label: str
    text: str
    raw_text: str
    turn_index: int
    paragraph_index: int
    target_eligible: bool
    is_filler_only: bool = False
    is_unclear_only: bool = False
    paired_uncertainty_count: int = 0
    isolated_uncertainty_count: int = 0


@dataclass(slots=True)
class ParsedInterview:
    interview_id: str
    source_path: Path
    extracted_text: str
    participant_characteristics: dict[str, str]
    cover_sheet_text: str
    turns: list[RtfTurn]
    excluded_filler_turns: list[RtfTurn]
    speaker_corrections: dict[str, int]
    warnings: list[str] = field(default_factory=list)


ProfileHandler = Callable[..., dict[str, Any]]
PROFILE_REGISTRY: dict[str, ProfileHandler] = {}


def register_profile(name: str, handler: ProfileHandler) -> None:
    if name in PROFILE_REGISTRY:
        raise ValueError(f"RTF preprocessing profile is already registered: {name}")
    PROFILE_REGISTRY[name] = handler


def preprocess_rtf_dataset(
    *,
    profile: str,
    input_path: Path,
    output_dir: Path,
    strict_inventory: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    try:
        handler = PROFILE_REGISTRY[profile]
    except KeyError as exc:
        options = ", ".join(sorted(PROFILE_REGISTRY)) or "none"
        raise ValueError(
            f"Unknown RTF preprocessing profile {profile!r}; available profiles: {options}."
        ) from exc
    return handler(
        input_path=input_path,
        output_dir=output_dir,
        strict_inventory=strict_inventory,
        overwrite=overwrite,
    )


def preprocess_ukda_4688_dataset(
    *,
    input_path: Path,
    output_dir: Path,
    strict_inventory: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    transcript_dir = input_path / "rtf"
    if not transcript_dir.is_dir():
        raise FileNotFoundError(f"UKDA 4688 transcript directory does not exist: {transcript_dir}")

    transcript_paths = sorted(transcript_dir.glob("int*t.rtf"))
    if not transcript_paths:
        raise FileNotFoundError(f"No UKDA 4688 transcript RTF files found in {transcript_dir}")
    documented_names = _documented_transcript_names(input_path / "4688_file_information.rtf")
    _validate_inventory(
        transcript_paths=transcript_paths,
        documented_names=documented_names,
        strict=strict_inventory,
    )

    source_text_dir = output_dir / "source_text"
    normalized_html_dir = output_dir / "normalized_html"
    segments_dir = output_dir / "segments_jsonl"
    manifest_path = output_dir / "preprocessing_manifest.json"
    qa_path = output_dir / "preprocessing_qa.json"
    if (manifest_path.exists() or qa_path.exists()) and not overwrite:
        raise FileExistsError(
            f"UKDA 4688 preprocessing output already exists in {output_dir}; use --overwrite."
        )
    source_text_dir.mkdir(parents=True, exist_ok=True)
    normalized_html_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    manifest_items: list[dict[str, Any]] = []
    qa_items: list[dict[str, Any]] = []
    aggregate = {
        "source_turn_count": 0,
        "normalized_turn_count": 0,
        "excluded_filler_turn_count": 0,
        "unclear_only_context_turn_count": 0,
        "paired_uncertainty_count": 0,
        "isolated_uncertainty_count": 0,
        "exchange_count": 0,
    }

    for source_path in transcript_paths:
        parsed = _parse_ukda_4688_interview(
            source_path,
            strict_speakers=strict_inventory,
        )
        source_text_path = source_text_dir / f"{parsed.interview_id}.txt"
        normalized_html_path = normalized_html_dir / f"{parsed.interview_id}.html"
        segments_path = segments_dir / f"{parsed.interview_id}_segments.jsonl"
        _ensure_writable_output(source_text_path, overwrite=overwrite)
        _ensure_writable_output(normalized_html_path, overwrite=overwrite)
        _ensure_writable_output(segments_path, overwrite=overwrite)

        source_text_path.write_text(parsed.extracted_text, encoding="utf-8")
        normalized_html = _render_normalized_html(parsed)
        normalized_html_path.write_text(normalized_html, encoding="utf-8")
        segments = _build_exchange_segments(
            parsed,
            source_text_path=source_text_path,
            normalized_html_path=normalized_html_path,
        )
        _write_jsonl(segments_path, segments)

        normalized_turns = [turn for turn in parsed.turns if not turn.is_filler_only]
        qa_item = _qa_item(parsed, segments=segments)
        qa_items.append(qa_item)
        for key in aggregate:
            aggregate[key] += qa_item[key]
        manifest_items.append(
            {
                "interview_id": parsed.interview_id,
                "source_rtf_path": str(source_path),
                "source_text_path": str(source_text_path),
                "normalized_html_path": str(normalized_html_path),
                "segments_path": str(segments_path),
                "source_sha256": _sha256_bytes(source_path.read_bytes()),
                "extracted_text_sha256": _sha256_text(parsed.extracted_text),
                "source_turn_count": len(parsed.turns),
                "normalized_turn_count": len(normalized_turns),
                "segment_count": len(segments),
                "participant_characteristics": parsed.participant_characteristics,
                "warnings": parsed.warnings,
            }
        )

    created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest = {
        "created_utc": created_utc,
        "profile": UKDA_4688_PROFILE,
        "dataset_id": UKDA_4688_DATASET_ID,
        "domain": UKDA_4688_DOMAIN,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "strict_inventory": strict_inventory,
        "documented_transcript_count": len(documented_names),
        "transcript_count": len(transcript_paths),
        "text_normalization": UKDA_4688_NORMALIZATION,
        "normalization_policy": {
            "paired_uncertainty": "[unclear: transcribed words]",
            "isolated_uncertainty": "[unclear]",
            "standalone_fillers": "excluded_from_normalized_turns",
            "unclear_only_participant_turns": "context_only",
            "incidental_speakers": "context_only",
            "target_unit": "question_led_adult_response_exchange",
        },
        "interviews": manifest_items,
    }
    qa_report = {
        "created_utc": created_utc,
        "profile": UKDA_4688_PROFILE,
        "transcript_count": len(transcript_paths),
        **aggregate,
        "interviews": qa_items,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    qa_path.write_text(
        json.dumps(qa_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _parse_ukda_4688_interview(
    source_path: Path,
    *,
    strict_speakers: bool,
) -> ParsedInterview:
    extracted_text = _extract_rtf_text(source_path)
    matches = list(_SPEAKER_PATTERN.finditer(extracted_text))
    if not matches:
        raise ValueError(f"No recognized speaker labels found in {source_path}")

    transcript_start = matches[0].start()
    if strict_speakers:
        unknown = sorted(
            {
                match.group(1).upper()
                for match in _GENERIC_SPEAKER_PATTERN.finditer(extracted_text)
                if match.group(1).upper() not in _CANONICAL_SPEAKERS
            }
        )
        if unknown:
            raise ValueError(
                f"Unknown speaker labels in {source_path.name}: {', '.join(unknown)}"
            )

    cover_sheet_text = extracted_text[:transcript_start].strip()
    participant_characteristics = _parse_cover_sheet(cover_sheet_text)
    turns: list[RtfTurn] = []
    excluded_fillers: list[RtfTurn] = []
    corrections: dict[str, int] = {}
    for index, match in enumerate(matches, start=1):
        raw_label = match.group(1).upper()
        body_start = match.end()
        body_end = matches[index].start() if index < len(matches) else len(extracted_text)
        raw_body = extracted_text[body_start:body_end].strip()
        normalized_body, paired_count, isolated_count = _normalize_turn_text(raw_body)
        role, speaker_label, target_eligible = _CANONICAL_SPEAKERS[raw_label]
        is_filler_only = _is_filler_only(normalized_body)
        is_unclear_only = _is_unclear_only(normalized_body)
        if raw_label not in {"Q", "MALE", "FEMALE", "IE1", "IE2"}:
            corrections[raw_label] = corrections.get(raw_label, 0) + 1
        turn = RtfTurn(
            role=role,
            speaker_label=speaker_label,
            raw_speaker_label=raw_label,
            text=normalized_body,
            raw_text=raw_body,
            turn_index=index,
            paragraph_index=_line_number(extracted_text, match.start()),
            target_eligible=target_eligible and not is_unclear_only,
            is_filler_only=is_filler_only,
            is_unclear_only=is_unclear_only,
            paired_uncertainty_count=paired_count,
            isolated_uncertainty_count=isolated_count,
        )
        turns.append(turn)
        if is_filler_only:
            excluded_fillers.append(turn)

    interview_id = source_path.stem.lower()
    return ParsedInterview(
        interview_id=interview_id,
        source_path=source_path,
        extracted_text=extracted_text,
        participant_characteristics=participant_characteristics,
        cover_sheet_text=cover_sheet_text,
        turns=turns,
        excluded_filler_turns=excluded_fillers,
        speaker_corrections=corrections,
        warnings=list(_KNOWN_ARCHIVE_WARNINGS.get(interview_id, [])),
    )


def _extract_rtf_text(path: Path) -> str:
    try:
        from striprtf.striprtf import rtf_to_text
    except ImportError as exc:
        raise RuntimeError(
            "striprtf==0.0.32 is required for RTF preprocessing. Activate the "
            "project virtual environment and install the project dependencies."
        ) from exc
    raw = path.read_text(encoding="cp1252")
    return rtf_to_text(raw, encoding="cp1252", errors="strict").replace("\r\n", "\n")


def _documented_transcript_names(file_information_path: Path) -> set[str]:
    if not file_information_path.is_file():
        return set()
    text = _extract_rtf_text(file_information_path)
    return {
        f"{match.group(1).lower()}.rtf"
        for match in re.finditer(
            r"(?im)^\s*(int\d+t)\s*[|\t]\s*interview\s+transcript\s*[|\t]?\s*$",
            text,
        )
    }


def _validate_inventory(
    *,
    transcript_paths: list[Path],
    documented_names: set[str],
    strict: bool,
) -> None:
    if not strict:
        return
    actual_names = {path.name.lower() for path in transcript_paths}
    if not documented_names:
        raise ValueError(
            "Strict UKDA 4688 inventory validation requires 4688_file_information.rtf."
        )
    missing = sorted(documented_names - actual_names)
    unexpected = sorted(actual_names - documented_names)
    problems: list[str] = []
    if len(documented_names) != UKDA_4688_EXPECTED_TRANSCRIPT_COUNT:
        problems.append(
            "file information documents "
            f"{len(documented_names)} transcripts instead of "
            f"{UKDA_4688_EXPECTED_TRANSCRIPT_COUNT}"
        )
    if len(actual_names) != UKDA_4688_EXPECTED_TRANSCRIPT_COUNT:
        problems.append(
            f"archive contains {len(actual_names)} transcripts instead of "
            f"{UKDA_4688_EXPECTED_TRANSCRIPT_COUNT}"
        )
    if missing:
        problems.append(f"missing documented files: {', '.join(missing)}")
    if unexpected:
        problems.append(f"unexpected transcript files: {', '.join(unexpected)}")
    if problems:
        raise ValueError("UKDA 4688 strict inventory validation failed: " + "; ".join(problems))


def _normalize_turn_text(text: str) -> tuple[str, int, int]:
    paired_count = 0
    isolated_count = 0
    normalized_lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t\u00a0]+", " ", raw_line).strip()
        if not line:
            continue

        def paired_replacement(match: re.Match[str]) -> str:
            nonlocal paired_count
            paired_count += 1
            return f"[unclear: {match.group(1).strip()}]"

        line = _UNCERTAINTY_PAIR.sub(paired_replacement, line)

        def isolated_replacement(_: re.Match[str]) -> str:
            nonlocal isolated_count
            isolated_count += 1
            return "[unclear]"

        line = _UNCERTAINTY_RUN.sub(isolated_replacement, line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        normalized_lines.append(line)
    return "\n".join(normalized_lines), paired_count, isolated_count


def _is_filler_only(text: str) -> bool:
    if not text.strip():
        return True
    tokens = [token for token in _FILLER_SEPARATORS.split(text) if token]
    return bool(tokens) and all(_FILLER_TOKEN.fullmatch(token) for token in tokens)


def _is_unclear_only(text: str) -> bool:
    without_tags = _UNCLEAR_TAG.sub("", text)
    return not re.search(r"[\w\d]", without_tags, flags=re.UNICODE)


def _parse_cover_sheet(text: str) -> dict[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result: dict[str, str] = {}
    for index, line in enumerate(lines):
        lowered = line.lower()
        if lowered.startswith("pseudonym") and index + 1 < len(lines):
            values = _tab_values(lines[index + 1])
            for key, value in zip(
                ("pseudonym", "interview_date", "city", "suburb_or_urban"),
                values,
            ):
                if value:
                    result[key] = value
        if lowered.startswith("household structure") and index + 1 < len(lines):
            values = _tab_values(lines[index + 1])
            for key, value in zip(
                ("household_structure", "household_composition", "childcare"),
                values,
            ):
                if value:
                    result[key] = value
            occupation_index = 1
            for candidate in lines[index + 2 : index + 6]:
                candidate_values = _tab_values(candidate)
                if not candidate_values or not candidate_values[0].startswith("*"):
                    continue
                occupation = candidate_values[0].lstrip("*").strip()
                if occupation:
                    result[f"occupation_{occupation_index}"] = occupation
                    occupation_index += 1
                if len(candidate_values) > 1 and "children" not in result:
                    household_detail = candidate_values[1].strip()
                    if re.search(r"(?i)\bchild(?:ren)?\b", household_detail):
                        result["children"] = household_detail
                if occupation_index > 2:
                    break
    return result


def _tab_values(line: str) -> list[str]:
    if "|" in line:
        return [value.strip() for value in line.split("|") if value.strip()]
    if "\t" in line:
        return [value.strip() for value in line.split("\t") if value.strip()]
    return [value.strip() for value in re.split(r"\s{2,}", line) if value.strip()]


def _build_exchange_segments(
    interview: ParsedInterview,
    *,
    source_text_path: Path,
    normalized_html_path: Path,
) -> list[dict[str, Any]]:
    normalized_turns = [turn for turn in interview.turns if not turn.is_filler_only]
    interview_turns = [_turn_payload(turn) for turn in normalized_turns]
    exchanges: list[tuple[RtfTurn, list[RtfTurn]]] = []
    current_question: RtfTurn | None = None
    current_responses: list[RtfTurn] = []
    for turn in normalized_turns:
        if turn.role == "interviewer" and not turn.is_unclear_only:
            if current_question is not None and current_responses:
                exchanges.append((current_question, current_responses))
            current_question = turn
            current_responses = []
            continue
        if (
            current_question is not None
            and turn.role == "participant"
            and turn.speaker_label in {"Male", "Female"}
            and turn.target_eligible
        ):
            current_responses.append(turn)
    if current_question is not None and current_responses:
        exchanges.append((current_question, current_responses))

    width = max(3, len(str(len(exchanges))))
    records: list[dict[str, Any]] = []
    for segment_index, (question, responses) in enumerate(exchanges, start=1):
        segment_id = f"SEG{segment_index:0{width}d}"
        target_turn_indexes = [turn.turn_index for turn in responses]
        response_speakers = list(dict.fromkeys(turn.speaker_label for turn in responses))
        target_text = _compose_exchange_target(responses)
        first_response = responses[0]
        last_response = responses[-1]
        next_turn = _next_normalized_turn(normalized_turns, last_response.turn_index)
        records.append(
            {
                "record_id": f"{interview.interview_id}_{segment_id}",
                "text": target_text,
                "interview_id": interview.interview_id,
                "segment_id": segment_id,
                "speaker": "participant",
                "turn_index": first_response.turn_index,
                "paragraph_index_start": min(turn.paragraph_index for turn in responses),
                "paragraph_index_end": max(turn.paragraph_index for turn in responses),
                "line_start": min(turn.paragraph_index for turn in responses),
                "line_end": max(turn.paragraph_index for turn in responses),
                "previous_context": f"Interviewer: {question.text}",
                "next_context": _format_context_turn(next_turn),
                "interview_turns": interview_turns,
                "participant_characteristics": interview.participant_characteristics,
                "interviewer_question": question.text,
                "question_turn_index": question.turn_index,
                "target_turn_indexes": target_turn_indexes,
                "response_speakers": response_speakers,
                "source_rtf_path": str(interview.source_path),
                "source_text_path": str(source_text_path),
                "source_html_path": str(normalized_html_path),
                "dataset_id": UKDA_4688_DATASET_ID,
                "domain": UKDA_4688_DOMAIN,
                "source_interview_label": interview.participant_characteristics.get(
                    "pseudonym", interview.interview_id
                ),
                "text_normalization": UKDA_4688_NORMALIZATION,
                "archive_warnings": interview.warnings,
            }
        )
    return records


def _compose_exchange_target(turns: list[RtfTurn]) -> str:
    return "\n".join(f"{turn.speaker_label}: {turn.text}" for turn in turns)


def _turn_payload(turn: RtfTurn) -> dict[str, Any]:
    return {
        "turn_index": turn.turn_index,
        "speaker": turn.role,
        "text": turn.text,
        "paragraph_index": turn.paragraph_index,
        "speaker_label": turn.speaker_label,
        "raw_speaker_label": turn.raw_speaker_label,
        "target_eligible": turn.target_eligible,
        "is_unclear_only": turn.is_unclear_only,
    }


def _next_normalized_turn(turns: list[RtfTurn], turn_index: int) -> RtfTurn | None:
    for turn in turns:
        if turn.turn_index > turn_index:
            return turn
    return None


def _format_context_turn(turn: RtfTurn | None) -> str:
    if turn is None:
        return ""
    return f"{turn.speaker_label}: {turn.text}"


def _render_normalized_html(interview: ParsedInterview) -> str:
    rows = "\n".join(
        "<tr><th>"
        + html.escape(key.replace("_", " ").title())
        + "</th><td>"
        + html.escape(value)
        + "</td></tr>"
        for key, value in interview.participant_characteristics.items()
    )
    paragraphs = []
    for turn in interview.turns:
        css_class = "excluded-filler" if turn.is_filler_only else turn.role
        paragraphs.append(
            f'<p class="{css_class}" data-turn-index="{turn.turn_index}" '
            f'data-speaker-label="{html.escape(turn.speaker_label)}" '
            f'data-raw-speaker-label="{html.escape(turn.raw_speaker_label)}">'
            f"<strong>{html.escape(turn.speaker_label)}:</strong> "
            f"{html.escape(turn.text).replace(chr(10), '<br/>')}</p>"
        )
    return (
        "<!DOCTYPE html>\n<html><head><meta charset=\"UTF-8\"/>"
        f"<title>{html.escape(interview.interview_id)}</title></head><body>\n"
        f"<h1>{html.escape(interview.interview_id)}</h1>\n"
        f"<table>{rows}</table>\n"
        + "\n".join(paragraphs)
        + "\n</body></html>\n"
    )


def _qa_item(interview: ParsedInterview, *, segments: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_turns = [turn for turn in interview.turns if not turn.is_filler_only]
    return {
        "interview_id": interview.interview_id,
        "source_turn_count": len(interview.turns),
        "normalized_turn_count": len(normalized_turns),
        "excluded_filler_turn_count": len(interview.excluded_filler_turns),
        "unclear_only_context_turn_count": sum(
            turn.is_unclear_only for turn in normalized_turns
        ),
        "paired_uncertainty_count": sum(
            turn.paired_uncertainty_count for turn in interview.turns
        ),
        "isolated_uncertainty_count": sum(
            turn.isolated_uncertainty_count for turn in interview.turns
        ),
        "exchange_count": len(segments),
        "speaker_corrections": interview.speaker_corrections,
        "warnings": interview.warnings,
    }


def _ensure_writable_output(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")


def _line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


register_profile(UKDA_4688_PROFILE, preprocess_ukda_4688_dataset)
