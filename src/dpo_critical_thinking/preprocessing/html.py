from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Turn:
    role: str
    text: str
    paragraph_index: int
    turn_index: int


@dataclass(slots=True)
class SourceInterview:
    interview_id: str
    source_path: Path
    html_path: Path
    source_interview_label: str | None = None


def preprocess_html_dataset(
    *,
    input_path: Path,
    raw_html_dir: Path,
    segments_dir: Path,
    manifest_path: Path,
    interview_id_prefix: str = "INT",
    heading_selector: str = "h2",
    interviewer_selector: str = "p.interviewer",
    participant_selector: str = "p.participant",
    interview_id_source: str = "generated",
    text_normalization: str = "none",
    dataset_id: str | None = None,
    domain: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    _validate_choice(interview_id_source, "interview_id_source", {"generated", "heading"})
    _validate_choice(text_normalization, "text_normalization", {"none", "mojibake"})
    raw_html_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {manifest_path}")

    interviews = _collect_interviews(
        input_path=input_path,
        interview_id_prefix=interview_id_prefix,
        heading_selector=heading_selector,
        interview_id_source=interview_id_source,
        text_normalization=text_normalization,
        overwrite=overwrite,
        raw_html_dir=raw_html_dir,
    )

    manifest_items: list[dict[str, Any]] = []
    for interview in interviews:
        html = interview.html_path.read_text(encoding="utf-8")
        participant_characteristics = _extract_participant_characteristics(html)
        turns = _extract_turns(
            html,
            interviewer_selector=interviewer_selector,
            participant_selector=participant_selector,
        )
        segments = _segments_from_turns(
            turns=turns,
            interview_id=interview.interview_id,
            source_html_path=interview.html_path,
            participant_characteristics=participant_characteristics,
            dataset_id=dataset_id,
            domain=domain,
            source_interview_label=interview.source_interview_label,
            text_normalization=(
                text_normalization if text_normalization != "none" else None
            ),
        )
        segments_path = segments_dir / f"{interview.interview_id}_segments.jsonl"
        if segments_path.exists() and not overwrite:
            raise FileExistsError(f"Segments already exist: {segments_path}")
        _write_jsonl(segments_path, segments)
        item = {
            "interview_id": interview.interview_id,
            "source_path": str(interview.source_path),
            "raw_html_path": str(interview.html_path),
            "segments_path": str(segments_path),
            "segment_count": len(segments),
            "participant_characteristics": participant_characteristics,
        }
        if interview.source_interview_label is not None:
            item["source_interview_label"] = interview.source_interview_label
        manifest_items.append(item)

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_path": str(input_path),
        "raw_html_dir": str(raw_html_dir),
        "segments_dir": str(segments_dir),
        "interviews": manifest_items,
    }
    if interview_id_source != "generated":
        manifest["interview_id_source"] = interview_id_source
    if text_normalization != "none":
        manifest["text_normalization"] = text_normalization
    if dataset_id is not None:
        manifest["dataset_id"] = dataset_id
    if domain is not None:
        manifest["domain"] = domain
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def _collect_interviews(
    *,
    input_path: Path,
    interview_id_prefix: str,
    heading_selector: str,
    interview_id_source: str,
    text_normalization: str,
    overwrite: bool,
    raw_html_dir: Path,
) -> list[SourceInterview]:
    if input_path.is_dir():
        interviews: list[SourceInterview] = []
        html_files = sorted(list(input_path.glob("*.html")) + list(input_path.glob("*.htm")))
        for html_file in html_files:
            interview_id = html_file.stem
            output_html = raw_html_dir / f"{interview_id}.html"
            if output_html.exists() and not overwrite:
                raise FileExistsError(f"Raw HTML already exists: {output_html}")
            if text_normalization == "none":
                shutil.copyfile(html_file, output_html)
            else:
                html = _normalise_text(
                    html_file.read_text(encoding="utf-8"),
                    text_normalization,
                )
                output_html.write_text(html, encoding="utf-8")
            interviews.append(SourceInterview(interview_id, html_file, output_html))
        if not interviews:
            raise FileNotFoundError(f"No HTML files found in {input_path}")
        return interviews

    html = _normalise_text(input_path.read_text(encoding="utf-8"), text_normalization)
    sections = _split_multi_interview_html(html, heading_selector=heading_selector)
    if len(sections) <= 1:
        labels = _heading_labels(html, heading_selector=heading_selector)
        source_label = labels[0] if labels else None
        if interview_id_source == "heading":
            if not source_label:
                raise ValueError("Cannot derive heading interview id without a heading.")
            interview_id = _normalise_interview_id(source_label)
        else:
            interview_id = f"{interview_id_prefix}01"
        output_html = raw_html_dir / f"{interview_id}.html"
        if output_html.exists() and not overwrite:
            raise FileExistsError(f"Raw HTML already exists: {output_html}")
        output_html.write_text(html, encoding="utf-8")
        return [
            SourceInterview(
                interview_id=interview_id,
                source_path=input_path,
                html_path=output_html,
                source_interview_label=(
                    source_label if interview_id_source == "heading" else None
                ),
            )
        ]

    interviews = []
    width = max(2, len(str(len(sections))))
    labels = _heading_labels(html, heading_selector=heading_selector)
    for index, section_html in enumerate(sections, start=1):
        source_label = labels[index - 1] if index <= len(labels) else None
        if interview_id_source == "heading":
            if not source_label:
                raise ValueError(
                    f"Missing heading label for interview section {index}."
                )
            interview_id = _normalise_interview_id(source_label)
        else:
            interview_id = f"{interview_id_prefix}{index:0{width}d}"
        output_html = raw_html_dir / f"{interview_id}.html"
        if output_html.exists() and not overwrite:
            raise FileExistsError(f"Raw HTML already exists: {output_html}")
        output_html.write_text(section_html, encoding="utf-8")
        interviews.append(
            SourceInterview(
                interview_id=interview_id,
                source_path=input_path,
                html_path=output_html,
                source_interview_label=(
                    source_label if interview_id_source == "heading" else None
                ),
            )
        )
    return interviews


def _split_multi_interview_html(html: str, *, heading_selector: str) -> list[str]:
    soup = _beautiful_soup(html)
    headings = soup.select(heading_selector)
    if not headings:
        return [html]

    head_html = str(soup.head) if soup.head else '<head><meta charset="UTF-8"/></head>'
    sections: list[str] = []
    for heading in headings:
        nodes = [heading]
        for sibling in heading.next_siblings:
            if getattr(sibling, "name", None) == heading.name:
                break
            nodes.append(sibling)
        body = "\n".join(str(node) for node in nodes)
        sections.append(
            "<!DOCTYPE html>\n"
            f"<html>{head_html}<body>\n"
            f"{body}\n</body></html>\n"
        )
    return sections


def _heading_labels(html: str, *, heading_selector: str) -> list[str]:
    soup = _beautiful_soup(html)
    return [
        heading.get_text(" ", strip=True)
        for heading in soup.select(heading_selector)
    ]


def _extract_turns(
    html: str,
    *,
    interviewer_selector: str,
    participant_selector: str,
) -> list[Turn]:
    soup = _beautiful_soup(html)
    paragraphs = soup.find_all("p")
    turns: list[Turn] = []
    turn_index = 0
    for paragraph_index, paragraph in enumerate(paragraphs, start=1):
        role: str | None = None
        if _matches_selector(paragraph, interviewer_selector):
            role = "interviewer"
        elif _matches_selector(paragraph, participant_selector):
            role = "participant"
        if role is None:
            continue
        turn_index += 1
        text = paragraph.get_text(" ", strip=True)
        text = _strip_role_prefix(text, role)
        turns.append(
            Turn(
                role=role,
                text=text,
                paragraph_index=paragraph_index,
                turn_index=turn_index,
            )
        )
    return turns


def _extract_participant_characteristics(html: str) -> dict[str, str]:
    soup = _beautiful_soup(html)
    table = soup.find("table")
    if table is None:
        return {}

    characteristics: dict[str, str] = {}
    for row in table.find_all("tr"):
        cells = [
            cell.get_text(" ", strip=True)
            for cell in row.find_all(["th", "td"])
            if cell.get_text(" ", strip=True)
        ]
        if len(cells) < 2:
            continue

        if len(cells) == 2:
            pairs = [(cells[0], cells[1])]
        else:
            pairs = list(zip(cells[0::2], cells[1::2]))

        for label, value in pairs:
            key = _normalise_metadata_key(label)
            if key and value:
                characteristics[key] = value
    return characteristics


def _segments_from_turns(
    *,
    turns: list[Turn],
    interview_id: str,
    source_html_path: Path,
    participant_characteristics: dict[str, str],
    dataset_id: str | None = None,
    domain: str | None = None,
    source_interview_label: str | None = None,
    text_normalization: str | None = None,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    participant_turns = [turn for turn in turns if turn.role == "participant"]
    width = max(3, len(str(len(participant_turns))))

    for segment_index, turn in enumerate(participant_turns, start=1):
        segment_id = f"SEG{segment_index:0{width}d}"
        previous_turn = _neighbor_turn(turns, turn.turn_index, offset=-1)
        next_turn = _neighbor_turn(turns, turn.turn_index, offset=1)
        record = {
            "record_id": f"{interview_id}_{segment_id}",
            "text": turn.text,
            "interview_id": interview_id,
            "segment_id": segment_id,
            "speaker": "participant",
            "turn_index": turn.turn_index,
            "paragraph_index_start": turn.paragraph_index,
            "paragraph_index_end": turn.paragraph_index,
            "line_start": None,
            "line_end": None,
            "previous_context": _format_context(previous_turn),
            "next_context": _format_context(next_turn),
            "participant_characteristics": participant_characteristics,
            "source_html_path": str(source_html_path),
        }
        if dataset_id is not None:
            record["dataset_id"] = dataset_id
        if domain is not None:
            record["domain"] = domain
        if source_interview_label is not None:
            record["source_interview_label"] = source_interview_label
        if text_normalization is not None:
            record["text_normalization"] = text_normalization
        segments.append(record)
    return segments


def _neighbor_turn(turns: list[Turn], turn_index: int, *, offset: int) -> Turn | None:
    wanted = turn_index + offset
    for turn in turns:
        if turn.turn_index == wanted:
            return turn
    return None


def _format_context(turn: Turn | None) -> str:
    if turn is None:
        return ""
    return f"{turn.role.capitalize()}: {turn.text}"


def _strip_role_prefix(text: str, role: str) -> str:
    label = role.capitalize()
    if text.startswith(label):
        return text[len(label) :].lstrip(" :\u00a0")
    return text


def _normalise_interview_id(label: str) -> str:
    chars = []
    previous_was_separator = False
    for char in label.strip():
        if char.isalnum():
            chars.append(char)
            previous_was_separator = False
        elif not previous_was_separator:
            chars.append("_")
            previous_was_separator = True
    interview_id = "".join(chars).strip("_")
    if not interview_id:
        raise ValueError(f"Could not derive interview id from heading {label!r}.")
    return interview_id


def _normalise_text(text: str, mode: str) -> str:
    if mode == "none":
        return text
    if mode != "mojibake":
        raise ValueError(f"Unsupported text normalization mode: {mode}")
    return _repair_mojibake(text)


def _repair_mojibake(text: str) -> str:
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        replacements = {
            "\u00e2\u20ac\u2122": "\u2019",
            "\u00e2\u20ac\u02dc": "\u2018",
            "\u00e2\u20ac\u0153": "\u201c",
            "\u00e2\u20ac\u009d": "\u201d",
            "\u00e2\u20ac\u201d": "\u2014",
            "\u00e2\u20ac\u201c": "\u2013",
            "\u00e2\u20ac\u00a6": "\u2026",
            "\u00c2 ": " ",
            "\u00c2": "",
        }
        repaired = text
        for old, new in replacements.items():
            repaired = repaired.replace(old, new)
        return repaired


def _normalise_metadata_key(label: str) -> str:
    key_chars = []
    previous_was_separator = False
    for char in label.strip().lower():
        if char.isalnum():
            key_chars.append(char)
            previous_was_separator = False
        elif not previous_was_separator:
            key_chars.append("_")
            previous_was_separator = True
    return "".join(key_chars).strip("_")


def _matches_selector(element: Any, selector: str) -> bool:
    if "." not in selector:
        return element.name == selector
    tag, class_name = selector.split(".", 1)
    return element.name == tag and class_name in element.get("class", [])


def _validate_choice(value: str, name: str, allowed: set[str]) -> None:
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of {options}; got {value!r}.")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _beautiful_soup(html: str) -> Any:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "beautifulsoup4 is required for HTML preprocessing. "
            "Install project dependencies from pyproject.toml."
        ) from exc
    return BeautifulSoup(html, "html.parser")
