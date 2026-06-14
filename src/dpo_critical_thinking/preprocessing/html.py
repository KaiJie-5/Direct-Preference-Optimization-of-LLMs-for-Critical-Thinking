from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .codebook import load_codebook


@dataclass(slots=True)
class Turn:
    role: str
    text: str
    paragraph_index: int
    turn_index: int


def preprocess_html_dataset(
    *,
    input_path: Path,
    raw_html_dir: Path,
    segments_dir: Path,
    manifest_path: Path,
    codebook_path: Path,
    interview_id_prefix: str = "INT",
    heading_selector: str = "h2",
    interviewer_selector: str = "p.interviewer",
    participant_selector: str = "p.participant",
    overwrite: bool = False,
) -> dict[str, Any]:
    codebook = load_codebook(codebook_path)
    raw_html_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {manifest_path}")

    interviews = _collect_interviews(
        input_path=input_path,
        interview_id_prefix=interview_id_prefix,
        heading_selector=heading_selector,
        overwrite=overwrite,
        raw_html_dir=raw_html_dir,
    )

    manifest_items: list[dict[str, Any]] = []
    for interview_id, source_path, html_path in interviews:
        turns = _extract_turns(
            html_path.read_text(encoding="utf-8"),
            interviewer_selector=interviewer_selector,
            participant_selector=participant_selector,
        )
        segments = _segments_from_turns(
            turns=turns,
            interview_id=interview_id,
            source_html_path=html_path,
            codebook=codebook,
        )
        segments_path = segments_dir / f"{interview_id}_segments.jsonl"
        if segments_path.exists() and not overwrite:
            raise FileExistsError(f"Segments already exist: {segments_path}")
        _write_jsonl(segments_path, segments)
        manifest_items.append(
            {
                "interview_id": interview_id,
                "source_path": str(source_path),
                "raw_html_path": str(html_path),
                "segments_path": str(segments_path),
                "segment_count": len(segments),
            }
        )

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_path": str(input_path),
        "raw_html_dir": str(raw_html_dir),
        "segments_dir": str(segments_dir),
        "codebook_path": str(codebook_path),
        "codebook_id": codebook["codebook_id"],
        "codebook_version": codebook["codebook_version"],
        "interviews": manifest_items,
    }
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
    overwrite: bool,
    raw_html_dir: Path,
) -> list[tuple[str, Path, Path]]:
    if input_path.is_dir():
        interviews: list[tuple[str, Path, Path]] = []
        for html_file in sorted(list(input_path.glob("*.html")) + list(input_path.glob("*.htm"))):
            interview_id = html_file.stem
            output_html = raw_html_dir / f"{interview_id}.html"
            if output_html.exists() and not overwrite:
                raise FileExistsError(f"Raw HTML already exists: {output_html}")
            shutil.copyfile(html_file, output_html)
            interviews.append((interview_id, html_file, output_html))
        if not interviews:
            raise FileNotFoundError(f"No HTML files found in {input_path}")
        return interviews

    html = input_path.read_text(encoding="utf-8")
    sections = _split_multi_interview_html(html, heading_selector=heading_selector)
    if len(sections) <= 1:
        interview_id = f"{interview_id_prefix}01"
        output_html = raw_html_dir / f"{interview_id}.html"
        if output_html.exists() and not overwrite:
            raise FileExistsError(f"Raw HTML already exists: {output_html}")
        output_html.write_text(html, encoding="utf-8")
        return [(interview_id, input_path, output_html)]

    interviews = []
    width = max(2, len(str(len(sections))))
    for index, section_html in enumerate(sections, start=1):
        interview_id = f"{interview_id_prefix}{index:0{width}d}"
        output_html = raw_html_dir / f"{interview_id}.html"
        if output_html.exists() and not overwrite:
            raise FileExistsError(f"Raw HTML already exists: {output_html}")
        output_html.write_text(section_html, encoding="utf-8")
        interviews.append((interview_id, input_path, output_html))
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


def _segments_from_turns(
    *,
    turns: list[Turn],
    interview_id: str,
    source_html_path: Path,
    codebook: dict[str, Any],
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    candidate_codes = codebook["codes"]
    participant_turns = [turn for turn in turns if turn.role == "participant"]
    width = max(3, len(str(len(participant_turns))))

    for segment_index, turn in enumerate(participant_turns, start=1):
        segment_id = f"SEG{segment_index:0{width}d}"
        previous_turn = _neighbor_turn(turns, turn.turn_index, offset=-1)
        next_turn = _neighbor_turn(turns, turn.turn_index, offset=1)
        segments.append(
            {
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
                "codebook_id": codebook["codebook_id"],
                "codebook_version": codebook["codebook_version"],
                "candidate_example_codes": candidate_codes,
                "source_html_path": str(source_html_path),
            }
        )
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
        return text[len(label) :].strip()
    return text


def _matches_selector(element: Any, selector: str) -> bool:
    if "." not in selector:
        return element.name == selector
    tag, class_name = selector.split(".", 1)
    return element.name == tag and class_name in element.get("class", [])


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
