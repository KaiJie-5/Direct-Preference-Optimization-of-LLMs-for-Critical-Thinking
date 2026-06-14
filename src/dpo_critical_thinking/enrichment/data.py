from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class DatasetRecord:
    """A single enrichment input item."""

    record_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)

    def to_prompt_vars(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "input_text": self.text,
            "record_json": json.dumps(
                {
                    "record_id": self.record_id,
                    "text": self.text,
                    "metadata": self.metadata,
                    "source": self.source,
                },
                ensure_ascii=False,
            ),
            **{f"metadata_{key}": value for key, value in self.metadata.items()},
        }


class _TextOnlyHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self._chunks.append(stripped)

    def text(self) -> str:
        return "\n".join(self._chunks)


def load_records(
    input_path: Path,
    *,
    input_format: str = "auto",
    text_field: str = "text",
    record_id_field: str | None = None,
    html_record_selector: str | None = None,
    html_text_selector: str | None = None,
    html_id_attr: str | None = None,
    limit: int | None = None,
) -> list[DatasetRecord]:
    """Load records from json/jsonl/csv/txt/html without imposing a fixed schema."""

    resolved_format = _resolve_format(input_path, input_format)
    if resolved_format == "jsonl":
        records = _load_jsonl(input_path, text_field, record_id_field)
    elif resolved_format == "json":
        records = _load_json(input_path, text_field, record_id_field)
    elif resolved_format == "csv":
        records = _load_csv(input_path, text_field, record_id_field)
    elif resolved_format == "txt":
        records = [_text_record(input_path, input_path.read_text(encoding="utf-8"))]
    elif resolved_format == "html":
        records = _load_html(
            input_path,
            html_record_selector=html_record_selector,
            html_text_selector=html_text_selector,
            html_id_attr=html_id_attr,
        )
    else:
        raise ValueError(f"Unsupported input format: {resolved_format}")

    if limit is not None:
        return records[:limit]
    return records


def _resolve_format(input_path: Path, input_format: str) -> str:
    if input_format != "auto":
        return input_format.lower()

    suffix = input_path.suffix.lower().lstrip(".")
    if suffix in {"jsonl", "json", "csv", "txt", "html", "htm"}:
        return "html" if suffix == "htm" else suffix
    raise ValueError(
        f"Could not infer input format from {input_path}. Pass --input-format explicitly."
    )


def _load_jsonl(
    input_path: Path, text_field: str, record_id_field: str | None
) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(_record_from_mapping(payload, index, text_field, record_id_field))
    return records


def _load_json(
    input_path: Path, text_field: str, record_id_field: str | None
) -> list[DatasetRecord]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items: Iterable[Any] = payload.get("records", payload.get("data", [payload]))
    else:
        items = payload

    return [
        _record_from_mapping(item, index, text_field, record_id_field)
        for index, item in enumerate(items, start=1)
    ]


def _load_csv(
    input_path: Path, text_field: str, record_id_field: str | None
) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            records.append(_record_from_mapping(row, index, text_field, record_id_field))
    return records


def _load_html(
    input_path: Path,
    *,
    html_record_selector: str | None,
    html_text_selector: str | None,
    html_id_attr: str | None,
) -> list[DatasetRecord]:
    html = input_path.read_text(encoding="utf-8")
    if html_record_selector or html_text_selector:
        return _load_html_with_bs4(
            input_path,
            html,
            html_record_selector=html_record_selector,
            html_text_selector=html_text_selector,
            html_id_attr=html_id_attr,
        )

    parser = _TextOnlyHTMLParser()
    parser.feed(html)
    return [_text_record(input_path, parser.text(), source_format="html")]


def _load_html_with_bs4(
    input_path: Path,
    html: str,
    *,
    html_record_selector: str | None,
    html_text_selector: str | None,
    html_id_attr: str | None,
) -> list[DatasetRecord]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError(
            "Install beautifulsoup4 or omit HTML CSS selectors. "
            "For example: python -m pip install beautifulsoup4"
        ) from exc

    soup = BeautifulSoup(html, "html.parser")
    elements = soup.select(html_record_selector) if html_record_selector else [soup]
    records: list[DatasetRecord] = []

    for index, element in enumerate(elements, start=1):
        text_element = element.select_one(html_text_selector) if html_text_selector else element
        text = text_element.get_text("\n", strip=True) if text_element else ""
        record_id = (
            str(element.get(html_id_attr))
            if html_id_attr and element.get(html_id_attr) is not None
            else f"{input_path.stem}-{index:06d}"
        )
        records.append(
            DatasetRecord(
                record_id=record_id,
                text=text,
                metadata={
                    "html_record_selector": html_record_selector,
                    "html_text_selector": html_text_selector,
                },
                source={"path": str(input_path), "format": "html", "index": index},
            )
        )

    return records


def _record_from_mapping(
    payload: Any, index: int, text_field: str, record_id_field: str | None
) -> DatasetRecord:
    if not isinstance(payload, dict):
        payload = {text_field: str(payload)}

    if text_field not in payload:
        raise KeyError(
            f"Record {index} does not contain text field {text_field!r}. "
            "Use --text-field to choose the correct field."
        )

    record_id = (
        str(payload[record_id_field])
        if record_id_field and record_id_field in payload
        else f"record-{index:06d}"
    )
    metadata = {key: value for key, value in payload.items() if key != text_field}
    return DatasetRecord(
        record_id=record_id,
        text=str(payload[text_field]),
        metadata=metadata,
        source={"format": "structured", "index": index},
    )


def _text_record(
    input_path: Path, text: str, *, source_format: str = "txt"
) -> DatasetRecord:
    return DatasetRecord(
        record_id=input_path.stem,
        text=text,
        metadata={},
        source={"path": str(input_path), "format": source_format, "index": 1},
    )
