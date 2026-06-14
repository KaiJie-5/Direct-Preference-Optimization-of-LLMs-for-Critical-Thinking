from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    """Append-only JSONL logger for manual review and evidence checking."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self.enriched_path = self.output_dir / "enriched_records.jsonl"
        self.failures_path = self.output_dir / "failures.jsonl"

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        (self.output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def event(self, payload: dict[str, Any]) -> None:
        self._append_jsonl(self.events_path, self._with_timestamp(payload))

    def enriched_record(self, payload: dict[str, Any]) -> None:
        self._append_jsonl(self.enriched_path, self._with_timestamp(payload))

    def failure(self, payload: dict[str, Any]) -> None:
        self._append_jsonl(self.failures_path, self._with_timestamp(payload))

    @staticmethod
    def _with_timestamp(payload: dict[str, Any]) -> dict[str, Any]:
        return {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **payload}

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
