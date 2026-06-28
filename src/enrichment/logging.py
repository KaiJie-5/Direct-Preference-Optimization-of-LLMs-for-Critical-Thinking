from __future__ import annotations

import json
import re
import time
from hashlib import sha256
from pathlib import Path
from typing import Any


class RunLogger:
    """Append-only JSONL logger for manual review and evidence checking."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self.enriched_path = self.output_dir / "enriched_records.jsonl"
        self.segments_dir = self.output_dir / "segments"
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

    def enriched_segment(self, record_id: str, payload: dict[str, Any]) -> Path:
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        path = self.segments_dir / f"{record_id}.json"
        path.write_text(
            json.dumps(self._with_timestamp(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def failure(self, payload: dict[str, Any]) -> None:
        self._append_jsonl(self.failures_path, self._with_timestamp(payload))

    def prompt_snapshot(
        self,
        *,
        record_id: str,
        strategy: str,
        step: str,
        sample_index: int,
        rendered_prompt: str,
    ) -> dict[str, Any]:
        prompt_hash = sha256(rendered_prompt.encode("utf-8")).hexdigest()
        prompt_id = (
            f"{record_id}:{strategy}:{step}:{sample_index}:{prompt_hash[:12]}"
        )
        self.event(
            {
                "event": "prompt_prepared",
                "record_id": record_id,
                "strategy": strategy,
                "step": step,
                "sample_index": sample_index,
                "prompt_id": prompt_id,
                "prompt_sha256": prompt_hash,
                "rendered_prompt": rendered_prompt,
            }
        )
        return {"prompt_id": prompt_id, "prompt_sha256": prompt_hash}

    def decode_artifact(
        self,
        *,
        record_id: str,
        strategy: str,
        step: str,
        sample_index: int,
        attempt_index: int,
        raw_text: str,
    ) -> dict[str, Any]:
        artifact_dir = self.output_dir / "decode_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        safe_record_id = re.sub(r"[^A-Za-z0-9_.-]", "_", record_id)
        filename = (
            f"{safe_record_id}_{strategy}_{step}_sample{sample_index}_"
            f"attempt{attempt_index}.txt"
        )
        path = artifact_dir / filename
        path.write_text(raw_text, encoding="utf-8")
        return {
            "raw_decoded_artifact_path": str(path.relative_to(self.output_dir)),
            "raw_decoded_artifact_sha256": sha256(
                raw_text.encode("utf-8")
            ).hexdigest(),
        }

    @staticmethod
    def _with_timestamp(payload: dict[str, Any]) -> dict[str, Any]:
        return {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **payload}

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
