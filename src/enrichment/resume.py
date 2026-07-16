from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from .data import DatasetRecord
from .prompts import PromptTemplate
from .schema import SAMPLE_SCHEMA_VERSION, validate_segment_enrichment_sample_result
from .strategies import render_single_pass_prompt


RUN_MANIFEST_SCHEMA_VERSION = "single_pass_enrichment_run_v1"
CHECKPOINT_SCHEMA_VERSION = "single_pass_enrichment_checkpoint_v1"


@dataclass(frozen=True, slots=True)
class CorruptCheckpoint:
    path: Path
    error: str


@dataclass(slots=True)
class ResumeAudit:
    run_dir: Path
    manifest: dict[str, Any]
    checkpoints: dict[str, dict[str, Any]]
    corrupt: list[CorruptCheckpoint]
    legacy_migration: bool
    total_count: int
    success_count: int
    failed_count: int
    missing_count: int

    @property
    def retry_or_missing_count(self) -> int:
        return self.total_count - self.success_count


def record_fingerprint(record: DatasetRecord) -> str:
    return _json_sha256(
        {
            "record_id": record.record_id,
            "text": record.text,
            "metadata": record.metadata,
            "source": record.source,
        }
    )


def build_run_identity(
    *,
    records: list[DatasetRecord],
    execution_config: dict[str, Any],
    prompt: PromptTemplate,
    codebook: dict[str, Any] | None,
) -> dict[str, Any]:
    record_hashes = {record.record_id: record_fingerprint(record) for record in records}
    input_fingerprint = _json_sha256(record_hashes)
    prompt_sha256 = sha256(prompt.template.encode("utf-8")).hexdigest()
    codebook_fingerprint = _json_sha256(codebook) if codebook is not None else None
    fingerprint_payload = {
        "execution_config": execution_config,
        "input_fingerprint": input_fingerprint,
        "prompt_sha256": prompt_sha256,
        "codebook_fingerprint": codebook_fingerprint,
    }
    return {
        **fingerprint_payload,
        "execution_fingerprint": _json_sha256(fingerprint_payload),
        "record_fingerprints": record_hashes,
    }


def new_run_manifest(
    *,
    identity: dict[str, Any],
    output_dir: Path,
    record_count: int,
) -> dict[str, Any]:
    now = _timestamp()
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": now,
        "python": sys.version,
        "platform": platform.platform(),
        "output_dir": str(output_dir),
        "record_count": record_count,
        **identity,
        "run_state": {
            "status": "running",
            "started_at_utc": now,
            "updated_at_utc": now,
            "resume_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "missing_count": record_count,
        },
    }


def audit_resume(
    *,
    run_dir: Path,
    records: list[DatasetRecord],
    identity: dict[str, Any],
    prompt: PromptTemplate,
    prompt_vars: dict[str, Any],
    codebook: dict[str, Any] | None,
    research_questions: list[str],
    context_scope: str,
    context_turns_before: int,
    context_turns_after: int,
    generation_options: dict[str, Any],
    current_args: dict[str, Any],
    exclusion_summary: dict[str, Any],
    model_prompt_renderer: Callable[[str], str] | None = None,
) -> ResumeAudit:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")

    root_manifest_path = run_dir / "run_manifest.json"
    legacy = not root_manifest_path.is_file()
    if legacy:
        _validate_legacy_run_metadata(
            run_dir=run_dir,
            current_args=current_args,
            exclusion_summary=exclusion_summary,
        )
        manifest = new_run_manifest(
            identity=identity,
            output_dir=run_dir,
            record_count=len(records),
        )
        manifest["legacy_migration"] = {
            "status": "validated_pending_write",
            "validated_at_utc": _timestamp(),
            "historical_unprocessed_input_fingerprint_available": False,
            "validation_basis": [
                "saved_interview_manifests",
                "saved_segment_checkpoints",
                "saved_rendered_prompt_hashes",
                "current_complete_input_fingerprint",
            ],
        }
    else:
        manifest = _read_json_object(root_manifest_path, "resume run manifest")
        _validate_root_manifest(manifest, identity, record_count=len(records))

    legacy_checkpoint_format = legacy or "legacy_migration" in manifest
    expected_paths = {
        _checkpoint_path(run_dir, record): record for record in records
    }
    actual_paths = set(run_dir.glob("*_single_pass/segments/*.json"))
    unknown = sorted(actual_paths - set(expected_paths), key=str)
    if unknown:
        raise ValueError(f"Unknown checkpoint file in resume run: {unknown[0]}")

    checkpoints: dict[str, dict[str, Any]] = {}
    corrupt: list[CorruptCheckpoint] = []
    for path in sorted(actual_paths, key=str):
        record = expected_paths[path]
        try:
            payload = _read_json_object(path, "segment checkpoint")
        except ValueError as exc:
            corrupt.append(CorruptCheckpoint(path=path, error=str(exc)))
            continue
        _validate_checkpoint(
            path=path,
            payload=payload,
            record=record,
            execution_fingerprint=identity["execution_fingerprint"],
            legacy=legacy_checkpoint_format,
            prompt=prompt,
            prompt_vars=prompt_vars,
            codebook=codebook,
            research_questions=research_questions,
            context_scope=context_scope,
            context_turns_before=context_turns_before,
            context_turns_after=context_turns_after,
            generation_options=generation_options,
            model_prompt_renderer=model_prompt_renderer,
        )
        checkpoints[record.record_id] = payload

    success_count = sum(
        payload.get("status") == "success" for payload in checkpoints.values()
    )
    failed_count = sum(
        payload.get("status") == "failed" for payload in checkpoints.values()
    )
    missing_count = len(records) - len(checkpoints)
    update_manifest_state(
        manifest,
        total_count=len(records),
        checkpoints=checkpoints,
        status="validated",
    )
    return ResumeAudit(
        run_dir=run_dir,
        manifest=manifest,
        checkpoints=checkpoints,
        corrupt=corrupt,
        legacy_migration=legacy,
        total_count=len(records),
        success_count=success_count,
        failed_count=failed_count,
        missing_count=missing_count,
    )


def checkpoint_metadata(
    *, execution_fingerprint: str, record: DatasetRecord
) -> dict[str, str]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "execution_fingerprint": execution_fingerprint,
        "record_input_sha256": record_fingerprint(record),
    }


def mark_manifest_resumed(manifest: dict[str, Any], *, legacy: bool) -> None:
    state = manifest["run_state"]
    state.update(
        {
            "status": "running",
            "updated_at_utc": _timestamp(),
            "last_resumed_at_utc": _timestamp(),
            "resume_count": int(state.get("resume_count", 0)) + 1,
        }
    )
    state.pop("completed_at_utc", None)
    if legacy:
        manifest["legacy_migration"]["status"] = "migrated"
        manifest["legacy_migration"]["migrated_at_utc"] = _timestamp()


def update_manifest_state(
    manifest: dict[str, Any],
    *,
    total_count: int,
    checkpoints: dict[str, dict[str, Any]],
    status: str,
) -> None:
    successes = sum(item.get("status") == "success" for item in checkpoints.values())
    failures = sum(item.get("status") == "failed" for item in checkpoints.values())
    missing = total_count - len(checkpoints)
    state = manifest.setdefault("run_state", {})
    state.update(
        {
            "status": status,
            "updated_at_utc": _timestamp(),
            "success_count": successes,
            "failure_count": failures,
            "missing_count": missing,
            "retry_or_missing_count": total_count - successes,
        }
    )
    if status in {"complete", "incomplete"}:
        state["completed_at_utc"] = _timestamp()


def write_run_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    _write_json_atomic(run_dir / "run_manifest.json", manifest)


def archive_corrupt_checkpoints(
    run_dir: Path, corrupt: list[CorruptCheckpoint]
) -> list[Path]:
    archived: list[Path] = []
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    for item in corrupt:
        interview_dir = item.path.parents[1].name
        destination_dir = run_dir / "resume_invalid_checkpoints" / interview_dir
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{stamp}_{item.path.name}"
        if destination.exists():
            destination = destination_dir / f"{stamp}_{time.time_ns()}_{item.path.name}"
        os.replace(item.path, destination)
        archived.append(destination)
        artifact_dir = item.path.parents[1] / "decode_artifacts"
        if artifact_dir.is_dir():
            for artifact in sorted(
                artifact_dir.glob(f"{item.path.stem}_single_pass_*"), key=str
            ):
                artifact_destination = destination_dir / (
                    f"{stamp}_artifact_{artifact.name}"
                )
                if artifact_destination.exists():
                    artifact_destination = destination_dir / (
                        f"{stamp}_{time.time_ns()}_artifact_{artifact.name}"
                    )
                os.replace(artifact, artifact_destination)
                archived.append(artifact_destination)
    return archived


def _validate_root_manifest(
    manifest: dict[str, Any], identity: dict[str, Any], *, record_count: int
) -> None:
    expected = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "record_count": record_count,
        "input_fingerprint": identity["input_fingerprint"],
        "prompt_sha256": identity["prompt_sha256"],
        "codebook_fingerprint": identity["codebook_fingerprint"],
        "execution_fingerprint": identity["execution_fingerprint"],
        "execution_config": identity["execution_config"],
        "record_fingerprints": identity["record_fingerprints"],
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"Resume manifest mismatch at {field}: expected {value!r}, "
                f"got {manifest.get(field)!r}."
            )
    if not isinstance(manifest.get("run_state"), dict):
        raise ValueError("Resume manifest run_state must be an object.")


def _validate_legacy_run_metadata(
    *,
    run_dir: Path,
    current_args: dict[str, Any],
    exclusion_summary: dict[str, Any],
) -> None:
    command_path = run_dir / "command.txt"
    if not command_path.is_file():
        raise FileNotFoundError(f"Legacy resume run is missing command.txt: {command_path}")
    manifests = sorted(run_dir.glob("*_single_pass/run_manifest.json"), key=str)
    if not manifests:
        raise FileNotFoundError(
            f"Legacy resume run has no per-interview manifests: {run_dir}"
        )
    relevant = {
        "segments_path",
        "exclude_records_path",
        "codebook_path",
        "limit",
        "context_scope",
        "context_turns_before",
        "context_turns_after",
        "strategy",
        "prompt_path",
        "prompt_var",
        "research_question",
        "teacher_backend",
        "model_path",
        "model_name",
        "torch_dtype",
        "device_map",
        "trust_remote_code",
        "use_chat_template",
        "force_think_prefix",
        "think_prefix",
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "seed",
        "stop",
    }
    if current_args.get("context_scope") != "turn_window":
        relevant -= {"context_turns_before", "context_turns_after"}
    expected_args = {key: current_args.get(key) for key in relevant}
    for path in manifests:
        payload = _read_json_object(path, "legacy interview manifest")
        saved_args = payload.get("args")
        if not isinstance(saved_args, dict):
            raise ValueError(f"Legacy manifest args must be an object: {path}")
        observed = {key: saved_args.get(key) for key in relevant}
        if observed != expected_args:
            differences = sorted(
                key for key in relevant if observed.get(key) != expected_args.get(key)
            )
            raise ValueError(
                f"Legacy resume config mismatch in {path}: {differences}."
            )
        if payload.get("output_schema_version") != SAMPLE_SCHEMA_VERSION:
            raise ValueError(f"Legacy output schema mismatch in {path}.")
        saved_exclusion = payload.get("exclusion_filter")
        if not isinstance(saved_exclusion, dict) or saved_exclusion.get("sha256") != (
            exclusion_summary.get("sha256")
        ):
            raise ValueError(f"Legacy exclusion fingerprint mismatch in {path}.")


def _validate_checkpoint(
    *,
    path: Path,
    payload: dict[str, Any],
    record: DatasetRecord,
    execution_fingerprint: str,
    legacy: bool,
    prompt: PromptTemplate,
    prompt_vars: dict[str, Any],
    codebook: dict[str, Any] | None,
    research_questions: list[str],
    context_scope: str,
    context_turns_before: int,
    context_turns_after: int,
    generation_options: dict[str, Any],
    model_prompt_renderer: Callable[[str], str] | None,
) -> None:
    expected = {
        "record_id": record.record_id,
        "interview_id": record.interview_id,
        "segment_id": record.metadata.get("segment_id"),
        "input_text": record.text,
        "metadata": record.metadata,
        "source": record.source,
        "strategy": "single_pass",
        "context_scope": context_scope,
    }
    if context_scope == "turn_window":
        expected.update(
            {
                "context_turns_before": context_turns_before,
                "context_turns_after": context_turns_after,
            }
        )
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"Checkpoint mismatch in {path} at {field}: expected "
                f"{value!r}, got {payload.get(field)!r}."
            )
    if not legacy or payload.get("checkpoint_schema_version") is not None:
        checkpoint_expected = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "execution_fingerprint": execution_fingerprint,
            "record_input_sha256": record_fingerprint(record),
        }
        for field, value in checkpoint_expected.items():
            if payload.get(field) != value:
                raise ValueError(f"Checkpoint mismatch in {path} at {field}.")

    status = payload.get("status")
    if status not in {"success", "failed"}:
        raise ValueError(f"Invalid checkpoint status in {path}: {status!r}.")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != 1:
        raise ValueError(f"Checkpoint must contain exactly one sample: {path}")
    sample = samples[0]
    attempts = sample.get("attempts") if isinstance(sample, dict) else None
    if not isinstance(attempts, list) or not attempts:
        raise ValueError(f"Checkpoint attempts must be a non-empty list: {path}")
    if sample.get("attempt_count") != len(attempts):
        raise ValueError(f"Checkpoint attempt_count mismatch: {path}")
    for index, attempt in enumerate(attempts, 1):
        if attempt.get("attempt_index") != index:
            raise ValueError(f"Checkpoint attempt indexes are not contiguous: {path}")
        if attempt.get("generation_options") != generation_options:
            raise ValueError(f"Checkpoint generation settings mismatch: {path}")

    if legacy:
        if model_prompt_renderer is None:
            raise ValueError("Legacy checkpoint validation requires a prompt renderer.")
        base_prompt = render_single_pass_prompt(
            record=record,
            prompt=prompt,
            prompt_vars=prompt_vars,
            codebook=codebook,
            context_scope=context_scope,
            context_turns_before=context_turns_before,
            context_turns_after=context_turns_after,
        )
        prompt_hash = sha256(model_prompt_renderer(base_prompt).encode("utf-8")).hexdigest()
        for attempt in attempts:
            if attempt.get("prompt_sha256") != prompt_hash:
                raise ValueError(f"Legacy rendered prompt hash mismatch: {path}")
            if attempt.get("attempt_prompt_sha256") != prompt_hash:
                raise ValueError(f"Legacy attempt prompt hash mismatch: {path}")

    if status == "success":
        selected = payload.get("selected_json")
        if not isinstance(selected, dict):
            raise ValueError(f"Successful checkpoint lacks selected_json: {path}")
        result = validate_segment_enrichment_sample_result(
            selected,
            record,
            expected_codebook_version=_codebook_version(record, codebook),
            expected_schema_version=SAMPLE_SCHEMA_VERSION,
            expected_context_scope=context_scope,
            expected_research_questions=research_questions,
            strict_prompt_schema=True,
            allow_target_text_mismatch=False,
        )
        if result.errors:
            raise ValueError(f"Invalid successful checkpoint {path}: {result.errors}")
        if sample.get("final_parse_status") != "valid":
            raise ValueError(f"Successful checkpoint parse status is not valid: {path}")
        if sample.get("parsed_output") != selected:
            raise ValueError(f"Successful checkpoint selected/sample JSON mismatch: {path}")
        if payload.get("selected_output") != sample.get("output_text"):
            raise ValueError(f"Successful checkpoint selected output mismatch: {path}")
    elif sample.get("final_parse_status") != "invalid":
        raise ValueError(f"Failed checkpoint parse status is not invalid: {path}")


def _checkpoint_path(run_dir: Path, record: DatasetRecord) -> Path:
    return (
        run_dir
        / f"{record.interview_id}_single_pass"
        / "segments"
        / f"{record.record_id}.json"
    )


def _codebook_version(
    record: DatasetRecord, codebook: dict[str, Any] | None
) -> str | None:
    value = (
        codebook.get("codebook_version")
        if codebook is not None
        else record.metadata.get("codebook_version")
    )
    return str(value) if value is not None else None


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description.capitalize()} must be an object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _json_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
