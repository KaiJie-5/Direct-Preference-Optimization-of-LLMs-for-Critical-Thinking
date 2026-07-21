from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from enrichment.prompts import PromptTemplate
from enrichment.schema import normalize_code_label
from enrichment.teachers import GenerationOptions, Teacher, build_teacher

from .config import ReflectiveConfig, config_to_jsonable
from .loaders import ReflectiveInput, load_reflective_inputs
from .schema import parse_response_result, required_output, validate_payload


TeacherFactory = Callable[[ReflectiveConfig], Teacher]


def run_reflective_enrichment(
    config: ReflectiveConfig,
    *,
    resume_dir: Path | None = None,
    migrate_label_normalization: bool = False,
    teacher_factory: TeacherFactory | None = None,
) -> Path:
    snapshot_record_ids = (
        _resume_snapshot_record_ids(resume_dir, config)
        if resume_dir is not None
        else None
    )
    loaded = load_reflective_inputs(
        input_mode=config.input_mode,
        ranking_run_dir=config.ranking_run_dir,
        review_pack_path=config.review_pack_path,
        single_pass_run_dir=config.single_pass_run_dir,
        input_status_policy=config.input_status_policy,
        context_scope=config.context_scope,
        context_turns_before=config.context_turns_before,
        context_turns_after=config.context_turns_after,
        snapshot_record_ids=snapshot_record_ids,
        limit=config.limit,
    )
    prompt_template = PromptTemplate(config.prompt_path)
    prompt_sha256 = sha256(prompt_template.template.encode("utf-8")).hexdigest()
    execution_fingerprint = _execution_fingerprint(config, loaded.fingerprint, prompt_sha256)

    is_resume = resume_dir is not None
    if migrate_label_normalization and not is_resume:
        raise ValueError("migrate_label_normalization requires resume_dir.")
    if is_resume:
        if migrate_label_normalization:
            _migrate_label_normalization(
                resume_dir=resume_dir,
                records=loaded.records,
                config=config,
                input_fingerprint=loaded.fingerprint,
                prompt_sha256=prompt_sha256,
                execution_fingerprint=execution_fingerprint,
            )
        run_dir, manifest = _open_resume(
            resume_dir,
            config=config,
            input_fingerprint=loaded.fingerprint,
            prompt_sha256=prompt_sha256,
            execution_fingerprint=execution_fingerprint,
        )
    else:
        run_dir = _new_run_dir(config.output_root, config.run_name)
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "segment_traces").mkdir()
        manifest = _new_manifest(
            config=config,
            record_count=len(loaded.records),
            input_fingerprint=loaded.fingerprint,
            prompt_sha256=prompt_sha256,
            execution_fingerprint=execution_fingerprint,
            input_snapshot=loaded.input_snapshot,
        )
        _write_json(run_dir / "run_manifest.json", manifest)

    checkpoints = _load_checkpoints(
        run_dir=run_dir,
        records=loaded.records,
        execution_fingerprint=execution_fingerprint,
    )
    successful = {
        key: payload for key, payload in checkpoints.items() if payload["status"] == "success"
    }
    remaining = len(loaded.records) - len(successful)
    if is_resume:
        manifest = _mark_resumed(manifest)
        _write_json(run_dir / "run_manifest.json", manifest)
        print(
            f"Resume validation complete: successful={len(successful)} "
            f"retry_or_missing={remaining} total={len(loaded.records)}",
            flush=True,
        )

    teacher = None
    if remaining:
        teacher = (teacher_factory or _default_teacher_factory)(config)
        if "teacher_metadata" not in manifest:
            manifest["teacher_metadata"] = teacher.metadata()
            _write_json(run_dir / "run_manifest.json", manifest)

    for index, record in enumerate(loaded.records, 1):
        key = (record.dataset, record.record_id)
        saved = checkpoints.get(key)
        if saved is not None and saved["status"] == "success":
            continue
        if teacher is None:
            raise RuntimeError("Teacher was not loaded despite remaining records.")
        print(
            f"Reflective enrichment {record.dataset} {record.record_id} "
            f"({index}/{len(loaded.records)})",
            flush=True,
        )
        previous_attempts = list(saved.get("attempts", [])) if saved else []
        checkpoint = _generate_record(
            record=record,
            teacher=teacher,
            prompt_template=prompt_template,
            config=config,
            previous_attempts=previous_attempts,
            execution_fingerprint=execution_fingerprint,
        )
        path = _checkpoint_path(run_dir, record)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, checkpoint)
        checkpoints[key] = checkpoint
        print(f"Completed {record.record_id}: {checkpoint['status']}", flush=True)

    success_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for record in loaded.records:
        checkpoint = checkpoints[(record.dataset, record.record_id)]
        if checkpoint["status"] == "success":
            success_rows.append(checkpoint)
        else:
            failure_rows.append(checkpoint)
    _write_jsonl(run_dir / "reflective_questions.jsonl", success_rows)
    _write_jsonl(run_dir / "failures.jsonl", failure_rows)
    manifest["run_state"].update(
        {
            "status": "complete" if not failure_rows else "incomplete",
            "updated_at_utc": _timestamp(),
            "completed_at_utc": _timestamp(),
            "success_count": len(success_rows),
            "failure_count": len(failure_rows),
        }
    )
    _write_json(run_dir / "run_manifest.json", manifest)
    print(
        f"Reflective enrichment finished: success={len(success_rows)} "
        f"failure={len(failure_rows)} run_dir={run_dir}",
        flush=True,
    )
    return run_dir


def _generate_record(
    *,
    record: ReflectiveInput,
    teacher: Teacher,
    prompt_template: PromptTemplate,
    config: ReflectiveConfig,
    previous_attempts: list[dict[str, Any]],
    execution_fingerprint: str,
) -> dict[str, Any]:
    selected_codes = list(record.selected_codes)
    prompt = _render_prompt(prompt_template, record)
    attempts = list(previous_attempts)
    current_prompt = prompt
    parsed: dict[str, Any] | None = None
    errors: list[str] = []
    sections: dict[str, str] = {
        "reasoning_text": "",
        "reasoning_block": "",
        "json_text": "",
        "reasoning_parse_status": "not_generated",
    }
    validation_warnings: list[str] = []
    json_extraction: dict[str, Any] = {
        "method": "none",
        "recovered_from_format_deviation": False,
        "valid_fenced_object_count": 0,
    }
    output_text = ""
    model_parsed: dict[str, Any] | None = None
    max_attempts = config.generation.json_repair_attempts + 1
    for current_index in range(1, max_attempts + 1):
        is_repair = current_index > 1
        options = _generation_options(config, repair=is_repair)
        try:
            result = teacher.generate(current_prompt, options)
        except Exception as exc:
            attempts.append(
                {
                    "attempt_index": len(attempts) + 1,
                    "current_run_attempt_index": current_index,
                    "is_repair_prompt": is_repair,
                    "generation_options": asdict(options),
                    "status": "operational_failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "prompt_sha256": sha256(current_prompt.encode("utf-8")).hexdigest(),
                }
            )
            errors = [f"{type(exc).__name__}: {exc}"]
            break
        output_text = result.text
        parse_result = parse_response_result(output_text, selected_codes)
        model_parsed = parse_result.model_parsed_output
        parsed = parse_result.parsed_output
        sections = parse_result.sections
        errors = parse_result.errors
        validation_warnings = parse_result.validation_warnings
        json_extraction = parse_result.json_extraction
        attempts.append(
            {
                "attempt_index": len(attempts) + 1,
                "current_run_attempt_index": current_index,
                "is_repair_prompt": is_repair,
                "generation_options": asdict(options),
                "status": "valid" if not errors else "invalid",
                "prompt_sha256": sha256(current_prompt.encode("utf-8")).hexdigest(),
                "rendered_prompt": result.rendered_prompt,
                "raw_output_text": output_text,
                **sections,
                "model_parsed_output": parse_result.model_parsed_output,
                "parsed_output": parsed,
                "canonical_corrections": parse_result.canonical_corrections,
                "validation_errors": errors,
                "validation_warnings": validation_warnings,
                "json_extraction": json_extraction,
                "raw_response": result.raw,
                "elapsed_seconds": result.elapsed_seconds,
            }
        )
        if not errors:
            break
        if current_index < max_attempts:
            current_prompt = _repair_prompt(
                original_prompt=prompt,
                invalid_output=output_text,
                errors=errors,
                expected=required_output(selected_codes),
            )

    status = "success" if parsed is not None and not errors else "failed"
    checkpoint = _checkpoint_base(record, execution_fingerprint)
    checkpoint.update(
        {
            "status": status,
            "updated_at_utc": _timestamp(),
            "prompt_path": str(prompt_template.path),
            "prompt_sha256": sha256(prompt_template.template.encode("utf-8")).hexdigest(),
            "attempt_count": len(attempts),
            "attempts": attempts,
            "raw_output_text": output_text,
            **sections,
            "model_parsed_output": model_parsed,
            "parsed_output": parsed,
            "canonical_corrections": (
                attempts[-1].get("canonical_corrections", []) if attempts else []
            ),
            "validation_errors": errors,
            "validation_warnings": validation_warnings,
            "json_extraction": json_extraction,
            "reflective_questions": (
                parsed["reflective_questions"] if status == "success" else None
            ),
        }
    )
    return checkpoint


def _render_prompt(template: PromptTemplate, record: ReflectiveInput) -> str:
    questions = "\n".join(
        f"{index}. {question}" for index, question in enumerate(record.research_questions, 1)
    )
    selected = list(record.selected_codes)
    return template.render(
        {
            "dataset": record.dataset,
            "record_id": record.record_id,
            "transcript_id": record.transcript_id,
            "segment_id": record.segment_id,
            "research_questions": questions,
            "target_segment": record.target_segment,
            "full_interview_context": record.full_interview_context,
            "selected_codes_json": json.dumps(selected, ensure_ascii=False, indent=2),
            "required_output_json": json.dumps(
                required_output(selected), ensure_ascii=False, indent=2
            ),
        }
    )


def _repair_prompt(
    *, original_prompt: str, invalid_output: str, errors: list[str], expected: dict[str, Any]
) -> str:
    return (
        original_prompt
        + "\n\nYour previous response was invalid. Correct only the response contract while "
        "preserving grounded reasoning. Return a fresh closed <think>...</think> block "
        "followed by exactly one strict JSON object. Do not use triple backticks, "
        "Markdown fences, surrounding prose, multiple objects, or trailing text.\n"
        "Validation errors:\n"
        + "\n".join(f"- {error}" for error in errors)
        + "\nRequired JSON shape and exact code/hint values:\n"
        + json.dumps(expected, ensure_ascii=False, indent=2)
        + "\nPrevious invalid response:\n"
        + invalid_output
    )


def _generation_options(config: ReflectiveConfig, *, repair: bool) -> GenerationOptions:
    generation = config.generation
    return GenerationOptions(
        max_new_tokens=generation.max_new_tokens,
        temperature=0.0 if repair else generation.temperature,
        top_p=1.0 if repair else generation.top_p,
        top_k=generation.top_k,
        repetition_penalty=generation.repetition_penalty,
        seed=generation.seed,
        stop=list(generation.stop) or None,
    )


def _default_teacher_factory(config: ReflectiveConfig) -> Teacher:
    teacher = config.teacher
    payload = asdict(teacher)
    payload["teacher_backend"] = payload.pop("backend")
    args = SimpleNamespace(**payload)
    return build_teacher(args)


def _checkpoint_base(record: ReflectiveInput, execution_fingerprint: str) -> dict[str, Any]:
    return {
        "schema_version": "reflective_question_enrichment_v2",
        "created_at_utc": _timestamp(),
        "execution_fingerprint": execution_fingerprint,
        "record_input_sha256": _record_sha256(record),
        "dataset": record.dataset,
        "record_id": record.record_id,
        "transcript_id": record.transcript_id,
        "segment_id": record.segment_id,
        "target_segment": record.target_segment,
        "research_questions": list(record.research_questions),
        "full_interview_context": record.full_interview_context,
        "source_segment_path": record.source_segment_path,
        "selected_codes": list(record.selected_codes),
    }


def _load_checkpoints(
    *, run_dir: Path, records: tuple[ReflectiveInput, ...], execution_fingerprint: str
) -> dict[tuple[str, str], dict[str, Any]]:
    expected = {_checkpoint_path(run_dir, record): record for record in records}
    trace_dir = run_dir / "segment_traces"
    if not trace_dir.is_dir():
        raise FileNotFoundError(f"Run is missing segment_traces directory: {trace_dir}")
    actual = set(trace_dir.rglob("*.json"))
    unknown = sorted(actual - set(expected), key=str)
    if unknown:
        raise ValueError(f"Unknown checkpoint file in resume run: {unknown[0]}")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(actual, key=str):
        record = expected[path]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read checkpoint {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Checkpoint must be an object: {path}")
        expected_fields = {
            "schema_version": "reflective_question_enrichment_v2",
            "execution_fingerprint": execution_fingerprint,
            "record_input_sha256": _record_sha256(record),
            "dataset": record.dataset,
            "record_id": record.record_id,
        }
        for field, expected_value in expected_fields.items():
            if payload.get(field) != expected_value:
                raise ValueError(
                    f"Checkpoint mismatch in {path} at {field}: expected "
                    f"{expected_value!r}, got {payload.get(field)!r}."
                )
        if payload.get("status") not in {"success", "failed"}:
            raise ValueError(f"Invalid checkpoint status in {path}.")
        if not isinstance(payload.get("attempts"), list):
            raise ValueError(f"Checkpoint attempts must be a list: {path}")
        if payload["status"] == "success":
            parsed = payload.get("parsed_output")
            if not isinstance(parsed, dict):
                raise ValueError(f"Successful checkpoint lacks parsed_output: {path}")
            errors = validate_payload(parsed, list(record.selected_codes))
            if errors:
                raise ValueError(f"Invalid successful checkpoint {path}: {errors}")
        result[(record.dataset, record.record_id)] = payload
    return result


def _new_manifest(
    *,
    config: ReflectiveConfig,
    record_count: int,
    input_fingerprint: str,
    prompt_sha256: str,
    execution_fingerprint: str,
    input_snapshot: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "reflective_question_run_v2",
        "created_at_utc": _timestamp(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": config_to_jsonable(config),
        "record_count": record_count,
        "input_fingerprint": input_fingerprint,
        "prompt_sha256": prompt_sha256,
        "execution_fingerprint": execution_fingerprint,
        "input_snapshot": input_snapshot,
        "output_layout": {
            "segment_traces": "segment_traces/{dataset}/{record_id}.json",
            "final_jsonl": "reflective_questions.jsonl",
            "failures": "failures.jsonl",
        },
        "run_state": {
            "status": "running",
            "started_at_utc": _timestamp(),
            "updated_at_utc": _timestamp(),
            "resume_count": 0,
        },
    }


def _resume_snapshot_record_ids(
    resume_dir: Path, config: ReflectiveConfig
) -> tuple[str, ...] | None:
    if config.input_mode != "single_pass":
        return None
    run_dir = resume_dir.expanduser().resolve()
    path = run_dir / "run_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read resume manifest {path}: {exc}") from exc
    snapshot = manifest.get("input_snapshot") if isinstance(manifest, dict) else None
    if not isinstance(snapshot, dict) or snapshot.get("input_mode") != "single_pass":
        raise ValueError("Single-pass resume manifest is missing its frozen input_snapshot.")
    accepted = snapshot.get("accepted_records")
    if not isinstance(accepted, list) or not accepted:
        raise ValueError("Single-pass resume input_snapshot has no accepted records.")
    record_ids: list[str] = []
    for index, item in enumerate(accepted):
        if not isinstance(item, dict):
            raise ValueError(
                f"Single-pass resume accepted_records[{index}] must be an object."
            )
        record_id = item.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(
                f"Single-pass resume accepted_records[{index}].record_id is invalid."
            )
        record_ids.append(record_id)
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Single-pass resume input_snapshot has duplicate record IDs.")
    if snapshot.get("accepted_count") != len(record_ids):
        raise ValueError("Single-pass resume input_snapshot accepted_count is inconsistent.")
    return tuple(record_ids)


def _open_resume(
    resume_dir: Path,
    *,
    config: ReflectiveConfig,
    input_fingerprint: str,
    prompt_sha256: str,
    execution_fingerprint: str,
) -> tuple[Path, dict[str, Any]]:
    run_dir = resume_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
    path = run_dir / "run_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read resume manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Resume manifest must be an object: {path}")
    expected = {
        "schema_version": "reflective_question_run_v2",
        "input_fingerprint": input_fingerprint,
        "prompt_sha256": prompt_sha256,
        "execution_fingerprint": execution_fingerprint,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"Resume manifest mismatch at {field}: expected {value!r}, "
                f"got {manifest.get(field)!r}."
            )
    saved_config = _resume_relevant_config(manifest.get("config"))
    current_config = _resume_relevant_config(config_to_jsonable(config))
    if saved_config != current_config:
        raise ValueError("Resume config does not match execution-relevant saved config.")
    if not isinstance(manifest.get("run_state"), dict):
        raise ValueError("Resume manifest run_state must be an object.")
    return run_dir, manifest


def _migrate_label_normalization(
    *,
    resume_dir: Path,
    records: tuple[ReflectiveInput, ...],
    config: ReflectiveConfig,
    input_fingerprint: str,
    prompt_sha256: str,
    execution_fingerprint: str,
) -> None:
    run_dir = resume_dir.expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") == "reflective_question_run_v2":
        expected = {
            "input_fingerprint": input_fingerprint,
            "prompt_sha256": prompt_sha256,
            "execution_fingerprint": execution_fingerprint,
        }
        mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
        if mismatches:
            raise ValueError(
                "Existing v2 label migration does not match current inputs at: "
                + ", ".join(mismatches)
            )
        return
    if manifest.get("schema_version") != "reflective_question_run_v1":
        raise ValueError("Label migration only supports reflective run schema v1.")
    if manifest.get("prompt_sha256") != prompt_sha256:
        raise ValueError("Label migration refuses unrelated prompt changes.")
    if _resume_relevant_config(manifest.get("config")) != _resume_relevant_config(
        config_to_jsonable(config)
    ):
        raise ValueError("Label migration refuses unrelated configuration changes.")

    record_by_key = {(record.dataset, record.record_id): record for record in records}
    trace_paths = sorted((run_dir / "segment_traces").rglob("*.json"), key=str)
    if len(trace_paths) != len(record_by_key):
        raise ValueError(
            "Label migration requires one existing checkpoint for every current record."
        )
    loaded: list[tuple[Path, dict[str, Any], ReflectiveInput]] = []
    for path in trace_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (str(payload.get("dataset")), str(payload.get("record_id")))
        record = record_by_key.get(key)
        if record is None:
            raise ValueError(f"Unknown checkpoint during label migration: {path}")
        _verify_label_only_record_migration(payload, record, path)
        loaded.append((path, payload, record))

    backup_dir = run_dir.with_name(
        f"{run_dir.name}_label_normalization_backup_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
    )
    if backup_dir.exists():
        raise FileExistsError(f"Migration backup already exists: {backup_dir}")
    shutil.copytree(run_dir, backup_dir)

    report: dict[str, Any] = {
        "schema_version": "reflective_label_normalization_report_v1",
        "created_at_utc": _timestamp(),
        "run_dir": str(run_dir),
        "backup_dir": str(backup_dir),
        "changed_files": [],
        "repaired_records": [],
        "unresolved_records": [],
        "correction_count": 0,
    }
    checkpoints: list[dict[str, Any]] = []
    for path, payload, record in loaded:
        before_hash = _file_sha256(path)
        migrated = _migrate_checkpoint_payload(
            payload, record=record, execution_fingerprint=execution_fingerprint
        )
        _write_json(path, migrated)
        after_hash = _file_sha256(path)
        report["changed_files"].append(
            {
                "path": str(path.relative_to(run_dir)),
                "before_sha256": before_hash,
                "after_sha256": after_hash,
                "correction_count": sum(
                    len(attempt.get("canonical_corrections", []))
                    for attempt in migrated.get("attempts", [])
                    if isinstance(attempt, dict)
                    and isinstance(attempt.get("canonical_corrections"), list)
                ),
            }
        )
        report["correction_count"] += report["changed_files"][-1][
            "correction_count"
        ]
        destination = (
            report["repaired_records"]
            if migrated["status"] == "success"
            else report["unresolved_records"]
        )
        destination.append({"dataset": record.dataset, "record_id": record.record_id})
        checkpoints.append(migrated)

    manifest["schema_version"] = "reflective_question_run_v2"
    manifest["input_fingerprint"] = input_fingerprint
    manifest["execution_fingerprint"] = execution_fingerprint
    manifest.setdefault("migrations", []).append(
        {
            "type": "code_label_normalization_v1_to_v2",
            "migrated_at_utc": _timestamp(),
            "backup_dir": str(backup_dir),
            "report": "label_normalization_migration_report.json",
        }
    )
    success_count = sum(item["status"] == "success" for item in checkpoints)
    failure_count = len(checkpoints) - success_count
    manifest["run_state"].update(
        {
            "status": "complete" if not failure_count else "incomplete",
            "updated_at_utc": _timestamp(),
            "success_count": success_count,
            "failure_count": failure_count,
        }
    )
    success_rows = [item for item in checkpoints if item["status"] == "success"]
    failure_rows = [item for item in checkpoints if item["status"] != "success"]
    aggregate_paths = (
        run_dir / "reflective_questions.jsonl",
        run_dir / "failures.jsonl",
        manifest_path,
    )
    aggregate_before = {
        path: _file_sha256(path) if path.is_file() else None for path in aggregate_paths
    }
    _write_jsonl(run_dir / "reflective_questions.jsonl", success_rows)
    _write_jsonl(run_dir / "failures.jsonl", failure_rows)
    _write_json(manifest_path, manifest)
    for path in aggregate_paths:
        report["changed_files"].append(
            {
                "path": str(path.relative_to(run_dir)),
                "before_sha256": aggregate_before[path],
                "after_sha256": _file_sha256(path),
            }
        )
    _write_json(run_dir / "label_normalization_migration_report.json", report)


def _verify_label_only_record_migration(
    checkpoint: dict[str, Any], record: ReflectiveInput, path: Path
) -> None:
    if checkpoint.get("schema_version") != "reflective_question_enrichment_v1":
        raise ValueError(f"Expected a v1 checkpoint for migration: {path}")
    old_record = {
        field: checkpoint.get(field)
        for field in (
            "dataset",
            "record_id",
            "transcript_id",
            "segment_id",
            "target_segment",
            "research_questions",
            "full_interview_context",
            "selected_codes",
            "source_segment_path",
        )
    }
    expected_old_hash = sha256(
        json.dumps(
            old_record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if checkpoint.get("record_input_sha256") != expected_old_hash:
        raise ValueError(f"Checkpoint input hash is invalid before migration: {path}")
    current = json.loads(json.dumps(record.jsonable(), ensure_ascii=False))
    old_comparable = deepcopy(old_record)
    new_comparable = deepcopy(current)
    _strip_allowed_label_migration(old_comparable)
    _strip_allowed_label_migration(new_comparable)
    if old_comparable != new_comparable:
        raise ValueError(f"Checkpoint contains changes beyond label normalization: {path}")


def _strip_allowed_label_migration(record: dict[str, Any]) -> None:
    selected = record.get("selected_codes")
    if not isinstance(selected, list):
        return
    for item in selected:
        if not isinstance(item, dict):
            continue
        item.pop("canonical_corrections", None)
        code = item.get("code")
        if isinstance(code, dict) and isinstance(code.get("code_label"), str):
            code["code_label"] = normalize_code_label(code["code_label"])


def _migrate_checkpoint_payload(
    payload: dict[str, Any], *, record: ReflectiveInput, execution_fingerprint: str
) -> dict[str, Any]:
    migrated = deepcopy(payload)
    selected_codes = list(record.selected_codes)
    migrated["schema_version"] = "reflective_question_enrichment_v2"
    migrated["execution_fingerprint"] = execution_fingerprint
    migrated["record_input_sha256"] = _record_sha256(record)
    migrated["selected_codes"] = selected_codes
    valid_attempts: list[dict[str, Any]] = []
    for attempt in migrated.get("attempts", []):
        raw_output = attempt.get("raw_output_text")
        if not isinstance(raw_output, str):
            continue
        previous_status = attempt.get("status")
        previous_errors = deepcopy(attempt.get("validation_errors"))
        old_parsed = deepcopy(
            attempt.get("model_parsed_output", attempt.get("parsed_output"))
        )
        result = parse_response_result(raw_output, selected_codes)
        attempt["model_parsed_output"] = old_parsed
        attempt["parsed_output"] = result.parsed_output
        attempt["canonical_corrections"] = result.canonical_corrections
        attempt["validation_errors"] = result.errors
        attempt["validation_warnings"] = result.validation_warnings
        attempt["json_extraction"] = result.json_extraction
        attempt["status"] = "valid" if not result.errors else "invalid"
        attempt.update(result.sections)
        attempt.setdefault("migrations", []).append(
            {
                "type": "code_label_normalization_v1_to_v2",
                "migrated_at_utc": _timestamp(),
                "previous_status": previous_status,
                "previous_validation_errors": previous_errors,
            }
        )
        if not result.errors and result.parsed_output is not None:
            valid_attempts.append(attempt)
    final_attempt = valid_attempts[-1] if valid_attempts else _last_generated_attempt(migrated)
    if final_attempt is not None:
        for field in (
            "raw_output_text",
            "reasoning_text",
            "reasoning_block",
            "json_text",
            "reasoning_parse_status",
            "model_parsed_output",
            "parsed_output",
            "canonical_corrections",
            "validation_errors",
            "validation_warnings",
            "json_extraction",
        ):
            migrated[field] = deepcopy(final_attempt.get(field))
    migrated["status"] = "success" if valid_attempts else "failed"
    migrated["reflective_questions"] = (
        migrated["parsed_output"]["reflective_questions"]
        if migrated["status"] == "success"
        else None
    )
    migrated["attempt_count"] = len(migrated.get("attempts", []))
    migrated["updated_at_utc"] = _timestamp()
    migrated.setdefault("migrations", []).append(
        {
            "type": "code_label_normalization_v1_to_v2",
            "migrated_at_utc": _timestamp(),
        }
    )
    return migrated


def _last_generated_attempt(payload: dict[str, Any]) -> dict[str, Any] | None:
    attempts = payload.get("attempts")
    if not isinstance(attempts, list):
        return None
    generated = [item for item in attempts if isinstance(item, dict) and "raw_output_text" in item]
    return generated[-1] if generated else None


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resume_relevant_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if key not in {"output_root", "run_name"}}


def _mark_resumed(manifest: dict[str, Any]) -> dict[str, Any]:
    state = manifest["run_state"]
    state["status"] = "running"
    state["updated_at_utc"] = _timestamp()
    state["last_resumed_at_utc"] = _timestamp()
    state["resume_count"] = int(state.get("resume_count", 0)) + 1
    state.pop("completed_at_utc", None)
    return manifest


def _execution_fingerprint(
    config: ReflectiveConfig, input_fingerprint: str, prompt_sha256: str
) -> str:
    payload = {
        "config": _resume_relevant_config(config_to_jsonable(config)),
        "input_fingerprint": input_fingerprint,
        "prompt_sha256": prompt_sha256,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _record_sha256(record: ReflectiveInput) -> str:
    return sha256(
        json.dumps(
            record.jsonable(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _checkpoint_path(run_dir: Path, record: ReflectiveInput) -> Path:
    return run_dir / "segment_traces" / record.dataset / f"{record.record_id}.json"


def _new_run_dir(output_root: Path, run_name: str) -> Path:
    return output_root / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
