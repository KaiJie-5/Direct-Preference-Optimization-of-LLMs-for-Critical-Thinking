from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path
from typing import Any, BinaryIO


DATASET_VERSIONS = ("category_evidence", "question_only")
SOURCE_FILENAMES = {
    "category_evidence": "preference_pairs_category_evidence.jsonl",
    "question_only": "preference_pairs_question_only.jsonl",
    "audit": "preference_pair_audit.jsonl",
}
OUTPUT_FILENAMES = {
    split: {
        "category_evidence": f"{split}_preference_pairs_category_evidence.jsonl",
        "question_only": f"{split}_preference_pairs_question_only.jsonl",
        "audit": f"{split}_preference_pair_audit.jsonl",
    }
    for split in ("train", "test")
}
MANIFEST_FILENAME = "run_manifest.json"
SOURCE_MANIFEST_SCHEMA = "reflective_question_preference_pair_run_v1"
HOLDOUT_MANIFEST_SCHEMA = "dpo_domain_holdout_run_v1"
HOLDOUT_AUDIT_SCHEMA = "domain_holdout_preference_pair_audit_v1"


@dataclass(frozen=True, slots=True)
class DomainHoldoutConfig:
    source_run_dir: Path
    output_root: Path
    run_name: str
    train_datasets: tuple[str, ...]
    test_datasets: tuple[str, ...]
    expected_record_counts: tuple[tuple[str, int], ...]
    expected_pair_counts: tuple[tuple[str, int], ...]

    def record_counts(self) -> dict[str, int]:
        return dict(self.expected_record_counts)

    def pair_counts(self) -> dict[str, int]:
        return dict(self.expected_pair_counts)


def load_domain_holdout_config(path: Path) -> DomainHoldoutConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read domain-holdout config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Domain-holdout config must be a JSON object.")
    expected_keys = {
        "source_run_dir",
        "output_root",
        "run_name",
        "train_datasets",
        "test_datasets",
        "expected_record_counts",
        "expected_pair_counts",
    }
    if set(payload) != expected_keys:
        raise ValueError(
            "Domain-holdout config fields are invalid: "
            f"{sorted(set(payload) ^ expected_keys)}."
        )
    base_dir = path.parent
    train_datasets = _dataset_list(payload, "train_datasets")
    test_datasets = _dataset_list(payload, "test_datasets")
    if set(train_datasets) & set(test_datasets):
        raise ValueError("Train and test datasets must be disjoint.")
    counts = _count_mapping(payload, "expected_record_counts")
    pair_counts = _count_mapping(payload, "expected_pair_counts")
    configured = set(train_datasets) | set(test_datasets)
    if set(counts) != configured or set(pair_counts) != configured:
        raise ValueError(
            "Expected record/pair count datasets must exactly match the train/test "
            "datasets."
        )
    for dataset, record_count in counts.items():
        if pair_counts[dataset] != record_count * 4:
            raise ValueError(
                f"Expected pair count for {dataset!r} must be four times its record "
                "count."
            )
    run_name = payload["run_name"]
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("Config field 'run_name' must be a non-empty string.")
    return DomainHoldoutConfig(
        source_run_dir=_path(payload, "source_run_dir", base_dir),
        output_root=_path(payload, "output_root", base_dir),
        run_name=run_name,
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        expected_record_counts=tuple(counts.items()),
        expected_pair_counts=tuple(pair_counts.items()),
    )


def build_domain_holdout(config: DomainHoldoutConfig) -> Path:
    source_paths = {
        name: config.source_run_dir / filename
        for name, filename in SOURCE_FILENAMES.items()
    }
    source_manifest_path = config.source_run_dir / MANIFEST_FILENAME
    run_dir = _new_run_dir(config.output_root, config.run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / MANIFEST_FILENAME
    output_paths = {
        split: {
            name: run_dir / filename
            for name, filename in OUTPUT_FILENAMES[split].items()
        }
        for split in ("train", "test")
    }
    temporary_paths = {
        split: {
            name: path.with_name(f".{path.name}.tmp")
            for name, path in paths.items()
        }
        for split, paths in output_paths.items()
    }
    manifest = _initial_manifest(
        config,
        source_manifest_path=source_manifest_path,
        source_paths=source_paths,
    )
    _write_json(manifest_path, manifest)

    handles: dict[str, dict[str, BinaryIO]] = {}
    split_pair_counts = {"train": 0, "test": 0}
    dataset_pair_counts: defaultdict[str, int] = defaultdict(int)
    records_by_dataset: defaultdict[str, set[str]] = defaultdict(set)
    split_records = {"train": set(), "test": set()}
    seen_pairs: set[str] = set()
    record_audits: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    configured_datasets = set(config.train_datasets) | set(config.test_datasets)
    source_row_count = 0
    try:
        source_manifest = _validate_source_inventory(
            source_manifest_path, source_paths
        )
        manifest["source"] = _source_metadata(
            config,
            source_manifest_path=source_manifest_path,
            source_manifest=source_manifest,
            source_paths=source_paths,
        )
        _write_json(manifest_path, manifest)
        for split in ("train", "test"):
            handles[split] = {
                "category_evidence": temporary_paths[split][
                    "category_evidence"
                ].open("wb"),
                "question_only": temporary_paths[split]["question_only"].open(
                    "wb"
                ),
                "audit": temporary_paths[split]["audit"].open("wb"),
            }
        with (
            source_paths["category_evidence"].open("rb") as evidence_handle,
            source_paths["question_only"].open("rb") as question_handle,
            source_paths["audit"].open("rb") as audit_handle,
        ):
            for source_line, lines in enumerate(
                zip_longest(
                    evidence_handle,
                    question_handle,
                    audit_handle,
                    fillvalue=None,
                ),
                start=1,
            ):
                source_row_count = source_line
                evidence_line, question_line, audit_line = lines
                if evidence_line is None or question_line is None or audit_line is None:
                    raise ValueError(
                        "Source evidence, question-only, and audit files are not "
                        "line-aligned."
                    )
                evidence = _jsonl_object(
                    evidence_line, source_paths["category_evidence"], source_line
                )
                question = _jsonl_object(
                    question_line, source_paths["question_only"], source_line
                )
                audit = _jsonl_object(
                    audit_line, source_paths["audit"], source_line
                )
                _validate_source_row(
                    evidence=evidence,
                    question=question,
                    audit=audit,
                    source_line=source_line,
                    seen_pairs=seen_pairs,
                )
                dataset = audit["dataset"]
                if dataset not in configured_datasets:
                    raise ValueError(
                        f"Source dataset {dataset!r} has no train/test assignment."
                    )
                split = "train" if dataset in config.train_datasets else "test"
                split_pair_counts[split] += 1
                dataset_pair_counts[dataset] += 1
                identity = (dataset, audit["record_id"])
                records_by_dataset[dataset].add(audit["record_id"])
                split_records[split].add(identity)
                record_audits[identity].append(audit)

                handles[split]["category_evidence"].write(evidence_line)
                handles[split]["question_only"].write(question_line)
                split_audit = dict(audit)
                split_audit.update(
                    {
                        "schema_version": HOLDOUT_AUDIT_SCHEMA,
                        "line_number": split_pair_counts[split],
                        "source_schema_version": audit["schema_version"],
                        "source_line_number": source_line,
                        "split": split,
                    }
                )
                handles[split]["audit"].write(
                    (
                        json.dumps(split_audit, ensure_ascii=False) + "\n"
                    ).encode("utf-8")
                )

        for split_handles in handles.values():
            for handle in split_handles.values():
                handle.close()
        handles.clear()

        expected_source_rows = source_manifest.get("counts", {}).get(
            "pair_count_per_version"
        )
        if source_row_count != expected_source_rows:
            raise ValueError(
                "Validated source row count does not match the source manifest: "
                f"{source_row_count} != {expected_source_rows}."
            )
        _validate_record_bundles(record_audits)
        actual_record_counts = {
            dataset: len(records_by_dataset[dataset])
            for dataset in sorted(configured_datasets)
        }
        actual_pair_counts = {
            dataset: dataset_pair_counts[dataset]
            for dataset in sorted(configured_datasets)
        }
        if actual_record_counts != dict(sorted(config.record_counts().items())):
            raise ValueError(
                "Domain-holdout record counts differ from configuration: "
                f"{actual_record_counts} != {dict(sorted(config.record_counts().items()))}."
            )
        if actual_pair_counts != dict(sorted(config.pair_counts().items())):
            raise ValueError(
                "Domain-holdout pair counts differ from configuration: "
                f"{actual_pair_counts} != {dict(sorted(config.pair_counts().items()))}."
            )
        overlap = split_records["train"] & split_records["test"]
        if overlap:
            raise ValueError("At least one source record crosses the domain holdout.")

        for split in ("train", "test"):
            for name in ("category_evidence", "question_only", "audit"):
                os.replace(temporary_paths[split][name], output_paths[split][name])

        dataset_summaries = {
            dataset: {
                "split": "train" if dataset in config.train_datasets else "test",
                "record_count": actual_record_counts[dataset],
                "pair_count_per_version": actual_pair_counts[dataset],
            }
            for dataset in sorted(configured_datasets)
        }
        manifest.update(
            {
                "dataset_summaries": dataset_summaries,
                "counts": {
                    "train_record_count": len(split_records["train"]),
                    "test_record_count": len(split_records["test"]),
                    "train_pair_count_per_version": split_pair_counts["train"],
                    "test_pair_count_per_version": split_pair_counts["test"],
                },
                "disjointness": {
                    "train_test_dataset_overlap": [],
                    "train_test_record_overlap_count": 0,
                    "is_disjoint": True,
                },
                "output_files": {
                    split: {
                        name: _file_metadata(
                            path, split_pair_counts[split]
                        )
                        for name, path in output_paths[split].items()
                    }
                    for split in ("train", "test")
                },
            }
        )
        completed = _timestamp()
        manifest["run_state"].update(
            {
                "status": "complete",
                "updated_at_utc": completed,
                "completed_at_utc": completed,
            }
        )
        _write_json(manifest_path, manifest)
    except Exception as exc:
        for split_handles in handles.values():
            for handle in split_handles.values():
                handle.close()
        for paths in temporary_paths.values():
            for path in paths.values():
                path.unlink(missing_ok=True)
        for paths in output_paths.values():
            for path in paths.values():
                path.unlink(missing_ok=True)
        manifest["run_state"].update(
            {
                "status": "failed",
                "updated_at_utc": _timestamp(),
                "completed_at_utc": _timestamp(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json(manifest_path, manifest)
        raise

    _print_summary(run_dir, manifest)
    return run_dir


def _validate_source_inventory(
    manifest_path: Path, source_paths: dict[str, Path]
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise ValueError(f"Source manifest does not exist: {manifest_path}")
    manifest = _read_json_object(manifest_path, "source manifest")
    if manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError("Unsupported source preference-pair manifest schema.")
    if manifest.get("run_state", {}).get("status") != "complete":
        raise ValueError("Source preference-pair run is not complete.")
    output_files = manifest.get("output_files")
    if not isinstance(output_files, dict):
        raise ValueError("Source manifest output_files must be an object.")
    for name, path in source_paths.items():
        if not path.is_file():
            raise ValueError(f"Required source file does not exist: {path}")
        metadata = output_files.get(name)
        if not isinstance(metadata, dict):
            raise ValueError(f"Source manifest has no metadata for {name}.")
        if metadata.get("sha256") != _file_sha256(path):
            raise ValueError(f"Source manifest checksum mismatch for {name}.")
    return manifest


def _validate_source_row(
    *,
    evidence: dict[str, Any],
    question: dict[str, Any],
    audit: dict[str, Any],
    source_line: int,
    seen_pairs: set[str],
) -> None:
    for version, row in (
        ("category_evidence", evidence),
        ("question_only", question),
    ):
        if set(row) != {"prompt", "chosen", "rejected"}:
            raise ValueError(f"Invalid {version} fields at source line {source_line}.")
        for field, role in (
            ("prompt", "user"),
            ("chosen", "assistant"),
            ("rejected", "assistant"),
        ):
            messages = row.get(field)
            if (
                not isinstance(messages, list)
                or len(messages) != 1
                or not isinstance(messages[0], dict)
                or set(messages[0]) != {"role", "content"}
                or messages[0].get("role") != role
                or not isinstance(messages[0].get("content"), str)
                or not messages[0]["content"]
            ):
                raise ValueError(
                    f"Invalid {version} {field} at source line {source_line}."
                )
    expected_audit_fields = {
        "schema_version",
        "line_number",
        "pair_id",
        "source_name",
        "source_trace_path",
        "dataset",
        "record_id",
        "transcript_id",
        "segment_id",
        "context_scope",
        "context_turns_before",
        "context_turns_after",
        "target_category",
        "target_code_label",
        "chosen_question",
        "rejected_source_category",
        "rejected_code_label",
        "rejected_question",
        "category_evidence_row_sha256",
        "question_only_row_sha256",
    }
    if set(audit) != expected_audit_fields:
        raise ValueError(
            f"Invalid audit fields at source line {source_line}: "
            f"{sorted(set(audit) ^ expected_audit_fields)}."
        )
    if audit.get("schema_version") != "preference_pair_audit_v1":
        raise ValueError(f"Unsupported audit schema at source line {source_line}.")
    if audit.get("line_number") != source_line:
        raise ValueError(f"Audit line number mismatch at source line {source_line}.")
    for field in (
        "pair_id",
        "source_name",
        "source_trace_path",
        "dataset",
        "record_id",
        "transcript_id",
        "segment_id",
        "context_scope",
        "target_category",
        "target_code_label",
        "chosen_question",
        "rejected_source_category",
        "rejected_code_label",
        "rejected_question",
        "category_evidence_row_sha256",
        "question_only_row_sha256",
    ):
        if not isinstance(audit.get(field), str) or not audit[field]:
            raise ValueError(f"Invalid audit {field} at source line {source_line}.")
    if audit["pair_id"] in seen_pairs:
        raise ValueError(f"Duplicate pair_id {audit['pair_id']}.")
    seen_pairs.add(audit["pair_id"])
    if audit["target_category"] == audit["rejected_source_category"]:
        raise ValueError(f"Audit rejects its target at source line {source_line}.")
    if audit["context_scope"] == "full_interview":
        if audit["context_turns_before"] is not None or audit["context_turns_after"] is not None:
            raise ValueError(
                f"Full-interview audit has turn limits at source line {source_line}."
            )
    elif audit["context_scope"] == "turn_window":
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (
                audit["context_turns_before"],
                audit["context_turns_after"],
            )
        ):
            raise ValueError(
                f"Invalid context-window audit at source line {source_line}."
            )
    else:
        raise ValueError(f"Invalid audit context scope at source line {source_line}.")
    if audit.get("category_evidence_row_sha256") != _canonical_row_sha256(evidence):
        raise ValueError(f"Evidence row hash mismatch at source line {source_line}.")
    if audit.get("question_only_row_sha256") != _canonical_row_sha256(question):
        raise ValueError(f"Question-only row hash mismatch at source line {source_line}.")
    if (
        question["chosen"][0]["content"] != audit.get("chosen_question")
        or question["rejected"][0]["content"] != audit.get("rejected_question")
    ):
        raise ValueError(f"Question-only row/audit mismatch at source line {source_line}.")


def _validate_record_bundles(
    records: dict[tuple[str, str], list[dict[str, Any]]]
) -> None:
    for identity, rows in records.items():
        if len(rows) != 4:
            raise ValueError(f"Record {identity} has {len(rows)} pairs instead of four.")
        if len({row["target_category"] for row in rows}) != 4:
            raise ValueError(f"Record {identity} does not have four target categories.")
        if len({row["transcript_id"] for row in rows}) != 1:
            raise ValueError(f"Record {identity} crosses transcript identities.")


def _initial_manifest(
    config: DomainHoldoutConfig,
    *,
    source_manifest_path: Path,
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    now = _timestamp()
    config_payload = asdict(config)
    config_payload["source_run_dir"] = str(config.source_run_dir)
    config_payload["output_root"] = str(config.output_root)
    config_payload["train_datasets"] = list(config.train_datasets)
    config_payload["test_datasets"] = list(config.test_datasets)
    config_payload["expected_record_counts"] = config.record_counts()
    config_payload["expected_pair_counts"] = config.pair_counts()
    return {
        "schema_version": HOLDOUT_MANIFEST_SCHEMA,
        "created_at_utc": now,
        "python": sys.version,
        "platform": platform.platform(),
        "config": config_payload,
        "source": {
            "run_dir": str(config.source_run_dir),
            "manifest_path": str(source_manifest_path),
            "files": {
                name: {"path": str(path)}
                for name, path in source_paths.items()
            },
        },
        "split_policy": {
            "strategy": "predefined_dataset_holdout",
            "train_datasets": list(config.train_datasets),
            "test_datasets": list(config.test_datasets),
            "model_visible_rows_preserved_byte_for_byte": True,
            "pair_ids_and_row_hashes_preserved": True,
        },
        "output_layout": OUTPUT_FILENAMES,
        "run_state": {
            "status": "running",
            "started_at_utc": now,
            "updated_at_utc": now,
            "completed_at_utc": None,
        },
    }


def _source_metadata(
    config: DomainHoldoutConfig,
    *,
    source_manifest_path: Path,
    source_manifest: dict[str, Any],
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "run_dir": str(config.source_run_dir),
        "manifest_path": str(source_manifest_path),
        "manifest_schema_version": source_manifest.get("schema_version"),
        "manifest_sha256": _file_sha256(source_manifest_path),
        "files": {
            name: {
                "path": str(path),
                "sha256": source_manifest["output_files"][name]["sha256"],
            }
            for name, path in source_paths.items()
        },
    }


def _dataset_list(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"Config field {key!r} must be a unique non-empty string list.")
    return tuple(value)


def _count_mapping(payload: dict[str, Any], key: str) -> dict[str, int]:
    value = payload.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Config field {key!r} must be a non-empty object.")
    result: dict[str, int] = {}
    for dataset, count in value.items():
        if not isinstance(dataset, str) or not dataset:
            raise ValueError(f"Config field {key!r} has an invalid dataset name.")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Config field {key!r}.{dataset} must be positive.")
        result[dataset] = count
    return result


def _path(payload: dict[str, Any], key: str, base_dir: Path) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field {key!r} must be a non-empty path string.")
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _jsonl_object(raw: bytes, path: Path, line_number: int) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} line {line_number} must be a JSON object.")
    return payload


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label.capitalize()} must be a JSON object: {path}")
    return payload


def _canonical_row_sha256(row: dict[str, Any]) -> str:
    return sha256(
        json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: Path, row_count: int) -> dict[str, Any]:
    return {
        "path": str(path),
        "row_count": row_count,
        "byte_count": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _new_run_dir(output_root: Path, run_name: str) -> Path:
    base = output_root / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
    if not base.exists():
        return base
    suffix = 1
    while True:
        candidate = base.with_name(f"{base.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _print_summary(run_dir: Path, manifest: dict[str, Any]) -> None:
    counts = manifest["counts"]
    print("DPO domain-holdout construction summary", flush=True)
    for dataset, summary in manifest["dataset_summaries"].items():
        print(
            f"  dataset={dataset} split={summary['split']} "
            f"records={summary['record_count']} "
            f"pairs_per_version={summary['pair_count_per_version']}",
            flush=True,
        )
    print(
        f"  train_records={counts['train_record_count']} "
        f"train_pairs={counts['train_pair_count_per_version']} "
        f"test_records={counts['test_record_count']} "
        f"test_pairs={counts['test_pair_count_per_version']}",
        flush=True,
    )
    print(f"  run_dir={run_dir}", flush=True)
