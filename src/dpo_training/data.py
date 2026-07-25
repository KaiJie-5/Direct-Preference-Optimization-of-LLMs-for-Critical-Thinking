from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable

from .config import DATASET_VERSIONS, DPOTrainingConfig


@dataclass(frozen=True, slots=True)
class PreferenceExample:
    row: dict[str, Any]
    audit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ValidatedPreferenceData:
    examples_by_version: dict[str, tuple[PreferenceExample, ...]]
    source_manifest: dict[str, Any]
    input_hashes: dict[str, str]


def validate_inputs(config: DPOTrainingConfig) -> ValidatedPreferenceData:
    _require_directory(config.input_run_dir, "input run directory")
    _require_directory(config.model.path, "model directory")
    _validate_model_files(config.model.path)
    source_manifest = _read_json_object(
        config.source_manifest_path, "preference-pair source manifest"
    )
    if source_manifest.get("run_state", {}).get("status") != "complete":
        raise ValueError("Preference-pair source manifest is not complete.")
    if source_manifest.get("schema_version") != "reflective_question_preference_pair_run_v1":
        raise ValueError("Unsupported preference-pair source manifest schema.")

    paths = {
        **{
            version: config.dataset_file(version)
            for version in DATASET_VERSIONS
        },
        "audit": config.audit_path,
        "source_manifest": config.source_manifest_path,
    }
    for name, path in paths.items():
        _require_file(path, name)

    input_hashes = {name: file_sha256(path) for name, path in paths.items()}
    output_files = source_manifest.get("output_files")
    if not isinstance(output_files, dict):
        raise ValueError("Source manifest output_files must be an object.")
    for version in DATASET_VERSIONS:
        metadata = output_files.get(version)
        if not isinstance(metadata, dict):
            raise ValueError(f"Source manifest has no metadata for {version}.")
        if metadata.get("sha256") != input_hashes[version]:
            raise ValueError(f"Source manifest checksum mismatch for {version}.")
    audit_metadata = output_files.get("audit")
    if (
        not isinstance(audit_metadata, dict)
        or audit_metadata.get("sha256") != input_hashes["audit"]
    ):
        raise ValueError("Source manifest checksum mismatch for audit.")

    examples: dict[str, list[PreferenceExample]] = {
        version: [] for version in DATASET_VERSIONS
    }
    seen_pairs: set[str] = set()
    seen_lines: set[int] = set()
    record_rows: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    files = {
        version: config.dataset_file(version).open("r", encoding="utf-8")
        for version in DATASET_VERSIONS
    }
    audit_handle = config.audit_path.open("r", encoding="utf-8")
    try:
        iterables = [files[version] for version in DATASET_VERSIONS]
        for index, values in enumerate(
            zip_longest(*iterables, audit_handle, fillvalue=None), start=1
        ):
            evidence_line, question_line, audit_line = values
            if evidence_line is None or question_line is None or audit_line is None:
                raise ValueError(
                    "Evidence, question-only, and audit JSONL files are not "
                    "line-aligned."
                )
            rows = {
                "category_evidence": _jsonl_object(
                    evidence_line, config.dataset_file("category_evidence"), index
                ),
                "question_only": _jsonl_object(
                    question_line, config.dataset_file("question_only"), index
                ),
            }
            audit = _jsonl_object(audit_line, config.audit_path, index)
            _validate_audit(audit, index, seen_pairs, seen_lines)
            for version, row in rows.items():
                _validate_conversation_row(row, version, index)
                expected_hash = audit.get(f"{version}_row_sha256")
                if expected_hash != canonical_row_sha256(row):
                    raise ValueError(
                        f"{version} row hash mismatch at line {index}."
                    )
                examples[version].append(PreferenceExample(row=row, audit=audit))
            if (
                rows["question_only"]["chosen"][0]["content"]
                != audit["chosen_question"]
                or rows["question_only"]["rejected"][0]["content"]
                != audit["rejected_question"]
            ):
                raise ValueError(
                    f"Question-only row does not match audit questions at line {index}."
                )
            record_rows[(audit["dataset"], audit["record_id"])].append(audit)
    finally:
        audit_handle.close()
        for handle in files.values():
            handle.close()

    _validate_record_bundles(record_rows)
    expected_rows = source_manifest.get("counts", {}).get("pair_count_per_version")
    if expected_rows != len(examples["category_evidence"]):
        raise ValueError(
            "Validated row count does not match source manifest pair count."
        )
    return ValidatedPreferenceData(
        examples_by_version={
            version: tuple(values) for version, values in examples.items()
        },
        source_manifest=source_manifest,
        input_hashes=input_hashes,
    )


def build_split_manifest(
    examples: Iterable[PreferenceExample],
    config: DPOTrainingConfig,
    *,
    source_fingerprint: str,
) -> dict[str, Any]:
    values = tuple(examples)
    records_by_group: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    all_records: defaultdict[str, set[str]] = defaultdict(set)
    for example in values:
        audit = example.audit
        dataset = audit["dataset"]
        transcript_id = audit["transcript_id"]
        record_id = audit["record_id"]
        records_by_group[(dataset, transcript_id)].add(record_id)
        all_records[dataset].add(record_id)

    test_transcripts: dict[str, set[str]] = {}
    dataset_summaries: dict[str, dict[str, Any]] = {}
    expected_test = config.split.expected_test_counts()
    if set(expected_test) != set(all_records):
        raise ValueError(
            "Configured expected test datasets do not match the validated data: "
            f"{sorted(expected_test)} != {sorted(all_records)}."
        )
    for dataset in sorted(all_records):
        groups = {
            transcript_id: len(records)
            for (group_dataset, transcript_id), records in records_by_group.items()
            if group_dataset == dataset
        }
        selected = _closest_group_subset(
            groups,
            target=len(all_records[dataset]) * config.split.test_fraction,
            seed=config.split.seed,
            dataset=dataset,
        )
        selected_count = sum(groups[transcript_id] for transcript_id in selected)
        if selected_count != expected_test[dataset]:
            raise ValueError(
                f"Deterministic test split for {dataset} contains {selected_count} "
                f"records; expected {expected_test[dataset]}."
            )
        test_transcripts[dataset] = selected
        dataset_summaries[dataset] = {
            "record_count": len(all_records[dataset]),
            "transcript_count": len(groups),
            "train_transcript_ids": sorted(set(groups) - selected),
            "test_transcript_ids": sorted(selected),
            "train_record_count": len(all_records[dataset]) - selected_count,
            "test_record_count": selected_count,
        }

    train_lines: list[int] = []
    test_lines: list[int] = []
    train_pair_ids: list[str] = []
    test_pair_ids: list[str] = []
    train_records: set[tuple[str, str]] = set()
    test_records: set[tuple[str, str]] = set()
    for example in values:
        audit = example.audit
        is_test = audit["transcript_id"] in test_transcripts[audit["dataset"]]
        lines = test_lines if is_test else train_lines
        pair_ids = test_pair_ids if is_test else train_pair_ids
        records = test_records if is_test else train_records
        lines.append(audit["line_number"])
        pair_ids.append(audit["pair_id"])
        records.add((audit["dataset"], audit["record_id"]))

    if train_records & test_records:
        raise ValueError("At least one source record crosses the train/test split.")
    if len(train_records) != config.split.expected_train_record_count:
        raise ValueError("Train record count does not match configured expectation.")
    if len(test_records) != config.split.expected_test_record_count:
        raise ValueError("Test record count does not match configured expectation.")
    if len(train_lines) != config.split.expected_train_pair_count:
        raise ValueError("Train pair count does not match configured expectation.")
    if len(test_lines) != config.split.expected_test_pair_count:
        raise ValueError("Test pair count does not match configured expectation.")

    manifest = {
        "schema_version": "dpo_transcript_split_v1",
        "source_fingerprint": source_fingerprint,
        "strategy": {
            "group_fields": list(config.split.group_fields),
            "optimize_for": config.split.optimize_for,
            "test_fraction": config.split.test_fraction,
            "seed": config.split.seed,
            "natural_source_frequencies": True,
        },
        "dataset_summaries": dataset_summaries,
        "counts": {
            "train_record_count": len(train_records),
            "test_record_count": len(test_records),
            "train_pair_count": len(train_lines),
            "test_pair_count": len(test_lines),
        },
        "train": {
            "line_numbers": train_lines,
            "pair_ids": train_pair_ids,
            "record_ids": [
                {"dataset": dataset, "record_id": record_id}
                for dataset, record_id in sorted(train_records)
            ],
        },
        "test": {
            "line_numbers": test_lines,
            "pair_ids": test_pair_ids,
            "record_ids": [
                {"dataset": dataset, "record_id": record_id}
                for dataset, record_id in sorted(test_records)
            ],
        },
    }
    manifest["split_sha256"] = canonical_json_sha256(manifest)
    return manifest


def split_examples(
    examples: Iterable[PreferenceExample], split_manifest: dict[str, Any]
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    test_lines = set(split_manifest["test"]["line_numbers"])
    train: list[PreferenceExample] = []
    test: list[PreferenceExample] = []
    for example in examples:
        (test if example.audit["line_number"] in test_lines else train).append(
            example
        )
    return train, test


def add_system_message(
    row: dict[str, Any], system_message: str
) -> dict[str, list[dict[str, str]]]:
    return {
        "prompt": [
            {"role": "system", "content": system_message},
            *[
                {"role": message["role"], "content": message["content"]}
                for message in row["prompt"]
            ],
        ],
        "chosen": [
            {"role": message["role"], "content": message["content"]}
            for message in row["chosen"]
        ],
        "rejected": [
            {"role": message["role"], "content": message["content"]}
            for message in row["rejected"]
        ],
    }


def ensure_shared_split(
    output_root: Path, split_manifest: dict[str, Any]
) -> Path:
    split_root = output_root / "shared_splits"
    split_root.mkdir(parents=True, exist_ok=True)
    path = split_root / f"{split_manifest['split_sha256']}.json"
    if path.exists():
        existing = _read_json_object(path, "shared split manifest")
        if existing != split_manifest:
            raise ValueError(f"Existing shared split differs from derived split: {path}")
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write_json(temporary, split_manifest)
    os.replace(temporary, path)
    return path


def model_file_hashes(model_path: Path) -> dict[str, dict[str, Any]]:
    index_path = next(
        (
            model_path / name
            for name in (
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            )
            if (model_path / name).is_file()
        ),
        None,
    )
    if index_path is not None:
        index = _read_json_object(index_path, "model weight index")
        weight_map = index.get("weight_map")
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or not all(
                isinstance(name, str) and name
                for name in weight_map.values()
            )
        ):
            raise ValueError("Model weight index has no weight_map.")
        weight_names = set(weight_map.values())
    elif (model_path / "model.safetensors").is_file():
        weight_names = {"model.safetensors"}
    elif (model_path / "pytorch_model.bin").is_file():
        weight_names = {"pytorch_model.bin"}
    else:
        raise ValueError(
            "Model directory has neither indexed nor single-file safetensors/PyTorch "
            "weights."
        )
    names = {
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        *weight_names,
    }
    for optional_name in (
        "generation_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        if (model_path / optional_name).is_file():
            names.add(optional_name)
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        path = model_path / name
        _require_file(path, f"model file {name}")
        result[name] = {
            "byte_count": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return result


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_row_sha256(row: dict[str, Any]) -> str:
    return sha256(
        json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    return sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _closest_group_subset(
    groups: dict[str, int], *, target: float, seed: int, dataset: str
) -> set[str]:
    if len(groups) < 2:
        raise ValueError(
            f"Dataset {dataset!r} requires at least two transcripts for splitting."
        )
    ordered = sorted(
        groups,
        key=lambda transcript: sha256(
            f"{seed}\0{dataset}\0{transcript}".encode("utf-8")
        ).hexdigest(),
    )
    choices: dict[int, tuple[str, ...]] = {0: ()}
    for transcript in ordered:
        size = groups[transcript]
        additions = {
            total + size: selected + (transcript,)
            for total, selected in tuple(choices.items())
            if total + size not in choices
        }
        choices.update(additions)
    total_records = sum(groups.values())
    candidates = [
        (count, selected)
        for count, selected in choices.items()
        if selected and len(selected) < len(groups) and count < total_records
    ]
    if not candidates:
        raise ValueError(f"Could not create a non-empty split for {dataset}.")

    def key(item: tuple[int, tuple[str, ...]]) -> tuple[float, str]:
        count, selected = item
        tie_hash = sha256(
            f"{seed}\0{dataset}\0{'|'.join(sorted(selected))}".encode("utf-8")
        ).hexdigest()
        return abs(count - target), tie_hash

    return set(min(candidates, key=key)[1])


def _validate_model_files(model_path: Path) -> None:
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        _require_file(model_path / name, f"model file {name}")
    if not (
        (model_path / "model.safetensors.index.json").is_file()
        or (model_path / "pytorch_model.bin.index.json").is_file()
        or (model_path / "model.safetensors").is_file()
        or (model_path / "pytorch_model.bin").is_file()
    ):
        raise ValueError(
            "Required model weights do not exist: expected safetensors or PyTorch "
            "single-file/indexed weights."
        )


def _validate_conversation_row(row: dict[str, Any], version: str, line: int) -> None:
    if set(row) != {"prompt", "chosen", "rejected"}:
        raise ValueError(
            f"{version} line {line} must contain exactly prompt, chosen, rejected."
        )
    expected_roles = {
        "prompt": "user",
        "chosen": "assistant",
        "rejected": "assistant",
    }
    for field, role in expected_roles.items():
        messages = row.get(field)
        if not isinstance(messages, list) or len(messages) != 1:
            raise ValueError(
                f"{version} line {line} {field} must contain exactly one message."
            )
        message = messages[0]
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message.get("role") != role
            or not isinstance(message.get("content"), str)
            or not message["content"]
        ):
            raise ValueError(
                f"{version} line {line} has an invalid {field} message."
            )


def _validate_audit(
    audit: dict[str, Any],
    line_number: int,
    seen_pairs: set[str],
    seen_lines: set[int],
) -> None:
    expected_fields = {
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
    if set(audit) != expected_fields:
        raise ValueError(
            f"Audit line {line_number} has invalid fields: "
            f"{sorted(set(audit) ^ expected_fields)}."
        )
    if audit.get("schema_version") != "preference_pair_audit_v1":
        raise ValueError(f"Audit line {line_number} has unsupported schema.")
    required_strings = (
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
    )
    for field in required_strings:
        value = audit.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Audit line {line_number} has invalid {field}.")
    if audit.get("line_number") != line_number:
        raise ValueError(f"Audit line_number mismatch at line {line_number}.")
    context_scope = audit["context_scope"]
    turns_before = audit["context_turns_before"]
    turns_after = audit["context_turns_after"]
    if context_scope == "full_interview":
        if turns_before is not None or turns_after is not None:
            raise ValueError(
                f"Audit line {line_number} full-interview context has turn limits."
            )
    elif context_scope == "turn_window":
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (turns_before, turns_after)
        ):
            raise ValueError(
                f"Audit line {line_number} has invalid context-window limits."
            )
    else:
        raise ValueError(f"Audit line {line_number} has invalid context_scope.")
    if audit["target_category"] == audit["rejected_source_category"]:
        raise ValueError(
            f"Audit line {line_number} rejects the target category itself."
        )
    if audit["pair_id"] in seen_pairs:
        raise ValueError(f"Duplicate audit pair_id {audit['pair_id']}.")
    if line_number in seen_lines:
        raise ValueError(f"Duplicate audit line number {line_number}.")
    seen_pairs.add(audit["pair_id"])
    seen_lines.add(line_number)


def _validate_record_bundles(
    record_rows: dict[tuple[str, str], list[dict[str, Any]]]
) -> None:
    for identity, rows in record_rows.items():
        if len(rows) != 4:
            raise ValueError(
                f"Record {identity} has {len(rows)} preference rows instead of four."
            )
        categories = [row["target_category"] for row in rows]
        if len(set(categories)) != 4:
            raise ValueError(f"Record {identity} does not have four target categories.")
        transcript_ids = {row["transcript_id"] for row in rows}
        if len(transcript_ids) != 1:
            raise ValueError(f"Record {identity} crosses transcript identities.")


def _jsonl_object(line: str, path: Path, line_number: int) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path} at line {line_number}: {exc}"
        ) from exc
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


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"Required {label} does not exist: {path}")


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise ValueError(f"Required {label} does not exist: {path}")
