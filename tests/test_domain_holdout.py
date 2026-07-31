from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from dpo_training.cli import build_parser as build_training_parser
from dpo_training.config import config_to_jsonable, load_training_config
from dpo_training.data import (
    PreferenceExample,
    build_split_manifest,
    split_examples,
    validate_inputs,
)
from preference_pairs.domain_holdout import (
    HOLDOUT_AUDIT_SCHEMA,
    HOLDOUT_MANIFEST_SCHEMA,
    OUTPUT_FILENAMES,
    DomainHoldoutConfig,
    build_domain_holdout,
    load_domain_holdout_config,
)


def test_domain_holdout_preserves_rows_and_writes_exact_dataset_split(
    tmp_path: Path,
) -> None:
    source, source_lines = _write_source_run(tmp_path / "source")
    config = _holdout_config(tmp_path, source)

    run_dir = build_domain_holdout(config)
    manifest = _read_json(run_dir / "run_manifest.json")

    assert manifest["schema_version"] == HOLDOUT_MANIFEST_SCHEMA
    assert manifest["run_state"]["status"] == "complete"
    assert manifest["counts"] == {
        "train_record_count": 1,
        "test_record_count": 2,
        "train_pair_count_per_version": 4,
        "test_pair_count_per_version": 8,
    }
    assert manifest["disjointness"]["is_disjoint"] is True
    assert manifest["dataset_summaries"]["ukda-4688"]["split"] == "train"
    assert manifest["dataset_summaries"]["energy"]["split"] == "test"
    assert manifest["dataset_summaries"]["sexual-health"]["split"] == "test"

    for version in ("category_evidence", "question_only"):
        train_path = run_dir / OUTPUT_FILENAMES["train"][version]
        test_path = run_dir / OUTPUT_FILENAMES["test"][version]
        assert train_path.read_bytes() == b"".join(
            raw for dataset, raw in source_lines[version] if dataset == "ukda-4688"
        )
        assert test_path.read_bytes() == b"".join(
            raw for dataset, raw in source_lines[version] if dataset != "ukda-4688"
        )

    train_audits = _read_jsonl(run_dir / OUTPUT_FILENAMES["train"]["audit"])
    test_audits = _read_jsonl(run_dir / OUTPUT_FILENAMES["test"]["audit"])
    assert [row["line_number"] for row in train_audits] == [1, 2, 3, 4]
    assert [row["line_number"] for row in test_audits] == list(range(1, 9))
    assert all(row["schema_version"] == HOLDOUT_AUDIT_SCHEMA for row in train_audits)
    assert all(row["split"] == "train" for row in train_audits)
    assert all(row["split"] == "test" for row in test_audits)
    assert {row["source_line_number"] for row in train_audits} == {9, 10, 11, 12}
    assert {row["pair_id"] for row in train_audits}.isdisjoint(
        {row["pair_id"] for row in test_audits}
    )

    root = Path(__file__).resolve().parents[1]
    training = load_training_config(
        root / "configs" / "dpo_training_smollm3_3b_domain_holdout.json",
        input_run_dir_override=run_dir,
    )
    model_path = tmp_path / "model"
    model_path.mkdir()
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (model_path / filename).write_text("{}", encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"weights")
    training = replace(
        training,
        model=replace(training.model, path=model_path),
        split=replace(
            training.split,
            expected_train_record_count=1,
            expected_test_record_count=2,
            expected_train_record_counts=(("ukda-4688", 1),),
            expected_test_record_counts=(("energy", 1), ("sexual-health", 1)),
            expected_train_pair_count=4,
            expected_test_pair_count=8,
        ),
    )
    validated = validate_inputs(training)
    assert len(validated.examples_by_version["category_evidence"]) == 12
    assert validated.predefined_examples_by_version is not None
    assert len(
        validated.predefined_examples_by_version["question_only"]["test"]
    ) == 8


def test_domain_holdout_hash_failure_leaves_only_failed_manifest(
    tmp_path: Path,
) -> None:
    source, _lines = _write_source_run(tmp_path / "source", bad_row_hash=True)

    with pytest.raises(ValueError, match="row hash mismatch"):
        build_domain_holdout(_holdout_config(tmp_path, source))

    run_dirs = list((tmp_path / "output").iterdir())
    assert len(run_dirs) == 1
    assert [path.name for path in run_dirs[0].iterdir()] == ["run_manifest.json"]
    manifest = _read_json(run_dirs[0] / "run_manifest.json")
    assert manifest["run_state"]["status"] == "failed"


def test_checked_in_holdout_config_and_five_training_configs_are_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    holdout = load_domain_holdout_config(root / "configs" / "dpo_domain_holdout.json")
    assert holdout.train_datasets == ("ukda-4688",)
    assert holdout.test_datasets == ("energy", "sexual-health")
    assert holdout.record_counts() == {
        "ukda-4688": 6304,
        "energy": 103,
        "sexual-health": 117,
    }
    assert (
        'dpo-build-domain-holdout = "preference_pairs.domain_holdout_cli:main"'
        in (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    config_paths = sorted((root / "configs").glob("dpo_training*_domain_holdout.json"))
    assert len(config_paths) == 5
    for path in config_paths:
        config = load_training_config(path)
        assert config.split.strategy == "predefined_files"
        assert config.split.seed == 42
        assert config.split.train_datasets == ("ukda-4688",)
        assert config.split.test_datasets == ("energy", "sexual-health")
        assert config.split.expected_train_record_count == 6304
        assert config.split.expected_test_record_count == 220
        assert config.split.expected_train_pair_count == 25216
        assert config.split.expected_test_pair_count == 880
        assert config.trainer.num_train_epochs == 1.0
        serialized = config_to_jsonable(config)
        assert serialized["split"]["strategy"] == "predefined_files"
        assert set(serialized["audit_files"]) == {"train", "test"}


def test_predefined_split_uses_saved_roles_without_resampling(tmp_path: Path) -> None:
    base = load_training_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "dpo_training_smollm3_3b_domain_holdout.json"
    )
    config = replace(base, input_run_dir=tmp_path / "input")
    values: list[tuple[str, str, str, int, int]] = []
    for record in range(6304):
        for category in range(4):
            line = record * 4 + category + 1
            values.append(("train", "ukda-4688", f"U{record}", line, line))
    for record in range(103):
        for category in range(4):
            line = record * 4 + category + 1
            values.append(
                ("test", "energy", f"E{record}", line, 30000 + line)
            )
    for record in range(117):
        for category in range(4):
            line = 412 + record * 4 + category + 1
            values.append(
                ("test", "sexual-health", f"S{record}", line, 31000 + line)
            )
    examples = tuple(
        PreferenceExample(
            row={"prompt": [], "chosen": [], "rejected": []},
            audit={
                "split": split,
                "line_number": line,
                "source_line_number": source_line,
                "pair_id": f"{split}-{line}",
                "dataset": dataset,
                "record_id": record_id,
                "transcript_id": f"{record_id}-transcript",
            },
        )
        for split, dataset, record_id, line, source_line in values
    )
    manifest = build_split_manifest(examples, config, source_fingerprint="source")
    train, test = split_examples(examples, manifest)
    assert manifest["schema_version"] == "dpo_predefined_domain_split_v1"
    assert manifest["counts"] == {
        "train_record_count": 6304,
        "test_record_count": 220,
        "train_pair_count": 25216,
        "test_pair_count": 880,
    }
    assert len(train) == 25216
    assert len(test) == 880
    assert {item.audit["dataset"] for item in train} == {"ukda-4688"}
    assert {item.audit["dataset"] for item in test} == {
        "energy",
        "sexual-health",
    }


def test_holdout_cli_override_and_slurm_array_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    args = build_training_parser().parse_args(
        [
            "--config",
            "config.json",
            "--input-run-dir",
            "/scratch/holdout",
            "--dataset-version",
            "category_evidence",
        ]
    )
    assert args.input_run_dir == Path("/scratch/holdout")

    script = (root / "submit_job_dpo_training_domain_holdout_array.slurm").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --array=0-9%1" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --cpus-per-task=12" in script
    assert "#SBATCH --mem=200G" in script
    assert "#SBATCH --time=2-12:00:00" in script
    assert 'DPO_INPUT_RUN_DIR="${DPO_INPUT_RUN_DIR:-}"' in script
    assert "DPO_INPUT_RUN_DIR must name a completed domain-holdout run" in script
    assert script.count('DATASET_VERSION="category_evidence"') == 5
    assert script.count('DATASET_VERSION="question_only"') == 5
    assert script.count("_domain_holdout.json") == 10
    assert "PREFLIGHT_ONLY=true cannot be combined with RESUME_RUN_DIR" in script
    assert "Resume requires one task selected with sbatch --array=<task-id>" in script
    assert '--input-run-dir "${DPO_INPUT_RUN_DIR}"' in script


def _holdout_config(tmp_path: Path, source: Path) -> DomainHoldoutConfig:
    return DomainHoldoutConfig(
        source_run_dir=source,
        output_root=tmp_path / "output",
        run_name="test_domain_holdout",
        train_datasets=("ukda-4688",),
        test_datasets=("energy", "sexual-health"),
        expected_record_counts=(("ukda-4688", 1), ("energy", 1), ("sexual-health", 1)),
        expected_pair_counts=(("ukda-4688", 4), ("energy", 4), ("sexual-health", 4)),
    )


def _write_source_run(
    root: Path, *, bad_row_hash: bool = False
) -> tuple[Path, dict[str, list[tuple[str, bytes]]]]:
    root.mkdir(parents=True)
    raw_by_version: dict[str, list[tuple[str, bytes]]] = {
        "category_evidence": [],
        "question_only": [],
    }
    audits: list[dict[str, Any]] = []
    source_line = 0
    for dataset, record_id in (
        ("energy", "E1"),
        ("sexual-health", "S1"),
        ("ukda-4688", "U1"),
    ):
        for category in range(4):
            source_line += 1
            evidence = _row(f"{dataset} evidence {category}")
            question = _row(f"{dataset} question {category}")
            evidence_raw = (json.dumps(evidence, ensure_ascii=False) + "\n").encode()
            question_raw = (json.dumps(question, ensure_ascii=False) + "\n").encode()
            raw_by_version["category_evidence"].append((dataset, evidence_raw))
            raw_by_version["question_only"].append((dataset, question_raw))
            audits.append(
                {
                    "schema_version": "preference_pair_audit_v1",
                    "line_number": source_line,
                    "pair_id": f"{dataset}-{category}",
                    "source_name": "test-source",
                    "source_trace_path": f"/traces/{dataset}/{record_id}.json",
                    "dataset": dataset,
                    "record_id": record_id,
                    "transcript_id": f"{record_id}-transcript",
                    "segment_id": record_id,
                    "context_scope": "turn_window" if dataset == "ukda-4688" else "full_interview",
                    "context_turns_before": 20 if dataset == "ukda-4688" else None,
                    "context_turns_after": 20 if dataset == "ukda-4688" else None,
                    "target_category": f"category-{category}",
                    "target_code_label": f"code-{category}",
                    "chosen_question": question["chosen"][0]["content"],
                    "rejected_source_category": f"category-{(category + 1) % 4}",
                    "rejected_code_label": f"code-{(category + 1) % 4}",
                    "rejected_question": question["rejected"][0]["content"],
                    "category_evidence_row_sha256": _row_hash(evidence),
                    "question_only_row_sha256": (
                        "bad-hash" if bad_row_hash and source_line == 1 else _row_hash(question)
                    ),
                }
            )
    filenames = {
        "category_evidence": "preference_pairs_category_evidence.jsonl",
        "question_only": "preference_pairs_question_only.jsonl",
        "audit": "preference_pair_audit.jsonl",
    }
    for version in ("category_evidence", "question_only"):
        (root / filenames[version]).write_bytes(
            b"".join(raw for _dataset, raw in raw_by_version[version])
        )
    (root / filenames["audit"]).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audits),
        encoding="utf-8",
        newline="\n",
    )
    output_files = {
        name: {"sha256": _path_hash(root / filename), "row_count": 12}
        for name, filename in filenames.items()
    }
    _write_json(
        root / "run_manifest.json",
        {
            "schema_version": "reflective_question_preference_pair_run_v1",
            "run_state": {"status": "complete"},
            "counts": {"pair_count_per_version": 12},
            "output_files": output_files,
        },
    )
    return root, raw_by_version


def _row(label: str) -> dict[str, Any]:
    return {
        "prompt": [{"role": "user", "content": f"Prompt {label}"}],
        "chosen": [{"role": "assistant", "content": f"Chosen {label}"}],
        "rejected": [{"role": "assistant", "content": f"Rejected {label}"}],
    }


def _row_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _path_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
