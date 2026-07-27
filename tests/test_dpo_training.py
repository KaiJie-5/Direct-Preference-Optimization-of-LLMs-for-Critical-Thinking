from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import dpo_training.runner as runner_module
from dpo_training.cli import build_parser
from dpo_training.config import (
    ChatConfig,
    DPOTrainingConfig,
    ModelConfig,
    SplitConfig,
    TrainerConfig,
    config_to_jsonable,
    load_training_config,
)
from dpo_training.data import (
    PreferenceExample,
    ValidatedPreferenceData,
    add_system_message,
    build_split_manifest,
    canonical_json_sha256,
    canonical_row_sha256,
    file_sha256,
    model_file_hashes,
    split_examples,
    validate_inputs,
)
from dpo_training.preflight import (
    assert_within_context_limit,
    render_and_profile,
    validate_model_context_limit,
)
from dpo_training.runner import (
    RENDERED_TEST_FILENAME,
    _create_trainer,
    _latest_checkpoint,
    _read_rendered_snapshot,
    _read_token_profile,
    _start_run,
    _validate_training_dependencies,
    _write_rendered_snapshot,
    run_training,
)


CATEGORIES = ("wrong", "descriptive", "broad", "useful")


def test_checked_in_smollm3_config_contains_approved_recipe() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_training_config(
        root / "configs" / "dpo_training_smollm3_3b.json"
    )

    assert config.model.path == Path(
        "/iridisfs/scratch/kjl1a21/DPO/models/student/"
        "HuggingFaceTB__SmolLM3-3B"
    )
    assert config.input_run_dir == Path(
        "/iridisfs/scratch/kjl1a21/DPO/data/dpo_preference_pairs/"
        "reflective_question_dpo_pairs_20260725_110033"
    )
    assert config.output_root == Path(
        "/iridisfs/scratch/kjl1a21/DPO/models/student/dpo_runs"
    )
    assert config.chat.system_message == "/no_think"
    assert config.chat.native_date_metadata is True
    assert config.split.seed == 42
    assert config.split.group_fields == ("dataset", "transcript_id")
    assert config.split.expected_test_counts() == {
        "energy": 11,
        "sexual-health": 12,
        "ukda-4688": 630,
    }
    assert config.split.expected_train_pair_count == 23484
    assert config.split.expected_test_pair_count == 2612
    assert config.trainer.loss_type == "sigmoid"
    assert config.trainer.beta == 0.1
    assert config.trainer.learning_rate == 5e-7
    assert config.trainer.num_train_epochs == 1.0
    assert config.trainer.gradient_accumulation_steps == 8
    assert config.trainer.max_length is None
    assert config.trainer.precompute_ref_log_probs is False
    assert config.trainer.report_to == "none"
    assert config.trainer.push_to_hub is False


def test_checked_in_qwen3_config_contains_approved_recipe() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_training_config(
        root / "configs" / "dpo_training_qwen3_4b_instruct_2507.json"
    )
    smol_config = load_training_config(
        root / "configs" / "dpo_training_smollm3_3b.json"
    )

    assert config.run_name == "qwen3_4b_instruct_2507_reflective_dpo"
    assert config.model.path == Path(
        "/iridisfs/scratch/kjl1a21/DPO/models/student/"
        "Qwen__Qwen3-4B-Instruct-2507"
    )
    assert config.model.local_files_only is True
    assert config.model.trust_remote_code is False
    assert config.model.dtype == "bfloat16"
    assert config.model.max_position_embeddings == 262144
    assert config.chat.system_message is None
    assert config.chat.native_date_metadata is False
    assert config_to_jsonable(config)["chat"] == {
        "system_message": None,
        "native_date_metadata": False,
    }
    assert config.input_run_dir == smol_config.input_run_dir
    assert config.output_root == smol_config.output_root
    assert config.dataset_files == smol_config.dataset_files
    assert config.split == smol_config.split
    assert config.trainer == smol_config.trainer
    assert config.trainer.loss_type == "sigmoid"
    assert config.trainer.beta == 0.1
    assert config.trainer.learning_rate == 5e-7
    assert config.trainer.num_train_epochs == 1.0
    assert config.trainer.per_device_train_batch_size == 1
    assert config.trainer.gradient_accumulation_steps == 8
    assert config.trainer.lr_scheduler_type == "cosine"
    assert config.trainer.precompute_ref_log_probs is False
    assert config.trainer.max_length is None


def test_checked_in_llama3_config_contains_approved_recipe() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_training_config(
        root / "configs" / "dpo_training_llama_3_2_3b_instruct.json"
    )
    smol_config = load_training_config(
        root / "configs" / "dpo_training_smollm3_3b.json"
    )

    assert config.run_name == "llama_3_2_3b_instruct_reflective_dpo"
    assert config.model.path == Path(
        "/iridisfs/scratch/kjl1a21/DPO/models/student/"
        "meta-llama__Llama-3.2-3B-Instruct"
    )
    assert config.model.local_files_only is True
    assert config.model.trust_remote_code is False
    assert config.model.dtype == "bfloat16"
    assert config.model.max_position_embeddings == 131072
    assert config.chat.system_message is None
    assert config.chat.native_date_metadata is True
    assert config_to_jsonable(config)["chat"] == {
        "system_message": None,
        "native_date_metadata": True,
    }
    assert config.input_run_dir == smol_config.input_run_dir
    assert config.output_root == smol_config.output_root
    assert config.dataset_files == smol_config.dataset_files
    assert config.split == smol_config.split
    assert config.trainer == smol_config.trainer
    assert config.trainer.loss_type == "sigmoid"
    assert config.trainer.beta == 0.1
    assert config.trainer.learning_rate == 5e-7
    assert config.trainer.num_train_epochs == 1.0
    assert config.trainer.per_device_train_batch_size == 1
    assert config.trainer.gradient_accumulation_steps == 8
    assert config.trainer.lr_scheduler_type == "cosine"
    assert config.trainer.precompute_ref_log_probs is False
    assert config.trainer.max_length is None


@pytest.mark.parametrize("system_message", ["", "   ", 123])
def test_config_rejects_invalid_non_null_system_message(
    tmp_path: Path, system_message: Any
) -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            root
            / "configs"
            / "dpo_training_qwen3_4b_instruct_2507.json"
        ).read_text(encoding="utf-8")
    )
    payload["chat"]["system_message"] = system_message
    path = tmp_path / "config.json"
    _write_json(path, payload)

    with pytest.raises(ValueError, match="null or a non-empty string"):
        load_training_config(path)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda payload: payload["trainer"].update({"max_length": 1024}), "never truncates"),
        (
            lambda payload: payload["split"].update({"group_fields": ["record_id"]}),
            "prevent interview-context leakage",
        ),
        (
            lambda payload: payload["trainer"].update(
                {"precompute_ref_log_probs": True}
            ),
            "reference model resident",
        ),
        (
            lambda payload: payload["dataset_files"].pop("question_only"),
            "must contain exactly",
        ),
        (
            lambda payload: payload["trainer"].update(
                {"max_prompt_length": 1024}
            ),
            "unknown",
        ),
    ],
)
def test_config_rejects_unsupported_or_unsafe_settings(
    tmp_path: Path, mutation: Any, error: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "configs" / "dpo_training_smollm3_3b.json").read_text(
            encoding="utf-8"
        )
    )
    mutation(payload)
    path = tmp_path / "config.json"
    _write_json(path, payload)
    with pytest.raises(ValueError, match=error):
        load_training_config(path)


@pytest.mark.parametrize("dataset_version", ["category_evidence", "question_only"])
def test_cli_requires_supported_dataset_version_and_exposes_resume(
    dataset_version: str,
) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            "config.json",
            "--dataset-version",
            dataset_version,
            "--resume",
            "run",
        ]
    )
    assert args.dataset_version == dataset_version
    assert args.resume == Path("run")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--config",
                "config.json",
                "--dataset-version",
                "unknown",
            ]
        )


def test_dependency_guard_enforces_pinned_trl_and_transformers_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        "trl": "1.9.0",
        "transformers": "5.2.0",
        "datasets": "4.1.0",
        "accelerate": "1.10.1",
        "torch": "2.8.0",
    }
    monkeypatch.setattr(
        runner_module.importlib.metadata,
        "version",
        lambda package: versions[package],
    )
    _validate_training_dependencies()

    versions["trl"] = "1.10.0"
    versions["transformers"] = "6.0.0"
    with pytest.raises(RuntimeError, match="pinned version 1.9.0"):
        _validate_training_dependencies()


def test_validate_inputs_preserves_unicode_and_checks_every_aligned_hash(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    rows, audits = _source_rows(
        [
            ("energy", "INT01", "INT01_SEG001"),
            ("energy", "INT02", "INT02_SEG001"),
        ]
    )
    _write_input_run(config, rows, audits)

    validated = validate_inputs(config)

    assert len(validated.examples_by_version["category_evidence"]) == 8
    question = validated.examples_by_version["question_only"][0].row["chosen"][0][
        "content"
    ]
    assert question == "Why does the participant’s £38 response matter?"
    assert set(validated.input_hashes) == {
        "category_evidence",
        "question_only",
        "audit",
        "source_manifest",
    }


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (
            lambda root: (root / "preference_pairs_question_only.jsonl").write_text(
                "{broken\n", encoding="utf-8"
            ),
            "Invalid JSON",
        ),
        (
            lambda root: (root / "preference_pair_audit.jsonl").write_text(
                "\n".join(
                    (root / "preference_pair_audit.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()[:-1]
                )
                + "\n",
                encoding="utf-8",
            ),
            "not line-aligned",
        ),
        (
            lambda root: _mutate_first_jsonl_row(
                root / "preference_pairs_category_evidence.jsonl",
                lambda row: row.update({"extra": "bad"}),
            ),
            "exactly prompt, chosen, rejected",
        ),
    ],
)
def test_invalid_json_alignment_and_schema_fail_fast(
    tmp_path: Path, change: Any, error: str
) -> None:
    config = _config(tmp_path)
    rows, audits = _source_rows(
        [
            ("energy", "INT01", "INT01_SEG001"),
            ("energy", "INT02", "INT02_SEG001"),
        ]
    )
    _write_input_run(config, rows, audits)
    change(config.input_run_dir)
    _refresh_source_manifest_hashes(config)
    with pytest.raises(ValueError, match=error):
        validate_inputs(config)


def test_row_checksum_and_duplicate_pair_identity_fail_fast(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rows, audits = _source_rows(
        [
            ("energy", "INT01", "INT01_SEG001"),
            ("energy", "INT02", "INT02_SEG001"),
        ]
    )
    _write_input_run(config, rows, audits)
    _mutate_first_jsonl_row(
        config.dataset_file("category_evidence"),
        lambda row: row["chosen"][0].update({"content": "changed"}),
    )
    _refresh_source_manifest_hashes(config)
    with pytest.raises(ValueError, match="row hash mismatch"):
        validate_inputs(config)

    _write_input_run(config, rows, audits)
    audit_path = config.audit_path
    values = _read_jsonl(audit_path)
    values[1]["pair_id"] = values[0]["pair_id"]
    _write_jsonl(audit_path, values)
    _refresh_source_manifest_hashes(config)
    with pytest.raises(ValueError, match="Duplicate audit pair_id"):
        validate_inputs(config)


def test_split_is_deterministic_grouped_and_optimized_for_records(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        expected_test={"energy": 1, "sexual-health": 1},
        train_records=4,
        test_records=2,
        train_pairs=16,
        test_pairs=8,
    )
    examples = _examples(
        [
            ("energy", "E1", ("e1",)),
            ("energy", "E2", ("e2", "e3")),
            ("sexual-health", "S1", ("s1",)),
            ("sexual-health", "S2", ("s2", "s3")),
        ]
    )

    first = build_split_manifest(examples, config, source_fingerprint="source")
    second = build_split_manifest(examples, config, source_fingerprint="source")
    train, test = split_examples(examples, first)

    assert first == second
    assert first["counts"] == {
        "train_record_count": 4,
        "test_record_count": 2,
        "train_pair_count": 16,
        "test_pair_count": 8,
    }
    train_transcripts = {
        (item.audit["dataset"], item.audit["transcript_id"]) for item in train
    }
    test_transcripts = {
        (item.audit["dataset"], item.audit["transcript_id"]) for item in test
    }
    assert train_transcripts.isdisjoint(test_transcripts)
    assert len(train) % 4 == len(test) % 4 == 0
    assert first["split_sha256"] == canonical_json_sha256(
        {key: value for key, value in first.items() if key != "split_sha256"}
    )


def test_four_pairs_per_record_are_required(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        expected_test={"energy": 1},
        train_records=1,
        test_records=1,
        train_pairs=4,
        test_pairs=4,
    )
    examples = _examples(
        [("energy", "E1", ("one",)), ("energy", "E2", ("two",))]
    )
    with pytest.raises(ValueError, match="pair count"):
        build_split_manifest(examples[:-1], config, source_fingerprint="source")


def test_no_think_injection_preserves_source_answers() -> None:
    row = _conversation_row("Prompt — £", "Chosen participant’s", "Rejected")
    transformed = add_system_message(row, "/no_think")

    assert transformed["prompt"][0] == {
        "role": "system",
        "content": "/no_think",
    }
    assert transformed["prompt"][1] == row["prompt"][0]
    assert transformed["chosen"] == row["chosen"]
    assert transformed["rejected"] == row["rejected"]
    assert row["prompt"][0]["role"] == "user"


def test_null_system_message_preserves_source_row_without_mutation() -> None:
    row = _conversation_row("Prompt", "Chosen", "Rejected")
    original = json.loads(json.dumps(row))

    transformed = add_system_message(row, None)

    assert transformed == original
    assert transformed is not row
    assert transformed["prompt"] is not row["prompt"]
    assert transformed["chosen"] is not row["chosen"]
    assert transformed["rejected"] is not row["rejected"]
    transformed["prompt"][0]["content"] = "changed"
    transformed["chosen"][0]["content"] = "changed"
    assert row == original


def test_render_profile_is_exact_prefix_preserving_and_never_truncates(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        model_limit=1000,
        expected_test={"energy": 1},
        train_records=1,
        test_records=1,
        train_pairs=4,
        test_pairs=4,
    )
    example = _examples([("energy", "E1", ("record",))])[0]
    tokenizer = FakeTokenizer()

    rendered, profile = render_and_profile(
        [example],
        tokenizer=tokenizer,
        config=config,
        dataset_version="question_only",
    )

    assert rendered[0].row["prompt"].startswith("SYSTEM:/no_think")
    assert example.row["chosen"][0]["content"] in rendered[0].row["chosen"]
    assert example.row["rejected"][0]["content"] in rendered[0].row["rejected"]
    assert profile["max_length"] is None
    assert profile["truncation"] is False
    assert profile["native_date_metadata"] is True
    assert profile["chat_template_sha256"]
    assert profile["statistics"]["energy"]["maximum_sequence"]["count"] == 1
    assert profile["over_limit_count"] == 0
    assert_within_context_limit(profile)


def test_qwen_template_preserves_prefix_and_completion_content(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        model_limit=262144,
        system_message=None,
        native_date_metadata=False,
        expected_test={"energy": 1},
        train_records=1,
        test_records=1,
        train_pairs=4,
        test_pairs=4,
    )
    example = _examples([("energy", "E1", ("record",))])[0]
    tokenizer = FakeQwenTokenizer()

    rendered, profile = render_and_profile(
        [example],
        tokenizer=tokenizer,
        config=config,
        dataset_version="category_evidence",
    )

    source_prompt = example.row["prompt"][0]["content"]
    source_chosen = example.row["chosen"][0]["content"]
    source_rejected = example.row["rejected"][0]["content"]
    assert rendered[0].row["prompt"] == (
        f"<|im_start|>user\n{source_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert "<|im_start|>system\n" not in rendered[0].row["prompt"]
    assert rendered[0].row["chosen"] == f"{source_chosen}<|im_end|>\n"
    assert rendered[0].row["rejected"] == f"{source_rejected}<|im_end|>\n"
    assert profile["system_message"] is None
    assert profile["native_date_metadata"] is False
    assert profile["enforced_model_limit"] == 262144
    assert example.row["prompt"][0]["content"] == source_prompt


def test_llama_template_preserves_native_metadata_prefix_and_completions(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        model_limit=131072,
        system_message=None,
        native_date_metadata=True,
        expected_test={"energy": 1},
        train_records=1,
        test_records=1,
        train_pairs=4,
        test_pairs=4,
    )
    example = _examples([("energy", "E1", ("record",))])[0]
    tokenizer = FakeLlamaTokenizer()

    rendered, profile = render_and_profile(
        [example],
        tokenizer=tokenizer,
        config=config,
        dataset_version="question_only",
    )

    source_prompt = example.row["prompt"][0]["content"]
    source_chosen = example.row["chosen"][0]["content"]
    source_rejected = example.row["rejected"][0]["content"]
    expected_prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\n"
        "Today Date: 27 Jul 2026\n\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{source_prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert rendered[0].row["prompt"] == expected_prompt
    assert "/no_think" not in rendered[0].row["prompt"]
    assert rendered[0].row["chosen"] == f"{source_chosen}<|eot_id|>"
    assert rendered[0].row["rejected"] == f"{source_rejected}<|eot_id|>"
    assert profile["system_message"] is None
    assert profile["native_date_metadata"] is True
    assert profile["enforced_model_limit"] == 131072
    assert profile["max_length"] is None
    assert profile["truncation"] is False
    assert example.row["prompt"][0]["content"] == source_prompt
    assert example.row["chosen"][0]["content"] == source_chosen
    assert example.row["rejected"][0]["content"] == source_rejected


def test_over_limit_profile_names_identity_and_fails(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        model_limit=2,
        expected_test={"energy": 1},
        train_records=1,
        test_records=1,
        train_pairs=4,
        test_pairs=4,
    )
    example = _examples([("energy", "E1", ("record",))])[0]
    _rendered, profile = render_and_profile(
        [example],
        tokenizer=FakeTokenizer(),
        config=config,
        dataset_version="category_evidence",
    )

    assert profile["over_limit_count"] == 1
    assert profile["over_limit_examples"][0]["record_id"] == "record"
    with pytest.raises(ValueError, match="exceed"):
        assert_within_context_limit(profile)


def test_context_limit_uses_model_config_not_tokenizer_limit(tmp_path: Path) -> None:
    config = _config(tmp_path, model_limit=65536)
    _write_json(
        config.model.path / "config.json",
        {"max_position_embeddings": 65536},
    )
    assert validate_model_context_limit(config) == {
        "model_max_position_embeddings": 65536,
        "configured_limit": 65536,
    }
    mismatched = _config(tmp_path / "mismatch", model_limit=131072)
    _write_json(
        mismatched.model.path / "config.json",
        {"max_position_embeddings": 65536},
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_model_context_limit(mismatched)


def test_rendered_snapshots_round_trip_unicode_and_detect_changes(
    tmp_path: Path,
) -> None:
    path = tmp_path / RENDERED_TEST_FILENAME
    rows = [
        {
            "prompt": "participant’s £38 — prompt",
            "chosen": "chosen",
            "rejected": "rejected",
        }
    ]
    metadata = _write_rendered_snapshot(path, rows)

    assert _read_rendered_snapshot(path, metadata) == rows
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _read_rendered_snapshot(path, metadata)


def test_resume_token_profile_checks_file_and_canonical_hash(
    tmp_path: Path,
) -> None:
    profile = {
        "schema_version": "dpo_token_profile_v1",
        "over_limit_count": 0,
        "over_limit_examples": [],
        "enforced_model_limit": 65536,
    }
    profile["profile_sha256"] = canonical_json_sha256(profile)
    path = tmp_path / "token_profile.json"
    _write_json(path, profile)
    metadata = {
        "sha256": file_sha256(path),
        "profile_sha256": profile["profile_sha256"],
    }

    assert _read_token_profile(path, metadata) == profile

    profile["over_limit_count"] = 1
    _write_json(path, profile)
    metadata["sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="content does not match"):
        _read_token_profile(path, metadata)


def test_current_dpo_config_mapping_keeps_reference_resident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    captured: dict[str, Any] = {}

    class FakeDPOConfig:
        def __init__(self, **kwargs: Any) -> None:
            captured["args"] = kwargs

    class FakeDataset:
        @staticmethod
        def from_list(rows: list[dict[str, str]]) -> list[dict[str, str]]:
            return rows

    class FakeTrainer:
        pass

    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(bfloat16="bfloat16")
    )
    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(Dataset=FakeDataset)
    )
    monkeypatch.setitem(
        sys.modules,
        "trl",
        types.SimpleNamespace(DPOConfig=FakeDPOConfig, DPOTrainer=FakeTrainer),
    )

    def factory(**kwargs: Any) -> dict[str, Any]:
        captured["trainer"] = kwargs
        return kwargs

    result = _create_trainer(
        config=config,
        dataset_version="category_evidence",
        run_dir=tmp_path / "run",
        tokenizer=FakeTokenizer(),
        rendered_train=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
        rendered_test=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
        trainer_factory=factory,
    )

    args = captured["args"]
    assert args["max_length"] is None
    assert "max_prompt_length" not in args
    assert args["loss_type"] == ["sigmoid"]
    assert args["beta"] == 0.1
    assert args["learning_rate"] == 5e-7
    assert args["lr_scheduler_type"] == "cosine"
    assert args["warmup_ratio"] == 0.1
    assert args["num_train_epochs"] == 1.0
    assert args["per_device_train_batch_size"] == 1
    assert args["gradient_accumulation_steps"] == 8
    assert args["gradient_checkpointing"] is True
    assert args["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert args["use_cache"] is False
    assert args["precompute_ref_log_probs"] is False
    assert args["disable_dropout"] is True
    assert args["logging_strategy"] == "steps"
    assert args["logging_first_step"] is True
    assert args["eval_strategy"] == "no"
    assert args["save_strategy"] == "steps"
    assert args["save_steps"] == 250
    assert args["save_total_limit"] == 2
    assert args["report_to"] == "none"
    assert args["push_to_hub"] is False
    assert result["ref_model"] is None
    assert len(result["train_dataset"]) == len(result["eval_dataset"]) == 1


def test_mocked_training_evaluates_before_and_after_and_saves_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    examples = tuple(
        PreferenceExample(
            row={"prompt": [], "chosen": [], "rejected": []},
            audit={
                "line_number": line,
                "pair_id": f"pair-{line}",
                "dataset": "energy",
                "record_id": "train-record" if line <= 4 else "test-record",
                "transcript_id": "train-transcript" if line <= 4 else "test-transcript",
            },
        )
        for line in range(1, 9)
    )
    validated = ValidatedPreferenceData(
        examples_by_version={
            "category_evidence": examples,
            "question_only": examples,
        },
        source_manifest={},
        input_hashes={"source": "hash"},
    )
    split = {
        "split_sha256": "split-hash",
        "counts": {
            "train_record_count": 1,
            "test_record_count": 1,
            "train_pair_count": 4,
            "test_pair_count": 4,
        },
        "train": {
            "line_numbers": [1, 2, 3, 4],
            "pair_ids": [f"pair-{line}" for line in range(1, 5)],
            "record_ids": [{"dataset": "energy", "record_id": "train-record"}],
        },
        "test": {
            "line_numbers": [5, 6, 7, 8],
            "pair_ids": [f"pair-{line}" for line in range(5, 9)],
            "record_ids": [{"dataset": "energy", "record_id": "test-record"}],
        },
    }
    profile = {
        "profile_sha256": "profile-hash",
        "render_date": "2026-07-25",
        "chat_template_sha256": "template-hash",
        "over_limit_count": 0,
        "over_limit_examples": [],
        "enforced_model_limit": 65536,
    }
    rendered = [
        runner_module.RenderedExample(
            row={"prompt": f"p{line}", "chosen": "c", "rejected": "r"},
            audit=example.audit,
        )
        for line, example in enumerate(examples, start=1)
    ]

    class FakeState:
        log_history = [{"loss": 0.5, "grad_norm": 1.0}]

        def save_to_json(self, path: str) -> None:
            _write_json(Path(path), {"global_step": 1})

    class FakeTokenizer:
        chat_template = "template"
        state = FakeState()

        def save_pretrained(self, path: Path) -> None:
            _write_json(Path(path) / "tokenizer_config.json", {"saved": True})

    class FakeTrainer:
        state = FakeState()

        def __init__(self) -> None:
            self.evaluations: list[str] = []

        def evaluate(self, *, metric_key_prefix: str) -> dict[str, float]:
            self.evaluations.append(metric_key_prefix)
            return {f"{metric_key_prefix}_loss": 0.5}

        def save_metrics(self, name: str, metrics: dict[str, float]) -> None:
            _write_json(
                config.output_root
                / next(config.output_root.iterdir()).name
                / "checkpoints"
                / f"{name}_results.json",
                metrics,
            )

        def train(self, *, resume_from_checkpoint: str | None) -> Any:
            assert resume_from_checkpoint is None
            return types.SimpleNamespace(metrics={"train_loss": 0.4})

        def save_model(self, path: str) -> None:
            target = Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "model.safetensors").write_bytes(b"model")

    trainer = FakeTrainer()
    monkeypatch.setattr(runner_module, "validate_inputs", lambda _config: validated)
    monkeypatch.setattr(runner_module, "model_file_hashes", lambda _path: {"model": {}})
    monkeypatch.setattr(
        runner_module,
        "validate_model_context_limit",
        lambda _config: {"configured_limit": 65536},
    )
    monkeypatch.setattr(
        runner_module,
        "build_split_manifest",
        lambda *_args, **_kwargs: split,
    )
    monkeypatch.setattr(
        runner_module,
        "ensure_shared_split",
        lambda root, _split: root / "shared-split.json",
    )
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(runner_module, "load_tokenizer", lambda _config: tokenizer)
    monkeypatch.setattr(
        runner_module,
        "render_and_profile",
        lambda *_args, **_kwargs: (rendered, profile),
    )
    monkeypatch.setattr(
        runner_module, "_validate_training_dependencies", lambda: None
    )
    monkeypatch.setattr(
        runner_module, "_environment_metadata", lambda: {"packages": {}}
    )
    monkeypatch.setattr(runner_module, "_create_trainer", lambda **_kwargs: trainer)

    run_dir = run_training(config, dataset_version="category_evidence")
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )

    assert trainer.evaluations == ["eval_before", "eval_after"]
    assert manifest["run_state"]["status"] == "complete"
    assert manifest["metrics"]["train"]["train_loss"] == 0.4
    assert set(manifest["artifacts"]) == {
        "resolved_config",
        "split_manifest",
        "token_profile",
        "rendered_train",
        "rendered_test",
        "metrics",
        "trainer_state",
        "log_history",
    }
    assert (run_dir / "final_model" / "model.safetensors").is_file()


def test_explicit_resume_requires_matching_version_and_incomplete_run(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run_dir, manifest, is_resume = _start_run(
        config, dataset_version="category_evidence", resume=None
    )
    assert is_resume is False
    assert manifest["run_state"]["status"] == "created"

    resumed_dir, _resumed_manifest, is_resume = _start_run(
        config, dataset_version="category_evidence", resume=run_dir
    )
    assert resumed_dir == run_dir.resolve()
    assert is_resume is True

    with pytest.raises(ValueError, match="does not match"):
        _start_run(config, dataset_version="question_only", resume=run_dir)
    manifest["run_state"]["status"] = "complete"
    _write_json(run_dir / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="already complete"):
        _start_run(config, dataset_version="category_evidence", resume=run_dir)


def test_latest_checkpoint_is_numeric_and_missing_resume_fails(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    (root / "checkpoint-250").mkdir(parents=True)
    (root / "checkpoint-1000").mkdir()
    (root / "checkpoint-bad").mkdir()
    assert _latest_checkpoint(root).name == "checkpoint-1000"
    with pytest.raises(ValueError, match="has no checkpoints"):
        _latest_checkpoint(tmp_path / "missing")


def test_model_fingerprint_covers_config_tokenizer_and_all_weight_shards(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    hashes = model_file_hashes(config.model.path)
    assert {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }.issubset(hashes)
    assert all(metadata["sha256"] for metadata in hashes.values())


def test_model_fingerprint_supports_standard_pytorch_weight_files(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    (config.model.path / "model.safetensors.index.json").unlink()
    for path in config.model.path.glob("*.safetensors"):
        path.unlink()
    (config.model.path / "pytorch_model.bin").write_bytes(b"pytorch weights")

    hashes = model_file_hashes(config.model.path)

    assert "pytorch_model.bin" in hashes


def test_slurm_uses_one_h200_guards_inputs_and_submits_one_version() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "submit_job_dpo_training.slurm").read_text(encoding="utf-8")

    assert "#SBATCH --partition=quad_h200" in script
    assert "#SBATCH --account=ecs" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --cpus-per-task=12" in script
    assert "#SBATCH --mem=200G" in script
    assert "#SBATCH --time=2-12:00:00" in script
    assert 'DATASET_VERSION="${DATASET_VERSION:-}"' in script
    assert "category_evidence|question_only" in script
    assert 'require_file "${CONFIG_PATH}"' in script
    assert 'require_file "${RESUME_RUN_DIR}/run_manifest.json"' in script
    assert "conda activate \"${CONDA_ENV}\"" in script
    assert "export TOKENIZERS_PARALLELISM=false" in script
    assert "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in script
    assert "--dataset-version \"${DATASET_VERSION}\"" in script
    assert "ARGS+=(--resume \"${RESUME_RUN_DIR}\")" in script
    assert "python -m dpo_training.cli" in script


def test_qwen_slurm_uses_approved_resources_config_and_command() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "submit_job_dpo_training_qwen3_4b_instruct_2507.slurm"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --job-name=dpo_qwen3_4b_instruct_2507" in script
    assert "#SBATCH --partition=quad_h200" in script
    assert "#SBATCH --account=ecs" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks-per-node=1" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --cpus-per-task=12" in script
    assert "#SBATCH --mem=200G" in script
    assert "#SBATCH --time=2-12:00:00" in script
    assert "#SBATCH --output=dpo_qwen3_4b_instruct_2507_%j.out" in script
    assert "#SBATCH --error=dpo_qwen3_4b_instruct_2507_%j.err" in script
    assert (
        'CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/'
        'dpo_training_qwen3_4b_instruct_2507.json}"'
    ) in script
    assert 'DATASET_VERSION="${DATASET_VERSION:-}"' in script
    assert "category_evidence|question_only" in script
    assert 'PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"' in script
    assert 'RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"' in script
    assert 'require_file "${CONFIG_PATH}"' in script
    assert 'require_file "${RESUME_RUN_DIR}/run_manifest.json"' in script
    assert "PREFLIGHT_ONLY=true cannot be combined with RESUME_RUN_DIR" in script
    assert "--config \"${CONFIG_PATH}\"" in script
    assert "--dataset-version \"${DATASET_VERSION}\"" in script
    assert "ARGS+=(--preflight-only)" in script
    assert "ARGS+=(--resume \"${RESUME_RUN_DIR}\")" in script
    assert "printf '%q ' python -m dpo_training.cli" in script
    assert 'python -m dpo_training.cli "${ARGS[@]}"' in script


def test_llama_slurm_uses_approved_resources_config_and_command() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "submit_job_dpo_training_llama_3_2_3b_instruct.slurm"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --job-name=dpo_llama_3_2_3b_instruct" in script
    assert "#SBATCH --partition=quad_h200" in script
    assert "#SBATCH --account=ecs" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks-per-node=1" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --cpus-per-task=12" in script
    assert "#SBATCH --mem=200G" in script
    assert "#SBATCH --time=2-12:00:00" in script
    assert "#SBATCH --output=dpo_llama_3_2_3b_instruct_%j.out" in script
    assert "#SBATCH --error=dpo_llama_3_2_3b_instruct_%j.err" in script
    assert (
        'CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/'
        'dpo_training_llama_3_2_3b_instruct.json}"'
    ) in script
    assert 'DATASET_VERSION="${DATASET_VERSION:-}"' in script
    assert "category_evidence|question_only" in script
    assert 'PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"' in script
    assert 'RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"' in script
    assert 'require_file "${CONFIG_PATH}"' in script
    assert 'require_file "${RESUME_RUN_DIR}/run_manifest.json"' in script
    assert "PREFLIGHT_ONLY=true cannot be combined with RESUME_RUN_DIR" in script
    assert "--config \"${CONFIG_PATH}\"" in script
    assert "--dataset-version \"${DATASET_VERSION}\"" in script
    assert "ARGS+=(--preflight-only)" in script
    assert "ARGS+=(--resume \"${RESUME_RUN_DIR}\")" in script
    assert "printf '%q ' python -m dpo_training.cli" in script
    assert 'python -m dpo_training.cli "${ARGS[@]}"' in script


def test_pyproject_registers_cli_package_and_pinned_training_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dpo-train = "dpo_training.cli:main"' in text
    assert '"dpo_training*"' in text
    assert '"trl==1.9.0"' in text
    assert '"transformers>=4.56.1,<6"' in text
    assert '"datasets>=4.1.0"' in text
    assert '"accelerate>=1.10.1"' in text
    assert '"torch>=2.8.0"' in text


class FakeTokenizer:
    chat_template = "fake-native-smollm3-template-with-date"
    model_max_length = 131072
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        system = next(
            message["content"] for message in messages if message["role"] == "system"
        )
        user = next(
            message["content"] for message in messages if message["role"] == "user"
        )
        prefix = (
            f"SYSTEM:{system}\nDATE:native-current-date\nUSER:{user}\n"
            "ASSISTANT:<think>\n\n</think>\n"
        )
        assistant = [
            message["content"] for message in messages if message["role"] == "assistant"
        ]
        if assistant:
            return prefix + assistant[0] + "<eos>"
        assert add_generation_prompt is True
        return prefix

    def __call__(
        self, text: str, *, add_special_tokens: bool, truncation: bool
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        assert truncation is False
        return {"input_ids": list(range(len(text.split())))}


class FakeQwenTokenizer:
    chat_template = "fake-native-qwen-chatml-template"
    model_max_length = 1010000
    pad_token = "<|endoftext|>"
    eos_token = "<|im_end|>"
    padding_side = "right"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        rendered = "".join(
            f"<|im_start|>{message['role']}\n"
            f"{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered

    def __call__(
        self, text: str, *, add_special_tokens: bool, truncation: bool
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        assert truncation is False
        return {"input_ids": list(range(len(text)))}


class FakeLlamaTokenizer:
    chat_template = "fake-native-llama-template-with-cutoff-and-date"
    model_max_length = 131072
    pad_token = None
    eos_token = "<|eot_id|>"
    padding_side = "right"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        values = list(messages)
        system_message = ""
        if values and values[0]["role"] == "system":
            system_message = values[0]["content"].strip()
            values = values[1:]
        rendered = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Cutting Knowledge Date: December 2023\n"
            "Today Date: 27 Jul 2026\n\n"
            f"{system_message}<|eot_id|>"
        )
        rendered += "".join(
            f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
            f"{message['content'].strip()}<|eot_id|>"
            for message in values
        )
        if add_generation_prompt:
            rendered += (
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        return rendered

    def __call__(
        self, text: str, *, add_special_tokens: bool, truncation: bool
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        assert truncation is False
        return {"input_ids": list(range(len(text)))}


def _config(
    tmp_path: Path,
    *,
    model_limit: int = 65536,
    system_message: str | None = "/no_think",
    native_date_metadata: bool = True,
    expected_test: dict[str, int] | None = None,
    train_records: int = 1,
    test_records: int = 1,
    train_pairs: int = 4,
    test_pairs: int = 4,
) -> DPOTrainingConfig:
    model_path = tmp_path / "model"
    _write_model(model_path, model_limit)
    return DPOTrainingConfig(
        run_name="test_dpo",
        input_run_dir=tmp_path / "input",
        output_root=tmp_path / "output",
        model=ModelConfig(
            path=model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype="bfloat16",
            max_position_embeddings=model_limit,
        ),
        chat=ChatConfig(
            system_message=system_message,
            native_date_metadata=native_date_metadata,
        ),
        split=SplitConfig(
            test_fraction=0.1,
            seed=42,
            group_fields=("dataset", "transcript_id"),
            optimize_for="source_records",
            expected_test_record_counts=tuple(
                (expected_test or {"energy": 1}).items()
            ),
            expected_train_record_count=train_records,
            expected_test_record_count=test_records,
            expected_train_pair_count=train_pairs,
            expected_test_pair_count=test_pairs,
        ),
        trainer=TrainerConfig(
            loss_type="sigmoid",
            beta=0.1,
            learning_rate=5e-7,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            num_train_epochs=1.0,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            max_grad_norm=1.0,
            optimizer="adamw_torch_fused",
            gradient_checkpointing=True,
            gradient_checkpointing_use_reentrant=False,
            use_cache=False,
            precompute_ref_log_probs=False,
            disable_dropout=True,
            logging_strategy="steps",
            logging_steps=10,
            logging_first_step=True,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=250,
            save_total_limit=2,
            dataloader_num_workers=0,
            report_to="none",
            push_to_hub=False,
            max_length=None,
        ),
        dataset_files=(
            ("category_evidence", "preference_pairs_category_evidence.jsonl"),
            ("question_only", "preference_pairs_question_only.jsonl"),
        ),
        audit_filename="preference_pair_audit.jsonl",
        source_manifest_filename="run_manifest.json",
    )


def _write_model(path: Path, model_limit: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_json(path / "config.json", {"max_position_embeddings": model_limit})
    _write_json(path / "tokenizer.json", {"version": "1"})
    _write_json(path / "tokenizer_config.json", {"model_max_length": 131072})
    (path / "chat_template.jinja").write_text(
        "native template", encoding="utf-8"
    )
    (path / "model-00001-of-00002.safetensors").write_bytes(b"weights-one")
    (path / "model-00002-of-00002.safetensors").write_bytes(b"weights-two")
    _write_json(
        path / "model.safetensors.index.json",
        {
            "weight_map": {
                "layer.one": "model-00001-of-00002.safetensors",
                "layer.two": "model-00002-of-00002.safetensors",
            }
        },
    )


def _source_rows(
    records: list[tuple[str, str, str]]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rows = {"category_evidence": [], "question_only": []}
    audits: list[dict[str, Any]] = []
    line_number = 0
    for dataset, transcript_id, record_id in records:
        for category_index, category in enumerate(CATEGORIES):
            line_number += 1
            evidence = _conversation_row(
                f"Prompt {record_id} {category}",
                f"Category evidence chosen {category_index}",
                f"Category evidence rejected {category_index}",
            )
            question = _conversation_row(
                f"Prompt {record_id} {category}",
                "Why does the participant’s £38 response matter?",
                f"Could rejected question {category_index} matter?",
            )
            rows["category_evidence"].append(evidence)
            rows["question_only"].append(question)
            audits.append(
                {
                    "schema_version": "preference_pair_audit_v1",
                    "line_number": line_number,
                    "pair_id": f"pair-{line_number}",
                    "source_name": "fixture-source",
                    "source_trace_path": f"/trace/{record_id}.json",
                    "dataset": dataset,
                    "record_id": record_id,
                    "transcript_id": transcript_id,
                    "segment_id": record_id.split("_")[-1],
                    "context_scope": "full_interview",
                    "context_turns_before": None,
                    "context_turns_after": None,
                    "target_category": category,
                    "target_code_label": f"target code {category_index}",
                    "chosen_question": question["chosen"][0]["content"],
                    "rejected_source_category": CATEGORIES[
                        (category_index + 1) % len(CATEGORIES)
                    ],
                    "rejected_code_label": f"rejected code {category_index}",
                    "rejected_question": question["rejected"][0]["content"],
                    "category_evidence_row_sha256": canonical_row_sha256(evidence),
                    "question_only_row_sha256": canonical_row_sha256(question),
                }
            )
    return rows, audits


def _write_input_run(
    config: DPOTrainingConfig,
    rows: dict[str, list[dict[str, Any]]],
    audits: list[dict[str, Any]],
) -> None:
    config.input_run_dir.mkdir(parents=True, exist_ok=True)
    for version, values in rows.items():
        _write_jsonl(config.dataset_file(version), values)
    _write_jsonl(config.audit_path, audits)
    _refresh_source_manifest_hashes(config)


def _refresh_source_manifest_hashes(config: DPOTrainingConfig) -> None:
    output_files = {
        version: {"sha256": file_sha256(config.dataset_file(version))}
        for version in ("category_evidence", "question_only")
    }
    output_files["audit"] = {"sha256": file_sha256(config.audit_path)}
    row_count = len(_read_jsonl(config.audit_path))
    _write_json(
        config.source_manifest_path,
        {
            "schema_version": "reflective_question_preference_pair_run_v1",
            "run_state": {"status": "complete"},
            "counts": {"pair_count_per_version": row_count},
            "output_files": output_files,
        },
    )


def _examples(
    groups: list[tuple[str, str, tuple[str, ...]]]
) -> list[PreferenceExample]:
    examples: list[PreferenceExample] = []
    line_number = 0
    for dataset, transcript_id, records in groups:
        for record_id in records:
            for category in CATEGORIES:
                line_number += 1
                examples.append(
                    PreferenceExample(
                        row=_conversation_row(
                            f"Prompt {record_id} {category}",
                            f"Chosen {record_id} {category}",
                            f"Rejected {record_id} {category}",
                        ),
                        audit={
                            "line_number": line_number,
                            "pair_id": f"pair-{line_number}",
                            "dataset": dataset,
                            "record_id": record_id,
                            "transcript_id": transcript_id,
                            "segment_id": record_id,
                            "target_category": category,
                        },
                    )
                )
    return examples


def _conversation_row(
    prompt: str, chosen: str, rejected: str
) -> dict[str, list[dict[str, str]]]:
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def _mutate_first_jsonl_row(path: Path, mutation: Any) -> None:
    rows = _read_jsonl(path)
    mutation(rows[0])
    _write_jsonl(path, rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
