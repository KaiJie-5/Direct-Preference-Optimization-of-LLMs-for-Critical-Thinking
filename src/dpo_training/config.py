from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DATASET_VERSIONS = ("category_evidence", "question_only")
MINISTRAL_TRAINING_PROFILE = "ministral3_fp8_text_dpo"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    path: Path
    local_files_only: bool
    trust_remote_code: bool
    dtype: str
    max_position_embeddings: int
    training_profile: str | None = None


@dataclass(frozen=True, slots=True)
class ChatConfig:
    system_message: str | None
    native_date_metadata: bool


@dataclass(frozen=True, slots=True)
class SplitConfig:
    test_fraction: float
    seed: int
    group_fields: tuple[str, ...]
    optimize_for: str
    expected_test_record_counts: tuple[tuple[str, int], ...]
    expected_train_record_count: int
    expected_test_record_count: int
    expected_train_pair_count: int
    expected_test_pair_count: int

    def expected_test_counts(self) -> dict[str, int]:
        return dict(self.expected_test_record_counts)


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    loss_type: str
    beta: float
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    optimizer: str
    gradient_checkpointing: bool
    gradient_checkpointing_use_reentrant: bool
    use_cache: bool
    precompute_ref_log_probs: bool
    disable_dropout: bool
    logging_strategy: str
    logging_steps: int
    logging_first_step: bool
    eval_strategy: str
    save_strategy: str
    save_steps: int
    save_total_limit: int
    dataloader_num_workers: int
    report_to: str
    push_to_hub: bool
    max_length: int | None


@dataclass(frozen=True, slots=True)
class DPOTrainingConfig:
    run_name: str
    input_run_dir: Path
    output_root: Path
    model: ModelConfig
    chat: ChatConfig
    split: SplitConfig
    trainer: TrainerConfig
    dataset_files: tuple[tuple[str, str], ...]
    audit_filename: str
    source_manifest_filename: str

    def dataset_file(self, version: str) -> Path:
        mapping = dict(self.dataset_files)
        if version not in mapping:
            raise ValueError(
                f"Unsupported dataset version {version!r}; expected one of "
                f"{sorted(mapping)}."
            )
        return self.input_run_dir / mapping[version]

    @property
    def audit_path(self) -> Path:
        return self.input_run_dir / self.audit_filename

    @property
    def source_manifest_path(self) -> Path:
        return self.input_run_dir / self.source_manifest_filename


def load_training_config(path: Path) -> DPOTrainingConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read DPO training config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("DPO training config must be a JSON object.")
    _require_exact_keys(
        payload,
        {
            "run_name",
            "input_run_dir",
            "output_root",
            "model",
            "dataset_files",
            "audit_filename",
            "source_manifest_filename",
            "chat",
            "split",
            "trainer",
        },
        "config",
    )

    base_dir = path.parent
    run_name = _string(payload, "run_name")
    input_run_dir = _path(payload, "input_run_dir", base_dir)
    output_root = _path(payload, "output_root", base_dir)

    model_payload = _object(payload, "model")
    required_model_keys = {
        "path",
        "local_files_only",
        "trust_remote_code",
        "dtype",
        "max_position_embeddings",
    }
    _require_required_and_allowed_keys(
        model_payload,
        required=required_model_keys,
        allowed=required_model_keys | {"training_profile"},
        label="model",
    )
    dtype = _string(model_payload, "dtype")
    if dtype != "bfloat16":
        raise ValueError("Full-model training currently requires model.dtype='bfloat16'.")
    training_profile = model_payload.get("training_profile")
    if training_profile is not None:
        training_profile = _choice(
            model_payload,
            "training_profile",
            {MINISTRAL_TRAINING_PROFILE},
        )
    model = ModelConfig(
        path=_path(model_payload, "path", base_dir),
        local_files_only=_bool(model_payload, "local_files_only"),
        trust_remote_code=_bool(model_payload, "trust_remote_code"),
        dtype=dtype,
        max_position_embeddings=_positive_int(
            model_payload, "max_position_embeddings"
        ),
        training_profile=training_profile,
    )
    if (
        model.training_profile == MINISTRAL_TRAINING_PROFILE
        and model.trust_remote_code
    ):
        raise ValueError(
            "The Ministral training profile requires trust_remote_code=false."
        )

    chat_payload = _object(payload, "chat")
    _require_exact_keys(
        chat_payload, {"system_message", "native_date_metadata"}, "chat"
    )
    chat = ChatConfig(
        system_message=_optional_string(chat_payload, "system_message"),
        native_date_metadata=_bool(chat_payload, "native_date_metadata"),
    )

    split_payload = _object(payload, "split")
    _require_exact_keys(
        split_payload,
        {
            "test_fraction",
            "seed",
            "group_fields",
            "optimize_for",
            "expected_test_record_counts",
            "expected_train_record_count",
            "expected_test_record_count",
            "expected_train_pair_count",
            "expected_test_pair_count",
        },
        "split",
    )
    group_fields = split_payload.get("group_fields")
    if (
        not isinstance(group_fields, list)
        or not group_fields
        or not all(isinstance(value, str) and value for value in group_fields)
    ):
        raise ValueError("split.group_fields must be a non-empty string list.")
    expected_counts_payload = _object(split_payload, "expected_test_record_counts")
    expected_counts: list[tuple[str, int]] = []
    for dataset, count in expected_counts_payload.items():
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("Expected split dataset names must be non-empty strings.")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"Expected test count for {dataset!r} must be a non-negative integer."
            )
        expected_counts.append((dataset, count))
    test_fraction = _number(split_payload, "test_fraction")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("split.test_fraction must be between zero and one.")
    optimize_for = _string(split_payload, "optimize_for")
    if optimize_for != "source_records":
        raise ValueError("split.optimize_for must be 'source_records'.")
    if tuple(group_fields) != ("dataset", "transcript_id"):
        raise ValueError(
            "split.group_fields must be ['dataset', 'transcript_id'] to prevent "
            "interview-context leakage."
        )
    split = SplitConfig(
        test_fraction=test_fraction,
        seed=_nonnegative_int(split_payload, "seed"),
        group_fields=tuple(group_fields),
        optimize_for=optimize_for,
        expected_test_record_counts=tuple(expected_counts),
        expected_train_record_count=_positive_int(
            split_payload, "expected_train_record_count"
        ),
        expected_test_record_count=_positive_int(
            split_payload, "expected_test_record_count"
        ),
        expected_train_pair_count=_positive_int(
            split_payload, "expected_train_pair_count"
        ),
        expected_test_pair_count=_positive_int(
            split_payload, "expected_test_pair_count"
        ),
    )

    trainer_payload = _object(payload, "trainer")
    _require_exact_keys(
        trainer_payload,
        {
            "loss_type",
            "beta",
            "learning_rate",
            "lr_scheduler_type",
            "warmup_ratio",
            "num_train_epochs",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "max_grad_norm",
            "optimizer",
            "gradient_checkpointing",
            "gradient_checkpointing_use_reentrant",
            "use_cache",
            "precompute_ref_log_probs",
            "disable_dropout",
            "logging_strategy",
            "logging_steps",
            "logging_first_step",
            "eval_strategy",
            "save_strategy",
            "save_steps",
            "save_total_limit",
            "dataloader_num_workers",
            "report_to",
            "push_to_hub",
            "max_length",
        },
        "trainer",
    )
    max_length = trainer_payload.get("max_length")
    if max_length is not None:
        raise ValueError(
            "trainer.max_length must be null: this workflow never truncates context."
        )
    trainer = TrainerConfig(
        loss_type=_choice(trainer_payload, "loss_type", {"sigmoid"}),
        beta=_positive_number(trainer_payload, "beta"),
        learning_rate=_positive_number(trainer_payload, "learning_rate"),
        lr_scheduler_type=_choice(
            trainer_payload, "lr_scheduler_type", {"cosine"}
        ),
        warmup_ratio=_fraction(trainer_payload, "warmup_ratio"),
        num_train_epochs=_positive_number(trainer_payload, "num_train_epochs"),
        per_device_train_batch_size=_positive_int(
            trainer_payload, "per_device_train_batch_size"
        ),
        per_device_eval_batch_size=_positive_int(
            trainer_payload, "per_device_eval_batch_size"
        ),
        gradient_accumulation_steps=_positive_int(
            trainer_payload, "gradient_accumulation_steps"
        ),
        max_grad_norm=_positive_number(trainer_payload, "max_grad_norm"),
        optimizer=_string(trainer_payload, "optimizer"),
        gradient_checkpointing=_bool(
            trainer_payload, "gradient_checkpointing"
        ),
        gradient_checkpointing_use_reentrant=_bool(
            trainer_payload, "gradient_checkpointing_use_reentrant"
        ),
        use_cache=_bool(trainer_payload, "use_cache"),
        precompute_ref_log_probs=_bool(
            trainer_payload, "precompute_ref_log_probs"
        ),
        disable_dropout=_bool(trainer_payload, "disable_dropout"),
        logging_strategy=_choice(
            trainer_payload, "logging_strategy", {"steps"}
        ),
        logging_steps=_positive_int(trainer_payload, "logging_steps"),
        logging_first_step=_bool(trainer_payload, "logging_first_step"),
        eval_strategy=_choice(trainer_payload, "eval_strategy", {"no"}),
        save_strategy=_choice(trainer_payload, "save_strategy", {"steps"}),
        save_steps=_positive_int(trainer_payload, "save_steps"),
        save_total_limit=_positive_int(trainer_payload, "save_total_limit"),
        dataloader_num_workers=_nonnegative_int(
            trainer_payload, "dataloader_num_workers"
        ),
        report_to=_choice(trainer_payload, "report_to", {"none"}),
        push_to_hub=_bool(trainer_payload, "push_to_hub"),
        max_length=None,
    )
    if trainer.num_train_epochs != 1.0:
        raise ValueError("trainer.num_train_epochs must be 1 for this experiment.")
    if trainer.precompute_ref_log_probs:
        raise ValueError(
            "trainer.precompute_ref_log_probs must be false to keep the reference "
            "model resident."
        )
    if trainer.push_to_hub:
        raise ValueError("trainer.push_to_hub must be false for local-only outputs.")

    dataset_payload = _object(payload, "dataset_files")
    if set(dataset_payload) != set(DATASET_VERSIONS):
        raise ValueError(
            "dataset_files must contain exactly category_evidence and question_only."
        )
    dataset_files = tuple(
        (version, _filename(dataset_payload, version))
        for version in DATASET_VERSIONS
    )

    return DPOTrainingConfig(
        run_name=run_name,
        input_run_dir=input_run_dir,
        output_root=output_root,
        model=model,
        chat=chat,
        split=split,
        trainer=trainer,
        dataset_files=dataset_files,
        audit_filename=_filename(payload, "audit_filename"),
        source_manifest_filename=_filename(payload, "source_manifest_filename"),
    )


def config_to_jsonable(config: DPOTrainingConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["input_run_dir"] = str(config.input_run_dir)
    payload["output_root"] = str(config.output_root)
    payload["model"]["path"] = str(config.model.path)
    if config.model.training_profile is None:
        payload["model"].pop("training_profile")
    payload["split"]["group_fields"] = list(config.split.group_fields)
    payload["split"]["expected_test_record_counts"] = (
        config.split.expected_test_counts()
    )
    payload["dataset_files"] = dict(config.dataset_files)
    return payload


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config field {key!r} must be an object.")
    return value


def _require_exact_keys(
    payload: dict[str, Any], expected: set[str], label: str
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(f"{label} fields are invalid: {', '.join(details)}.")


def _require_required_and_allowed_keys(
    payload: dict[str, Any],
    *,
    required: set[str],
    allowed: set[str],
    label: str,
) -> None:
    actual = set(payload)
    missing = sorted(required - actual)
    unknown = sorted(actual - allowed)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(f"{label} fields are invalid: {', '.join(details)}.")


def _string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field {key!r} must be a non-empty string.")
    return value


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Config field {key!r} must be null or a non-empty string."
        )
    return value


def _filename(payload: dict[str, Any], key: str) -> str:
    value = _string(payload, key)
    if Path(value).name != value:
        raise ValueError(f"Config field {key!r} must be a filename, not a path.")
    return value


def _path(payload: dict[str, Any], key: str, base_dir: Path) -> Path:
    value = _string(payload, key)
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _bool(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Config field {key!r} must be boolean.")
    return value


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Config field {key!r} must be numeric.")
    return float(value)


def _positive_number(payload: dict[str, Any], key: str) -> float:
    value = _number(payload, key)
    if value <= 0:
        raise ValueError(f"Config field {key!r} must be positive.")
    return value


def _fraction(payload: dict[str, Any], key: str) -> float:
    value = _number(payload, key)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"Config field {key!r} must be in [0, 1).")
    return value


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Config field {key!r} must be a positive integer.")
    return value


def _nonnegative_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Config field {key!r} must be a non-negative integer.")
    return value


def _choice(payload: dict[str, Any], key: str, choices: set[str]) -> str:
    value = _string(payload, key)
    if value not in choices:
        raise ValueError(f"Config field {key!r} must be one of {sorted(choices)}.")
    return value
