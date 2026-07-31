from __future__ import annotations

import gzip
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from .config import (
    DPOTrainingConfig,
    MINISTRAL_TRAINING_PROFILE,
    config_to_jsonable,
)
from .data import (
    PreferenceExample,
    build_split_manifest,
    canonical_json_sha256,
    ensure_shared_split,
    file_sha256,
    model_file_hashes,
    split_examples,
    validate_inputs,
    write_json,
)
from .preflight import (
    RenderedExample,
    assert_within_context_limit,
    load_tokenizer,
    render_and_profile,
    validate_model_context_limit,
    verify_saved_native_token_profile,
)


MANIFEST_FILENAME = "run_manifest.json"
SPLIT_FILENAME = "split_manifest.json"
PROFILE_FILENAME = "token_profile.json"
RESOLVED_CONFIG_FILENAME = "resolved_config.json"
RENDERED_TRAIN_FILENAME = "rendered_train.jsonl.gz"
RENDERED_TEST_FILENAME = "rendered_test.jsonl.gz"

TrainerFactory = Callable[..., Any]


def run_training(
    config: DPOTrainingConfig,
    *,
    dataset_version: str,
    preflight_only: bool = False,
    resume: Path | None = None,
    trainer_factory: TrainerFactory | None = None,
) -> Path:
    if config.split.strategy == "predefined_files":
        config.predefined_dataset_file(dataset_version, "train")
        config.predefined_dataset_file(dataset_version, "test")
    else:
        config.dataset_file(dataset_version)
    if preflight_only and resume is not None:
        raise ValueError("--preflight-only cannot be combined with --resume.")

    run_dir, manifest, is_resume = _start_run(
        config,
        dataset_version=dataset_version,
        resume=resume,
    )
    manifest_path = run_dir / MANIFEST_FILENAME
    try:
        _set_state(manifest, "validating_inputs")
        _write_manifest(manifest_path, manifest)
        validated = validate_inputs(config)
        model_hashes = model_file_hashes(
            config.model.path,
            training_profile=config.model.training_profile,
        )
        context_limits = validate_model_context_limit(config)
        source_fingerprint = canonical_json_sha256(validated.input_hashes)
        examples = validated.examples_by_version[dataset_version]
        split_manifest = build_split_manifest(
            examples, config, source_fingerprint=source_fingerprint
        )
        shared_split_path = ensure_shared_split(config.output_root, split_manifest)
        immutable = {
            "config": config_to_jsonable(config),
            "dataset_version": dataset_version,
            "input_hashes": validated.input_hashes,
            "model_files": model_hashes,
            "split_sha256": split_manifest["split_sha256"],
        }
        immutable_fingerprint = canonical_json_sha256(immutable)
        if is_resume:
            if manifest.get("immutable_fingerprint") != immutable_fingerprint:
                raise ValueError(
                    "Resume fingerprint differs from the saved training run."
                )
            saved_config = _read_json(run_dir / RESOLVED_CONFIG_FILENAME)
            if saved_config != config_to_jsonable(config):
                raise ValueError("Resume resolved configuration differs from config.")
            saved_split = _read_json(run_dir / SPLIT_FILENAME)
            if saved_split != split_manifest:
                raise ValueError("Resume split manifest differs from derived split.")
        else:
            write_json(run_dir / SPLIT_FILENAME, split_manifest)
            write_json(run_dir / RESOLVED_CONFIG_FILENAME, config_to_jsonable(config))
            manifest.update(
                {
                    "immutable_fingerprint": immutable_fingerprint,
                    "input_hashes": validated.input_hashes,
                    "model_files": model_hashes,
                    "context_limits": context_limits,
                    "shared_split_path": str(shared_split_path),
                    "split_sha256": split_manifest["split_sha256"],
                    "split_counts": split_manifest["counts"],
                }
            )
            _write_manifest(manifest_path, manifest)

        tokenizer = load_tokenizer(config)
        train_examples, test_examples = split_examples(examples, split_manifest)
        if is_resume:
            rendered_train = _read_rendered_snapshot(
                run_dir / RENDERED_TRAIN_FILENAME,
                manifest["rendered_data"]["train"],
            )
            rendered_test = _read_rendered_snapshot(
                run_dir / RENDERED_TEST_FILENAME,
                manifest["rendered_data"]["test"],
            )
            profile = _read_token_profile(
                run_dir / PROFILE_FILENAME,
                manifest.get("token_profile"),
            )
            template_sha = sha256(
                str(tokenizer.chat_template).encode("utf-8")
            ).hexdigest()
            if template_sha != profile.get("chat_template_sha256"):
                raise ValueError(
                    "Current tokenizer chat template differs from the saved run."
                )
            verify_saved_native_token_profile(
                examples,
                tokenizer=tokenizer,
                config=config,
                profile=profile,
            )
            assert_within_context_limit(profile)
        else:
            _set_state(manifest, "token_preflight")
            _write_manifest(manifest_path, manifest)
            rendered_all, profile = render_and_profile(
                examples,
                tokenizer=tokenizer,
                config=config,
                dataset_version=dataset_version,
            )
            write_json(run_dir / PROFILE_FILENAME, profile)
            manifest["token_profile"] = {
                "path": str(run_dir / PROFILE_FILENAME),
                "sha256": file_sha256(run_dir / PROFILE_FILENAME),
                "profile_sha256": profile["profile_sha256"],
                "render_date": profile["render_date"],
                "chat_template_sha256": profile["chat_template_sha256"],
                "over_limit_count": profile["over_limit_count"],
            }
            if "prompt_policy" in profile:
                manifest["token_profile"]["prompt_policy"] = profile[
                    "prompt_policy"
                ]
            if "native_token_verification" in profile:
                manifest["token_profile"]["native_token_verification"] = profile[
                    "native_token_verification"
                ]
            _write_manifest(manifest_path, manifest)
            assert_within_context_limit(profile)
            rendered_train, rendered_test = _split_rendered(
                rendered_all, split_manifest
            )
            train_metadata = _write_rendered_snapshot(
                run_dir / RENDERED_TRAIN_FILENAME, rendered_train
            )
            test_metadata = _write_rendered_snapshot(
                run_dir / RENDERED_TEST_FILENAME, rendered_test
            )
            manifest.update(
                {
                    "rendered_data": {
                        "train": train_metadata,
                        "test": test_metadata,
                    },
                }
            )
            _write_manifest(manifest_path, manifest)

        if preflight_only:
            _set_state(manifest, "preflight_complete", completed=True)
            _write_manifest(manifest_path, manifest)
            _print_preflight_summary(run_dir, dataset_version, profile, split_manifest)
            return run_dir

        _validate_training_dependencies(config)
        _set_state(manifest, "training")
        manifest["environment"] = _environment_metadata(config)
        _write_manifest(manifest_path, manifest)
        trainer = _create_trainer(
            config=config,
            dataset_version=dataset_version,
            run_dir=run_dir,
            tokenizer=tokenizer,
            rendered_train=rendered_train,
            rendered_test=rendered_test,
            trainer_factory=trainer_factory,
        )
        frozen_components = getattr(
            trainer, "_dpo_frozen_components", None
        )
        if frozen_components is not None:
            manifest["frozen_components"] = frozen_components
            _write_manifest(manifest_path, manifest)
        saved_before_path = run_dir / "checkpoints" / "eval_before_results.json"
        if is_resume and saved_before_path.is_file():
            before_metrics = _read_json(saved_before_path)
        else:
            before_metrics = trainer.evaluate(metric_key_prefix="eval_before")
            trainer.save_metrics("eval_before", before_metrics)
        checkpoint = _latest_checkpoint(run_dir / "checkpoints") if is_resume else None
        train_result = trainer.train(
            resume_from_checkpoint=str(checkpoint) if checkpoint else None
        )
        train_metrics = dict(train_result.metrics)
        trainer.save_metrics("train", train_metrics)
        after_metrics = trainer.evaluate(metric_key_prefix="eval_after")
        trainer.save_metrics("eval_after", after_metrics)

        final_model_dir = run_dir / "final_model"
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(final_model_dir)
        trainer.state.save_to_json(str(run_dir / "trainer_state.json"))
        write_json(run_dir / "log_history.json", trainer.state.log_history)
        metrics = {
            "eval_before": before_metrics,
            "train": train_metrics,
            "eval_after": after_metrics,
        }
        write_json(run_dir / "metrics.json", metrics)
        final_hashes = _directory_hashes(final_model_dir)
        artifact_paths = {
            "resolved_config": run_dir / RESOLVED_CONFIG_FILENAME,
            "split_manifest": run_dir / SPLIT_FILENAME,
            "token_profile": run_dir / PROFILE_FILENAME,
            "rendered_train": run_dir / RENDERED_TRAIN_FILENAME,
            "rendered_test": run_dir / RENDERED_TEST_FILENAME,
            "metrics": run_dir / "metrics.json",
            "trainer_state": run_dir / "trainer_state.json",
            "log_history": run_dir / "log_history.json",
        }
        manifest.update(
            {
                "metrics": metrics,
                "final_model": {
                    "path": str(final_model_dir),
                    "files": final_hashes,
                },
                "checkpoints": _checkpoint_inventory(run_dir / "checkpoints"),
                "artifacts": {
                    name: {
                        "path": str(path),
                        "byte_count": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                    for name, path in artifact_paths.items()
                },
            }
        )
        _set_state(manifest, "complete", completed=True)
        _write_manifest(manifest_path, manifest)
        _print_training_summary(run_dir, dataset_version, split_manifest, metrics)
        return run_dir
    except Exception as exc:
        manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _set_state(manifest, "failed", completed=True)
        _write_manifest(manifest_path, manifest)
        raise


def _create_trainer(
    *,
    config: DPOTrainingConfig,
    dataset_version: str,
    run_dir: Path,
    tokenizer: Any,
    rendered_train: list[dict[str, str]],
    rendered_test: list[dict[str, str]],
    trainer_factory: TrainerFactory | None,
) -> Any:
    try:
        import torch
        from datasets import Dataset
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "TRL training dependencies are unavailable. Install the project's "
            "training optional dependency group."
        ) from exc
    trainer_config = config.trainer
    model_init_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "local_files_only": config.model.local_files_only,
        "trust_remote_code": config.model.trust_remote_code,
    }
    if config.model.training_profile == MINISTRAL_TRAINING_PROFILE:
        try:
            from transformers import FineGrainedFP8Config
        except ImportError as exc:
            raise RuntimeError(
                "The Ministral training profile requires Transformers with "
                "FineGrainedFP8Config support."
            ) from exc
        model_init_kwargs["quantization_config"] = FineGrainedFP8Config(
            dequantize=True
        )
    args = DPOConfig(
        output_dir=str(run_dir / "checkpoints"),
        run_name=f"{config.run_name}_{dataset_version}",
        model_init_kwargs=model_init_kwargs,
        trust_remote_code=config.model.trust_remote_code,
        per_device_train_batch_size=trainer_config.per_device_train_batch_size,
        per_device_eval_batch_size=trainer_config.per_device_eval_batch_size,
        gradient_accumulation_steps=trainer_config.gradient_accumulation_steps,
        num_train_epochs=trainer_config.num_train_epochs,
        learning_rate=trainer_config.learning_rate,
        lr_scheduler_type=trainer_config.lr_scheduler_type,
        warmup_ratio=trainer_config.warmup_ratio,
        optim=trainer_config.optimizer,
        max_grad_norm=trainer_config.max_grad_norm,
        bf16=True,
        fp16=False,
        gradient_checkpointing=trainer_config.gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": trainer_config.gradient_checkpointing_use_reentrant
        },
        use_cache=trainer_config.use_cache,
        logging_strategy=trainer_config.logging_strategy,
        logging_steps=trainer_config.logging_steps,
        logging_first_step=trainer_config.logging_first_step,
        report_to=trainer_config.report_to,
        eval_strategy=trainer_config.eval_strategy,
        save_strategy=trainer_config.save_strategy,
        save_steps=trainer_config.save_steps,
        save_total_limit=trainer_config.save_total_limit,
        seed=config.split.seed,
        data_seed=config.split.seed,
        dataloader_num_workers=trainer_config.dataloader_num_workers,
        max_length=trainer_config.max_length,
        precompute_ref_log_probs=trainer_config.precompute_ref_log_probs,
        loss_type=[trainer_config.loss_type],
        beta=trainer_config.beta,
        push_to_hub=trainer_config.push_to_hub,
        disable_dropout=trainer_config.disable_dropout,
    )
    factory = trainer_factory or DPOTrainer
    trainer = factory(
        model=str(config.model.path),
        ref_model=None,
        args=args,
        train_dataset=Dataset.from_list(rendered_train),
        eval_dataset=Dataset.from_list(rendered_test),
        processing_class=tokenizer,
    )
    if config.model.training_profile == MINISTRAL_TRAINING_PROFILE:
        frozen_components = _freeze_ministral_text_only_components(
            trainer.model
        )
        setattr(trainer, "_dpo_frozen_components", frozen_components)
    return trainer


def _freeze_ministral_text_only_components(model: Any) -> dict[str, Any]:
    multimodal_model = getattr(model, "model", None)
    if multimodal_model is None:
        raise ValueError(
            "Ministral model has no top-level 'model' component."
        )
    component_metadata: dict[str, dict[str, int]] = {}
    for component_name in ("vision_tower", "multi_modal_projector"):
        component = getattr(multimodal_model, component_name, None)
        if component is None:
            raise ValueError(
                f"Ministral model is missing model.{component_name}."
            )
        parameters = list(component.parameters())
        if not parameters:
            raise ValueError(
                f"Ministral model component model.{component_name} has no "
                "parameters."
            )
        parameter_count = sum(parameter.numel() for parameter in parameters)
        for parameter in parameters:
            parameter.requires_grad = False
        if any(parameter.requires_grad for parameter in parameters):
            raise ValueError(
                f"Could not freeze Ministral model.{component_name}."
            )
        component_metadata[f"model.{component_name}"] = {
            "parameter_tensors": len(parameters),
            "parameter_count": parameter_count,
        }

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    unexpected = [
        name
        for name in trainable_names
        if not (
            name.startswith("model.language_model.")
            or name.startswith("lm_head.")
        )
    ]
    if unexpected:
        raise ValueError(
            "Ministral text-only profile found trainable parameters outside the "
            "language model: "
            + ", ".join(unexpected[:10])
        )
    if not trainable_names:
        raise ValueError("Ministral text-only profile has no trainable parameters.")
    return {
        "training_profile": MINISTRAL_TRAINING_PROFILE,
        "components": component_metadata,
        "trainable_parameter_tensors": len(trainable_names),
        "trainable_parameter_count": sum(
            parameter.numel()
            for _name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
    }


def _start_run(
    config: DPOTrainingConfig,
    *,
    dataset_version: str,
    resume: Path | None,
) -> tuple[Path, dict[str, Any], bool]:
    if resume is not None:
        run_dir = resume.resolve()
        if not run_dir.is_dir():
            raise ValueError(f"Resume run directory does not exist: {run_dir}")
        manifest = _read_json(run_dir / MANIFEST_FILENAME)
        if manifest.get("dataset_version") != dataset_version:
            raise ValueError(
                "Resume dataset version does not match the requested version."
            )
        if manifest.get("run_state", {}).get("status") == "complete":
            raise ValueError("The requested DPO training run is already complete.")
        return run_dir, manifest, True

    config.output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = f"{config.run_name}_{dataset_version}_{timestamp}"
    run_dir = config.output_root / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = config.output_root / f"{base_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=False, exist_ok=False)
    now = _utc_now()
    manifest = {
        "schema_version": "dpo_training_run_v1",
        "dataset_version": dataset_version,
        "config": config_to_jsonable(config),
        "run_state": {
            "status": "created",
            "started_at_utc": now,
            "updated_at_utc": now,
            "completed_at_utc": None,
        },
    }
    _write_manifest(run_dir / MANIFEST_FILENAME, manifest)
    return run_dir, manifest, False


def _split_rendered(
    rendered: list[RenderedExample], split_manifest: dict[str, Any]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if split_manifest.get("strategy", {}).get("name") == "predefined_files":
        train = [
            example.row for example in rendered if example.audit.get("split") == "train"
        ]
        test = [
            example.row for example in rendered if example.audit.get("split") == "test"
        ]
        if len(train) != split_manifest["counts"]["train_pair_count"]:
            raise ValueError("Rendered predefined train count does not match manifest.")
        if len(test) != split_manifest["counts"]["test_pair_count"]:
            raise ValueError("Rendered predefined test count does not match manifest.")
        return train, test
    test_lines = set(split_manifest["test"]["line_numbers"])
    train: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for example in rendered:
        (test if example.audit["line_number"] in test_lines else train).append(
            example.row
        )
    if len(train) != split_manifest["counts"]["train_pair_count"]:
        raise ValueError("Rendered train count does not match split manifest.")
    if len(test) != split_manifest["counts"]["test_pair_count"]:
        raise ValueError("Rendered test count does not match split manifest.")
    return train, test


def _write_rendered_snapshot(
    path: Path, rows: list[dict[str, str]]
) -> dict[str, Any]:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return {
        "path": str(path),
        "row_count": len(rows),
        "byte_count": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _read_rendered_snapshot(
    path: Path, metadata: dict[str, Any]
) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Rendered resume snapshot does not exist: {path}")
    if file_sha256(path) != metadata.get("sha256"):
        raise ValueError(f"Rendered resume snapshot checksum mismatch: {path}")
    rows: list[dict[str, str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid rendered snapshot JSON at {path}:{line_number}: {exc}"
                ) from exc
            if (
                not isinstance(row, dict)
                or set(row) != {"prompt", "chosen", "rejected"}
                or not all(isinstance(value, str) for value in row.values())
            ):
                raise ValueError(f"Invalid rendered snapshot row at {path}:{line_number}.")
            rows.append(row)
    if len(rows) != metadata.get("row_count"):
        raise ValueError(f"Rendered resume snapshot row count mismatch: {path}")
    return rows


def _read_token_profile(
    path: Path, metadata: Any
) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("Resume manifest has no token-profile metadata.")
    if not path.is_file():
        raise ValueError(f"Token profile does not exist: {path}")
    if file_sha256(path) != metadata.get("sha256"):
        raise ValueError(f"Token-profile checksum mismatch: {path}")
    profile = _read_json(path)
    saved_profile_sha = profile.get("profile_sha256")
    if (
        not isinstance(saved_profile_sha, str)
        or saved_profile_sha != metadata.get("profile_sha256")
    ):
        raise ValueError("Token-profile canonical checksum mismatch.")
    unsigned_profile = dict(profile)
    unsigned_profile.pop("profile_sha256", None)
    if canonical_json_sha256(unsigned_profile) != saved_profile_sha:
        raise ValueError("Token-profile content does not match its canonical checksum.")
    return profile


def _validate_training_dependencies(
    config: DPOTrainingConfig | None = None,
) -> None:
    training_profile = (
        config.model.training_profile if config is not None else None
    )
    minimums = {
        "trl": (1, 9, 0),
        "transformers": (
            (5, 14, 1)
            if training_profile == MINISTRAL_TRAINING_PROFILE
            else (4, 56, 1)
        ),
        "datasets": (4, 1, 0),
        "accelerate": (1, 10, 1),
        "torch": (2, 8, 0),
    }
    if training_profile == MINISTRAL_TRAINING_PROFILE:
        minimums["mistral-common"] = (1, 8, 6)
    errors: list[str] = []
    installed: dict[str, tuple[str, tuple[int, int, int]]] = {}
    for package, minimum in minimums.items():
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{package} is not installed")
            continue
        parsed = _version_tuple(version)
        installed[package] = (version, parsed)
        if parsed < minimum:
            errors.append(
                f"{package} {version} is older than {'.'.join(map(str, minimum))}"
            )
    trl_version = installed.get("trl")
    if trl_version is not None and trl_version[1] != (1, 9, 0):
        errors.append(f"trl {trl_version[0]} is not the pinned version 1.9.0")
    transformers_version = installed.get("transformers")
    if transformers_version is not None and transformers_version[1] >= (6, 0, 0):
        errors.append(
            f"transformers {transformers_version[0]} is outside the supported <6 range"
        )
    if errors:
        raise RuntimeError("Training dependency check failed: " + "; ".join(errors))


def _version_tuple(version: str) -> tuple[int, int, int]:
    numbers: list[int] = []
    for part in version.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        numbers.append(int(digits))
        if len(numbers) == 3:
            break
    return tuple((numbers + [0, 0, 0])[:3])  # type: ignore[return-value]


def _latest_checkpoint(checkpoint_root: Path) -> Path:
    checkpoints = _checkpoint_paths(checkpoint_root)
    if not checkpoints:
        raise ValueError(f"Resume run has no checkpoints: {checkpoint_root}")
    return checkpoints[-1]


def _checkpoint_paths(checkpoint_root: Path) -> list[Path]:
    if not checkpoint_root.is_dir():
        return []
    values: list[tuple[int, Path]] = []
    for path in checkpoint_root.glob("checkpoint-*"):
        if path.is_dir():
            try:
                step = int(path.name.removeprefix("checkpoint-"))
            except ValueError:
                continue
            values.append((step, path))
    return [path for _step, path in sorted(values)]


def _checkpoint_inventory(checkpoint_root: Path) -> list[dict[str, Any]]:
    return [
        {"path": str(path), "step": int(path.name.removeprefix("checkpoint-"))}
        for path in _checkpoint_paths(checkpoint_root)
    ]


def _directory_hashes(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {
            "byte_count": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _environment_metadata(
    config: DPOTrainingConfig | None = None,
) -> dict[str, Any]:
    training_profile = (
        config.model.training_profile if config is not None else None
    )
    packages = {}
    package_names = ["trl", "transformers", "datasets", "accelerate", "torch"]
    if training_profile == MINISTRAL_TRAINING_PROFILE:
        package_names.append("mistral-common")
    for package in package_names:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        metadata["cuda"] = {
            "available": torch.cuda.is_available(),
            "runtime_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        metadata["cuda"] = {"available": False}
    return metadata


def _set_state(
    manifest: dict[str, Any], status: str, *, completed: bool = False
) -> None:
    now = _utc_now()
    state = manifest.setdefault("run_state", {})
    state["status"] = status
    state["updated_at_utc"] = now
    if completed:
        state["completed_at_utc"] = now


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write_json(temporary, manifest)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _print_preflight_summary(
    run_dir: Path,
    dataset_version: str,
    profile: dict[str, Any],
    split_manifest: dict[str, Any],
) -> None:
    print("DPO training preflight summary")
    print(f"  dataset_version={dataset_version}")
    print(
        f"  train_pairs={split_manifest['counts']['train_pair_count']} "
        f"test_pairs={split_manifest['counts']['test_pair_count']}"
    )
    overall = profile["statistics"]["__all__"]["maximum_sequence"]
    print(
        "  tokens "
        f"min={overall['min']} p50={overall['p50']} p90={overall['p90']} "
        f"p95={overall['p95']} p99={overall['p99']} max={overall['max']} "
        f"limit={profile['enforced_model_limit']}"
    )
    print(f"  truncation=false over_limit={profile['over_limit_count']}")
    print(f"  run_dir={run_dir}")


def _print_training_summary(
    run_dir: Path,
    dataset_version: str,
    split_manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    print("DPO training summary")
    print(f"  dataset_version={dataset_version}")
    print(
        f"  train_pairs={split_manifest['counts']['train_pair_count']} "
        f"test_pairs={split_manifest['counts']['test_pair_count']}"
    )
    print(f"  eval_before={json.dumps(metrics['eval_before'], sort_keys=True)}")
    print(f"  eval_after={json.dumps(metrics['eval_after'], sort_keys=True)}")
    print(f"  final_model={run_dir / 'final_model'}")
    print(f"  manifest={run_dir / MANIFEST_FILENAME}")
