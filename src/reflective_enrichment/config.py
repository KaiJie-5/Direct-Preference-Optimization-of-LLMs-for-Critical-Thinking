from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TeacherConfig:
    backend: str = "transformers"
    model_path: str | None = None
    model_name: str | None = None
    torch_dtype: str = "auto"
    device_map: Any = "auto"
    trust_remote_code: bool = False
    use_chat_template: bool = True
    force_think_prefix: bool = True
    think_prefix: str = "<think>\n"
    api_base: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 600.0


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    max_new_tokens: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()
    json_repair_attempts: int = 2


@dataclass(frozen=True, slots=True)
class ReflectiveConfig:
    # Keep the historical leading field order for callers that instantiate this
    # public dataclass positionally. Direct single-pass callers pass both as None.
    ranking_run_dir: Path | None
    review_pack_path: Path | None
    output_root: Path
    prompt_path: Path
    teacher: TeacherConfig
    input_mode: str = "ranked"
    single_pass_run_dir: Path | None = None
    input_status_policy: str = "successful_only"
    context_scope: str = "full_interview"
    context_turns_before: int = 20
    context_turns_after: int = 20
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    run_name: str = "reflective_questions_enrichment"
    limit: int | None = None


def load_reflective_config(path: Path) -> ReflectiveConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Reflective enrichment config must be a JSON object.")
    base_dir = path.parent
    teacher_payload = _object(payload.get("teacher", {}), "teacher")
    generation_payload = _object(payload.get("generation", {}), "generation")
    teacher = TeacherConfig(**teacher_payload)
    generation = GenerationConfig(
        **{**generation_payload, "stop": tuple(generation_payload.get("stop", ()))},
    )
    input_mode = str(payload.get("input_mode", "ranked"))
    config = ReflectiveConfig(
        output_root=_required_path(payload, "output_root", base_dir),
        prompt_path=_required_path(payload, "prompt_path", base_dir),
        teacher=teacher,
        input_mode=input_mode,
        ranking_run_dir=_optional_path(payload, "ranking_run_dir", base_dir),
        review_pack_path=_optional_path(payload, "review_pack_path", base_dir),
        single_pass_run_dir=_optional_path(payload, "single_pass_run_dir", base_dir),
        input_status_policy=str(payload.get("input_status_policy", "successful_only")),
        context_scope=str(payload.get("context_scope", "full_interview")),
        context_turns_before=int(payload.get("context_turns_before", 20)),
        context_turns_after=int(payload.get("context_turns_after", 20)),
        generation=generation,
        run_name=str(payload.get("run_name", "reflective_questions_enrichment")),
        limit=(int(payload["limit"]) if payload.get("limit") is not None else None),
    )
    _validate(config)
    return config


def config_to_jsonable(config: ReflectiveConfig) -> dict[str, Any]:
    payload = {
        "output_root": str(config.output_root),
        "prompt_path": str(config.prompt_path),
        "teacher": asdict(config.teacher),
        "generation": {**asdict(config.generation), "stop": list(config.generation.stop)},
        "run_name": config.run_name,
        "limit": config.limit,
    }
    if config.input_mode == "ranked":
        # Keep the historical shape byte-for-byte comparable at the value level so
        # existing strict resume manifests remain valid.
        return {
            "ranking_run_dir": str(config.ranking_run_dir),
            "review_pack_path": str(config.review_pack_path),
            **payload,
        }
    return {
        "input_mode": config.input_mode,
        "single_pass_run_dir": str(config.single_pass_run_dir),
        "input_status_policy": config.input_status_policy,
        "context_scope": config.context_scope,
        "context_turns_before": config.context_turns_before,
        "context_turns_after": config.context_turns_after,
        **payload,
    }


def _required_path(payload: dict[str, Any], key: str, base_dir: Path) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field {key!r} must be a non-empty path string.")
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _optional_path(payload: dict[str, Any], key: str, base_dir: Path) -> Path | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field {key!r} must be a non-empty path string.")
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Config field {name!r} must be an object.")
    return dict(value)


def _validate(config: ReflectiveConfig) -> None:
    if config.input_mode not in {"ranked", "single_pass"}:
        raise ValueError(f"Unsupported reflective input_mode: {config.input_mode!r}")
    if config.input_mode == "ranked":
        if config.ranking_run_dir is None or config.review_pack_path is None:
            raise ValueError(
                "ranking_run_dir and review_pack_path are required for input_mode='ranked'."
            )
        if config.single_pass_run_dir is not None:
            raise ValueError(
                "single_pass_run_dir cannot be used with input_mode='ranked'."
            )
    else:
        if config.single_pass_run_dir is None:
            raise ValueError(
                "single_pass_run_dir is required for input_mode='single_pass'."
            )
        if config.ranking_run_dir is not None or config.review_pack_path is not None:
            raise ValueError(
                "ranking_run_dir/review_pack_path cannot be used with input_mode='single_pass'."
            )
        if config.input_status_policy != "successful_only":
            raise ValueError(
                "input_mode='single_pass' currently requires "
                "input_status_policy='successful_only'."
            )
        if config.context_scope != "turn_window":
            raise ValueError(
                "input_mode='single_pass' currently requires context_scope='turn_window'."
            )
        for name, value in (
            ("context_turns_before", config.context_turns_before),
            ("context_turns_after", config.context_turns_after),
        ):
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
    if config.teacher.backend not in {"dry-run", "transformers", "openai-compatible"}:
        raise ValueError(f"Unsupported teacher backend: {config.teacher.backend!r}")
    if config.teacher.backend == "transformers" and not config.teacher.model_path:
        raise ValueError("teacher.model_path is required for transformers.")
    if config.generation.max_new_tokens <= 0:
        raise ValueError("generation.max_new_tokens must be positive.")
    if config.generation.json_repair_attempts < 0:
        raise ValueError("generation.json_repair_attempts must be non-negative.")
    if config.limit is not None and config.limit <= 0:
        raise ValueError("limit must be positive when supplied.")
    if not config.run_name.strip():
        raise ValueError("run_name must be non-empty.")
