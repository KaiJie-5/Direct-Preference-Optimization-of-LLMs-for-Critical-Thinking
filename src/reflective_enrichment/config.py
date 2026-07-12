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
    ranking_run_dir: Path
    review_pack_path: Path
    output_root: Path
    prompt_path: Path
    teacher: TeacherConfig
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
    config = ReflectiveConfig(
        ranking_run_dir=_required_path(payload, "ranking_run_dir", base_dir),
        review_pack_path=_required_path(payload, "review_pack_path", base_dir),
        output_root=_required_path(payload, "output_root", base_dir),
        prompt_path=_required_path(payload, "prompt_path", base_dir),
        teacher=teacher,
        generation=generation,
        run_name=str(payload.get("run_name", "reflective_questions_enrichment")),
        limit=(int(payload["limit"]) if payload.get("limit") is not None else None),
    )
    _validate(config)
    return config


def config_to_jsonable(config: ReflectiveConfig) -> dict[str, Any]:
    return {
        "ranking_run_dir": str(config.ranking_run_dir),
        "review_pack_path": str(config.review_pack_path),
        "output_root": str(config.output_root),
        "prompt_path": str(config.prompt_path),
        "teacher": asdict(config.teacher),
        "generation": {**asdict(config.generation), "stop": list(config.generation.stop)},
        "run_name": config.run_name,
        "limit": config.limit,
    }


def _required_path(payload: dict[str, Any], key: str, base_dir: Path) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field {key!r} must be a non-empty path string.")
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Config field {name!r} must be an object.")
    return dict(value)


def _validate(config: ReflectiveConfig) -> None:
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
