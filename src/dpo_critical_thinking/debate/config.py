from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    dataset: str
    enriched_parent_path: Path | None = None
    enriched_run_path: Path | None = None


@dataclass(frozen=True, slots=True)
class AgentConfig:
    id: str
    name: str
    role: str
    backend: str
    model_path: str | None
    prompt_path: Path | None = None
    torch_dtype: str = "auto"
    device_map: str = "auto"
    trust_remote_code: bool = False


@dataclass(frozen=True, slots=True)
class TurnConfig:
    id: str
    agent_id: str
    role: str
    prompt_path: Path
    contributes_to_aggregation: bool = False


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    json_retry_attempts: int = 2
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class DebateConfig:
    review_pack_path: Path
    output_root: Path
    datasets: tuple[DatasetConfig, ...]
    agents: tuple[AgentConfig, ...]
    turns: tuple[TurnConfig, ...]
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    review_blocks: tuple[str, ...] = ()
    ranking_method: str = "multi_agent_fine_grained_ranking"
    limit: int | None = None
    run_name: str = "qwen_32b_72b_multi_agent_debate"


def load_debate_config(path: Path) -> DebateConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return debate_config_from_mapping(payload, base_dir=path.parent)


def debate_config_from_mapping(payload: dict[str, Any], *, base_dir: Path) -> DebateConfig:
    datasets = tuple(
        DatasetConfig(
            dataset=str(item["dataset"]),
            enriched_parent_path=_optional_path(item.get("enriched_parent_path"), base_dir),
            enriched_run_path=_optional_path(item.get("enriched_run_path"), base_dir),
        )
        for item in payload.get("datasets", [])
    )
    if not datasets:
        raise ValueError("Config must include at least one dataset.")

    agents = tuple(
        AgentConfig(
            id=str(item["id"]),
            name=str(item.get("name", item["id"])),
            role=str(item.get("role", item["id"])),
            backend=str(item.get("backend", "transformers")),
            model_path=item.get("model_path"),
            prompt_path=_optional_path(item.get("prompt_path"), base_dir),
            torch_dtype=str(item.get("torch_dtype", "auto")),
            device_map=str(item.get("device_map", "auto")),
            trust_remote_code=bool(item.get("trust_remote_code", False)),
        )
        for item in payload.get("agents", [])
    )
    if len(agents) != 2:
        raise ValueError("Config must include exactly two agents for this debate flow.")

    agent_ids = {agent.id for agent in agents}
    turns = tuple(
        TurnConfig(
            id=str(item["id"]),
            agent_id=str(item["agent_id"]),
            role=str(item["role"]),
            prompt_path=_required_path(item["prompt_path"], base_dir),
            contributes_to_aggregation=bool(
                item.get("contributes_to_aggregation", False)
            ),
        )
        for item in payload.get("turns", [])
    )
    if len(turns) != 4:
        raise ValueError("Config must include exactly four debate turns.")
    unknown_agents = [turn.agent_id for turn in turns if turn.agent_id not in agent_ids]
    if unknown_agents:
        raise ValueError(f"Turn references unknown agent ids: {unknown_agents}")
    if sum(1 for turn in turns if turn.contributes_to_aggregation) != 2:
        raise ValueError("Exactly two turns must contribute to aggregation.")

    generation_payload = payload.get("generation", {})
    generation = GenerationConfig(
        max_new_tokens=int(generation_payload.get("max_new_tokens", 2048)),
        temperature=float(generation_payload.get("temperature", 0.0)),
        top_p=float(generation_payload.get("top_p", 1.0)),
        do_sample=bool(generation_payload.get("do_sample", False)),
        json_retry_attempts=int(generation_payload.get("json_retry_attempts", 2)),
        seed=(
            int(generation_payload["seed"])
            if generation_payload.get("seed") is not None
            else None
        ),
    )

    return DebateConfig(
        review_pack_path=_required_path(payload["review_pack_path"], base_dir),
        output_root=_required_path(payload["output_root"], base_dir),
        datasets=datasets,
        agents=agents,
        turns=turns,
        generation=generation,
        review_blocks=tuple(str(item) for item in payload.get("review_blocks", [])),
        ranking_method=str(
            payload.get("ranking_method", "multi_agent_fine_grained_ranking")
        ),
        limit=(int(payload["limit"]) if payload.get("limit") is not None else None),
        run_name=str(payload.get("run_name", "qwen_32b_72b_multi_agent_debate")),
    )


def config_to_jsonable(config: DebateConfig) -> dict[str, Any]:
    return {
        "review_pack_path": str(config.review_pack_path),
        "output_root": str(config.output_root),
        "datasets": [
            {
                "dataset": item.dataset,
                "enriched_parent_path": (
                    str(item.enriched_parent_path) if item.enriched_parent_path else None
                ),
                "enriched_run_path": (
                    str(item.enriched_run_path) if item.enriched_run_path else None
                ),
            }
            for item in config.datasets
        ],
        "agents": [
            {
                "id": item.id,
                "name": item.name,
                "role": item.role,
                "backend": item.backend,
                "model_path": item.model_path,
                "prompt_path": str(item.prompt_path) if item.prompt_path else None,
                "torch_dtype": item.torch_dtype,
                "device_map": item.device_map,
                "trust_remote_code": item.trust_remote_code,
            }
            for item in config.agents
        ],
        "turns": [
            {
                "id": item.id,
                "agent_id": item.agent_id,
                "role": item.role,
                "prompt_path": str(item.prompt_path),
                "contributes_to_aggregation": item.contributes_to_aggregation,
            }
            for item in config.turns
        ],
        "generation": {
            "max_new_tokens": config.generation.max_new_tokens,
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "do_sample": config.generation.do_sample,
            "json_retry_attempts": config.generation.json_retry_attempts,
            "seed": config.generation.seed,
        },
        "review_blocks": list(config.review_blocks),
        "ranking_method": config.ranking_method,
        "limit": config.limit,
        "run_name": config.run_name,
    }


def _required_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _optional_path(value: str | None, base_dir: Path) -> Path | None:
    return _required_path(value, base_dir) if value else None
