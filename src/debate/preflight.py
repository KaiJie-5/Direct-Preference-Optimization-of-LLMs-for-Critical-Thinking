from __future__ import annotations

import json
from typing import Any

from enrichment.schema import parse_json_object

from .agents import build_agent
from .config import DebateConfig


def run_preflight(config: DebateConfig, *, generate_qwen_json: bool = False) -> None:
    agents = []
    for agent_config in config.agents:
        print(f"Loading agent {agent_config.id}: {agent_config.model_path}", flush=True)
        agent = build_agent(agent_config)
        agents.append(agent)
        print(json.dumps(agent.metadata(), indent=2, ensure_ascii=False), flush=True)
        _print_cuda_memory(f"after loading {agent_config.id}")

    if generate_qwen_json:
        qwen_agent = next(
            (agent for agent in agents if agent.agent_id == "qwen_72b"),
            None,
        )
        if qwen_agent is None:
            raise ValueError("No qwen_72b agent is configured for preflight generation.")
        result = qwen_agent.generate(
            [
                {
                    "role": "system",
                    "content": "Return strict JSON only.",
                },
                {
                    "role": "user",
                    "content": (
                        'Return exactly this JSON object and nothing else: {"status":"ok"}'
                    ),
                },
            ],
            config.generation,
        )
        parsed, parse_error = parse_json_object(result.text)
        if parse_error or not isinstance(parsed, dict) or parsed.get("status") != "ok":
            raise ValueError(
                "Qwen preflight did not produce the expected strict JSON object. "
                f"parse_error={parse_error!r}, text={result.text!r}"
            )
        print("Qwen strict JSON preflight passed.", flush=True)

    print("Debate model preflight complete.", flush=True)


def _print_cuda_memory(label: str) -> None:
    try:
        import torch
    except ImportError:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return
    print(f"CUDA memory {label}:", flush=True)
    for index in range(cuda.device_count()):
        allocated = cuda.memory_allocated(index) / (1024**3)
        reserved = cuda.memory_reserved(index) / (1024**3)
        print(
            f"  gpu {index}: allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB",
            flush=True,
        )
