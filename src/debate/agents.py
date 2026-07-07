from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .config import AgentConfig, GenerationConfig


@dataclass(frozen=True, slots=True)
class DebateGenerationResult:
    text: str
    raw: dict[str, Any]
    rendered_prompt: str
    elapsed_seconds: float


class DebateAgent(Protocol):
    agent_id: str
    name: str

    def generate(
        self,
        messages: list[dict[str, str]],
        generation: GenerationConfig,
    ) -> DebateGenerationResult:
        ...

    def metadata(self) -> dict[str, Any]:
        ...


class DryRunDebateAgent:
    def __init__(self, *, agent_id: str, name: str, ranking: list[str] | None = None) -> None:
        self.agent_id = agent_id
        self.name = name
        self.ranking = ranking or ["A", "B", "C", "D", "E"]

    def generate(
        self,
        messages: list[dict[str, str]],
        generation: GenerationConfig,
    ) -> DebateGenerationResult:
        started = time.perf_counter()
        candidate_labels = _message_json_list(messages, "Candidate labels JSON")
        ranking = candidate_labels or self.ranking
        text = json.dumps(
            {
                "record_id": _message_value(messages, "Record ID"),
                "review_block": _message_value(messages, "Review block"),
                "candidate_assessments": {
                    label: f"Dry-run assessment for Candidate {label}."
                    for label in ranking
                },
                "debate_response": "Dry-run debate response for smoke testing.",
                "uncertainty": "Dry-run uncertainty for smoke testing.",
                "ranking": ranking,
                "rationale": "Dry-run ranking for smoke testing.",
            }
        )
        return DebateGenerationResult(
            text=text,
            raw={"backend": "dry-run", "message_count": len(messages)},
            rendered_prompt="\n\n".join(item["content"] for item in messages),
            elapsed_seconds=time.perf_counter() - started,
        )

    def metadata(self) -> dict[str, Any]:
        return {"backend": "dry-run", "agent_id": self.agent_id, "name": self.name}


class TransformersChatAgent:
    def __init__(self, config: AgentConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install the transformers optional dependencies before using this backend: "
                "python -m pip install -e .[transformers]"
            ) from exc

        if not config.model_path:
            raise ValueError(f"Agent {config.id} requires model_path.")

        self.agent_id = config.id
        self.name = config.name
        self.model_path = config.model_path
        self.torch_dtype = config.torch_dtype
        self.device_map = config.device_map
        self.max_memory = config.max_memory
        self.trust_remote_code = config.trust_remote_code
        self._torch = torch
        dtype = _resolve_torch_dtype(torch, config.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": config.device_map,
            "trust_remote_code": config.trust_remote_code,
        }
        if config.max_memory is not None:
            model_kwargs["max_memory"] = config.max_memory
        self.model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_kwargs)
        self.hf_device_map = getattr(self.model, "hf_device_map", None)
        if config.max_memory is not None:
            _validate_loaded_device_map(
                self.hf_device_map,
                allowed_gpu_indexes=set(config.max_memory),
                agent_id=config.id,
            )
        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        generation: GenerationConfig,
    ) -> DebateGenerationResult:
        started = time.perf_counter()
        if generation.seed is not None:
            random.seed(generation.seed)
            self._torch.manual_seed(generation.seed)
            if self._torch.cuda.is_available():
                self._torch.cuda.manual_seed_all(generation.seed)

        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        prompt_token_count = int(inputs["input_ids"].shape[-1])
        max_new_tokens = generation.max_new_tokens
        model_config = getattr(self.model, "config", None)
        context_window = getattr(model_config, "max_position_embeddings", None)
        if isinstance(context_window, int):
            remaining_tokens = context_window - prompt_token_count
            if remaining_tokens <= 0:
                raise ValueError(
                    f"Prompt has {prompt_token_count} tokens, which reaches or exceeds "
                    f"the model context window of {context_window}."
                )
            max_new_tokens = min(max_new_tokens, remaining_tokens)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": generation.do_sample,
        }
        native_generation = self.model.generation_config
        eos_token_id = native_generation.eos_token_id
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        pad_token_id = native_generation.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if generation.do_sample:
            generation_kwargs["temperature"] = generation.temperature
            generation_kwargs["top_p"] = generation.top_p

        with self._torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        new_tokens = output_ids[0][prompt_token_count:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return DebateGenerationResult(
            text=text,
            raw={
                "backend": "transformers",
                "model_path": self.model_path,
                "generation_kwargs": generation_kwargs,
                "configured_max_new_tokens": generation.max_new_tokens,
                "model_context_window": context_window,
                "prompt_token_count": prompt_token_count,
                "new_token_count": int(new_tokens.shape[-1]),
            },
            rendered_prompt=rendered_prompt,
            elapsed_seconds=time.perf_counter() - started,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "transformers",
            "agent_id": self.agent_id,
            "name": self.name,
            "model_path": self.model_path,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "max_memory": self.max_memory,
            "hf_device_map": self.hf_device_map,
            "trust_remote_code": self.trust_remote_code,
        }


def build_agent(config: AgentConfig) -> DebateAgent:
    if config.backend == "dry-run":
        return DryRunDebateAgent(agent_id=config.id, name=config.name)
    if config.backend == "transformers":
        return TransformersChatAgent(config)
    raise ValueError(f"Unsupported debate agent backend: {config.backend}")


def _resolve_torch_dtype(torch: Any, torch_dtype: str) -> Any:
    if torch_dtype == "auto":
        return "auto"
    if not hasattr(torch, torch_dtype):
        raise ValueError(f"Unknown torch dtype: {torch_dtype}")
    return getattr(torch, torch_dtype)


def _validate_loaded_device_map(
    hf_device_map: Any,
    *,
    allowed_gpu_indexes: set[int],
    agent_id: str,
) -> None:
    if not isinstance(hf_device_map, dict) or not hf_device_map:
        raise ValueError(
            f"Agent {agent_id} was configured with max_memory, but the loaded model "
            "did not expose a non-empty hf_device_map to verify placement."
        )

    used_gpu_indexes: set[int] = set()
    for module_name, raw_device in hf_device_map.items():
        device = _device_index_from_loaded_map_value(raw_device)
        if device is None:
            raise ValueError(
                f"Agent {agent_id} placed module {module_name!r} on unsupported "
                f"device {raw_device!r}; CPU/disk/offload placements are not allowed."
            )
        used_gpu_indexes.add(device)

    unexpected_gpu_indexes = used_gpu_indexes - allowed_gpu_indexes
    if unexpected_gpu_indexes:
        raise ValueError(
            f"Agent {agent_id} used GPU indexes {sorted(unexpected_gpu_indexes)}, "
            f"outside allowed max_memory GPUs {sorted(allowed_gpu_indexes)}."
        )


def _device_index_from_loaded_map_value(raw_device: Any) -> int | None:
    if isinstance(raw_device, bool):
        return None
    if isinstance(raw_device, int):
        return raw_device if raw_device >= 0 else None

    device_text = str(raw_device).strip().lower()
    if device_text.isdecimal():
        return int(device_text)
    if device_text.startswith("cuda:"):
        device_index = device_text.removeprefix("cuda:")
        if device_index.isdecimal():
            return int(device_index)
    return None


def _message_value(messages: list[dict[str, str]], label: str) -> str:
    prefix = f"{label}:"
    for message in reversed(messages):
        for line in message["content"].splitlines():
            if line.startswith(prefix):
                return line.removeprefix(prefix).strip()
    return ""


def _message_json_list(messages: list[dict[str, str]], label: str) -> list[str]:
    value = _message_value(messages, label)
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        return []
    return parsed
