from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from .config import DPOTrainingConfig
from .data import PreferenceExample, add_system_message, canonical_json_sha256


@dataclass(frozen=True, slots=True)
class RenderedExample:
    row: dict[str, str]
    audit: dict[str, Any]


def load_tokenizer(config: DPOTrainingConfig) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required for DPO token preflight. Install the "
            "project's training optional dependencies."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.model.path),
        local_files_only=config.model.local_files_only,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The configured model tokenizer has no chat template.")
    return tokenizer


def validate_model_context_limit(config: DPOTrainingConfig) -> dict[str, int]:
    config_path = config.model.path / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read model config {config_path}: {exc}") from exc
    model_limit = payload.get("max_position_embeddings")
    if (
        isinstance(model_limit, bool)
        or not isinstance(model_limit, int)
        or model_limit <= 0
    ):
        raise ValueError("Model config has invalid max_position_embeddings.")
    if model_limit != config.model.max_position_embeddings:
        raise ValueError(
            "Configured model context limit does not match config.json: "
            f"{config.model.max_position_embeddings} != {model_limit}."
        )
    return {
        "model_max_position_embeddings": model_limit,
        "configured_limit": config.model.max_position_embeddings,
    }


def render_and_profile(
    examples: Iterable[PreferenceExample],
    *,
    tokenizer: Any,
    config: DPOTrainingConfig,
    dataset_version: str,
) -> tuple[list[RenderedExample], dict[str, Any]]:
    values = tuple(examples)
    rendered: list[RenderedExample] = []
    lengths: defaultdict[str, dict[str, list[int]]] = defaultdict(
        lambda: {
            "prompt": [],
            "chosen_sequence": [],
            "rejected_sequence": [],
            "maximum_sequence": [],
        }
    )
    over_limit: list[dict[str, Any]] = []
    render_date = date.today().isoformat()
    template = str(tokenizer.chat_template)
    template_sha = sha256(template.encode("utf-8")).hexdigest()

    for example in values:
        conversational = add_system_message(
            example.row, config.chat.system_message
        )
        prompt_messages = conversational["prompt"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        chosen_text = _render_completion(
            tokenizer,
            prompt_messages,
            conversational["chosen"],
            prompt_text,
            field="chosen",
            audit=example.audit,
        )
        rejected_text = _render_completion(
            tokenizer,
            prompt_messages,
            conversational["rejected"],
            prompt_text,
            field="rejected",
            audit=example.audit,
        )
        standard_row = {
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        }
        prompt_length = _token_count(tokenizer, prompt_text)
        chosen_length = _token_count(tokenizer, prompt_text + chosen_text)
        rejected_length = _token_count(tokenizer, prompt_text + rejected_text)
        maximum = max(chosen_length, rejected_length)
        dataset = example.audit["dataset"]
        for key in (dataset, "__all__"):
            lengths[key]["prompt"].append(prompt_length)
            lengths[key]["chosen_sequence"].append(chosen_length)
            lengths[key]["rejected_sequence"].append(rejected_length)
            lengths[key]["maximum_sequence"].append(maximum)
        if maximum > config.model.max_position_embeddings:
            over_limit.append(
                {
                    "line_number": example.audit["line_number"],
                    "pair_id": example.audit["pair_id"],
                    "dataset": dataset,
                    "record_id": example.audit["record_id"],
                    "transcript_id": example.audit["transcript_id"],
                    "chosen_tokens": chosen_length,
                    "rejected_tokens": rejected_length,
                    "limit": config.model.max_position_embeddings,
                }
            )
        rendered.append(RenderedExample(row=standard_row, audit=example.audit))

    profile = {
        "schema_version": "dpo_token_profile_v1",
        "dataset_version": dataset_version,
        "render_date": render_date,
        "native_date_metadata": config.chat.native_date_metadata,
        "system_message": config.chat.system_message,
        "chat_template_sha256": template_sha,
        "tokenizer_model_max_length": _safe_int(
            getattr(tokenizer, "model_max_length", None)
        ),
        "enforced_model_limit": config.model.max_position_embeddings,
        "max_length": None,
        "truncation": False,
        "row_count": len(rendered),
        "statistics": {
            dataset: {
                field: _statistics(field_lengths)
                for field, field_lengths in dataset_lengths.items()
            }
            for dataset, dataset_lengths in sorted(lengths.items())
        },
        "over_limit_count": len(over_limit),
        "over_limit_examples": over_limit,
    }
    profile["profile_sha256"] = canonical_json_sha256(profile)
    return rendered, profile


def assert_within_context_limit(profile: dict[str, Any]) -> None:
    if profile["over_limit_count"]:
        first = profile["over_limit_examples"][0]
        raise ValueError(
            f"{profile['over_limit_count']} rendered preference pairs exceed the "
            f"{profile['enforced_model_limit']}-token model limit. First: "
            f"{first['dataset']} {first['record_id']} line {first['line_number']}."
        )


def _render_completion(
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    completion_messages: list[dict[str, str]],
    prompt_text: str,
    *,
    field: str,
    audit: dict[str, Any],
) -> str:
    full_text = tokenizer.apply_chat_template(
        prompt_messages + completion_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "Model chat template is not prefix-preserving for "
            f"{field} at pair {audit['pair_id']}."
        )
    completion_text = full_text[len(prompt_text) :]
    source_content = completion_messages[0]["content"]
    if source_content not in completion_text:
        raise ValueError(
            f"Rendered {field} does not preserve source assistant content for "
            f"pair {audit['pair_id']}."
        )
    return completion_text


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    input_ids = encoded["input_ids"]
    return len(input_ids)


def _statistics(values: list[int]) -> dict[str, int]:
    if not values:
        return {
            "count": 0,
            "min": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _nearest_rank(ordered, 0.50),
        "p90": _nearest_rank(ordered, 0.90),
        "p95": _nearest_rank(ordered, 0.95),
        "p99": _nearest_rank(ordered, 0.99),
        "max": ordered[-1],
    }


def _nearest_rank(values: list[int], percentile: float) -> int:
    index = max(0, math.ceil(percentile * len(values)) - 1)
    return values[index]


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
