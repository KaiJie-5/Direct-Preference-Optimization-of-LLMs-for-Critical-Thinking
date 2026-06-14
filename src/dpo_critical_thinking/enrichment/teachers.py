from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Protocol


DEFAULT_MAX_NEW_TOKENS = 32768


@dataclass(slots=True)
class GenerationOptions:
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    stop: list[str] | None = None


@dataclass(slots=True)
class GenerationResult:
    text: str
    raw: dict[str, Any]
    rendered_prompt: str
    elapsed_seconds: float


class Teacher(Protocol):
    def generate(self, prompt: str, options: GenerationOptions) -> GenerationResult:
        ...

    def metadata(self) -> dict[str, Any]:
        ...


class DryRunTeacher:
    """Backend used for smoke tests when no model should be loaded."""

    def __init__(self, model_name: str = "dry-run-teacher") -> None:
        self.model_name = model_name

    def generate(self, prompt: str, options: GenerationOptions) -> GenerationResult:
        started = time.perf_counter()
        text = (
            "<think>\n"
            "Dry-run backend did not call a model.\n"
            "</think>\n\n"
            "[PLACEHOLDER_TEACHER_OUTPUT]"
        )
        return GenerationResult(
            text=text,
            raw={"backend": "dry-run", "generation_options": asdict(options)},
            rendered_prompt=prompt,
            elapsed_seconds=time.perf_counter() - started,
        )

    def metadata(self) -> dict[str, Any]:
        return {"backend": "dry-run", "model_name": self.model_name}


class TransformersTeacher:
    """Local Hugging Face causal-LM backend for teacher generation."""

    def __init__(
        self,
        *,
        model_path: str,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        trust_remote_code: bool = False,
        use_chat_template: bool = True,
        force_think_prefix: bool = True,
        think_prefix: str = "<think>\n",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install the transformers optional dependencies before using this backend: "
                "python -m pip install -e .[transformers]"
            ) from exc

        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.use_chat_template = use_chat_template
        self.force_think_prefix = force_think_prefix
        self.think_prefix = think_prefix
        self._torch = torch
        dtype = _resolve_torch_dtype(torch, torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    def generate(self, prompt: str, options: GenerationOptions) -> GenerationResult:
        started = time.perf_counter()
        if options.seed is not None:
            random.seed(options.seed)
            self._torch.manual_seed(options.seed)
            if self._torch.cuda.is_available():
                self._torch.cuda.manual_seed_all(options.seed)

        rendered_prompt = self._render_prompt(prompt)
        inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        prompt_token_count = int(inputs["input_ids"].shape[-1])
        token_budget = resolve_effective_max_new_tokens(
            prompt_token_count=prompt_token_count,
            requested_max_new_tokens=options.max_new_tokens,
            context_window=_model_context_window(self.model),
        )
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": token_budget["effective_max_new_tokens"],
            "temperature": options.temperature,
            "top_p": options.top_p,
            "do_sample": options.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if options.top_k is not None:
            generation_kwargs["top_k"] = options.top_k
        if options.repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = options.repetition_penalty

        with self._torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        new_tokens = output_ids[0][prompt_token_count:]
        generated = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        if self.force_think_prefix and not generated.startswith(self.think_prefix):
            generated = self.think_prefix + generated

        return GenerationResult(
            text=generated,
            raw={
                "backend": "transformers",
                "model_path": self.model_path,
                "generation_kwargs": generation_kwargs,
                "prompt_token_count": prompt_token_count,
                "new_token_count": int(new_tokens.shape[-1]),
                **token_budget,
            },
            rendered_prompt=rendered_prompt,
            elapsed_seconds=time.perf_counter() - started,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "transformers",
            "model_path": self.model_path,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "use_chat_template": self.use_chat_template,
            "force_think_prefix": self.force_think_prefix,
            "think_prefix": self.think_prefix,
        }

    def _render_prompt(self, prompt: str) -> str:
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = prompt

        if self.force_think_prefix:
            rendered += self.think_prefix
        return rendered


class OpenAICompatibleTeacher:
    """OpenAI-compatible HTTP backend for servers such as vLLM."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key_env: str | None = None,
        timeout_seconds: float = 600.0,
        force_think_prefix: bool = True,
        think_prefix: str = "<think>\n",
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.force_think_prefix = force_think_prefix
        self.think_prefix = think_prefix

    def generate(self, prompt: str, options: GenerationOptions) -> GenerationResult:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": options.max_new_tokens,
            "temperature": options.temperature,
            "top_p": options.top_p,
        }
        if options.stop:
            payload["stop"] = options.stop

        headers = {"Content-Type": "application/json"}
        if self.api_key_env and os.getenv(self.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"

        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible backend failed: {body}") from exc

        text = raw["choices"][0]["message"]["content"]
        if self.force_think_prefix and not text.startswith(self.think_prefix):
            text = self.think_prefix + text

        return GenerationResult(
            text=text,
            raw=raw,
            rendered_prompt=prompt,
            elapsed_seconds=time.perf_counter() - started,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "openai-compatible",
            "model": self.model,
            "api_base": self.api_base,
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            "force_think_prefix": self.force_think_prefix,
            "think_prefix": self.think_prefix,
        }


def build_teacher(args: Any) -> Teacher:
    if args.teacher_backend == "dry-run":
        return DryRunTeacher(model_name=args.model_name or "dry-run-teacher")
    if args.teacher_backend == "transformers":
        if not args.model_path:
            raise ValueError("--model-path is required for --teacher-backend transformers")
        return TransformersTeacher(
            model_path=args.model_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            use_chat_template=args.use_chat_template,
            force_think_prefix=args.force_think_prefix,
            think_prefix=args.think_prefix,
        )
    if args.teacher_backend == "openai-compatible":
        if not args.model_name:
            raise ValueError("--model-name is required for --teacher-backend openai-compatible")
        if not args.api_base:
            raise ValueError("--api-base is required for --teacher-backend openai-compatible")
        return OpenAICompatibleTeacher(
            model=args.model_name,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            timeout_seconds=args.timeout_seconds,
            force_think_prefix=args.force_think_prefix,
            think_prefix=args.think_prefix,
        )
    raise ValueError(f"Unsupported teacher backend: {args.teacher_backend}")


def _resolve_torch_dtype(torch: Any, torch_dtype: str) -> Any:
    if torch_dtype == "auto":
        return "auto"
    if not hasattr(torch, torch_dtype):
        raise ValueError(f"Unknown torch dtype: {torch_dtype}")
    return getattr(torch, torch_dtype)


def _model_context_window(model: Any) -> int | None:
    context_window = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    return context_window if isinstance(context_window, int) else None


def resolve_effective_max_new_tokens(
    *,
    prompt_token_count: int,
    requested_max_new_tokens: int,
    context_window: int | None,
) -> dict[str, Any]:
    if context_window is None:
        return {
            "requested_max_new_tokens": requested_max_new_tokens,
            "effective_max_new_tokens": requested_max_new_tokens,
            "context_window": None,
            "token_budget_clamped": False,
        }

    remaining_context = context_window - prompt_token_count
    if remaining_context <= 0:
        raise ValueError(
            "Prompt token count exceeds the model context window: "
            f"prompt_token_count={prompt_token_count}, context_window={context_window}."
        )

    effective_max_new_tokens = min(requested_max_new_tokens, remaining_context)
    return {
        "requested_max_new_tokens": requested_max_new_tokens,
        "effective_max_new_tokens": effective_max_new_tokens,
        "context_window": context_window,
        "token_budget_clamped": effective_max_new_tokens != requested_max_new_tokens,
    }
