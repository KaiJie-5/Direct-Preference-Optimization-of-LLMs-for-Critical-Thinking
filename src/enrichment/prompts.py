from __future__ import annotations

from pathlib import Path
from string import Formatter
from typing import Any


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class PromptTemplate:
    """Small file-backed template that leaves unknown placeholders visible."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.template = path.read_text(encoding="utf-8")

    def render(self, variables: dict[str, Any]) -> str:
        return self.template.format_map(_SafeFormatDict(variables))

    def uses_variable(self, name: str) -> bool:
        return any(
            field_name == name
            for _, field_name, _, _ in Formatter().parse(self.template)
            if field_name is not None
        )


def parse_prompt_vars(values: list[str] | None) -> dict[str, str]:
    variables: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                f"Prompt variable {value!r} must be in KEY=VALUE format."
            )
        key, raw = value.split("=", 1)
        variables[key] = raw
    return variables


def align_prompt_context_contract(rendered_prompt: str, context_scope: str) -> str:
    """Align the existing v2 JSON example for additive windowed context runs.

    The checked-in full-interview prompt remains unchanged for historical runs. Only
    the new turn-window runtime replaces its literal schema example after rendering.
    Custom prompts that already use the runtime scope are left untouched.
    """

    if context_scope != "turn_window":
        return rendered_prompt
    return rendered_prompt.replace(
        '"analysis_context_scope": "full_interview"',
        '"analysis_context_scope": "turn_window"',
    )
