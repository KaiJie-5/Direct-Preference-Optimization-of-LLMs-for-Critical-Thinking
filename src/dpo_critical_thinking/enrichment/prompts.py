from __future__ import annotations

from pathlib import Path
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
