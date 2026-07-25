from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceConfig:
    name: str
    trace_root: Path
    manifest_path: Path
    datasets: tuple[str, ...]
    context_scope: str
    context_turns_before: int | None = None
    context_turns_after: int | None = None


@dataclass(frozen=True, slots=True)
class PreferencePairConfig:
    sources: tuple[SourceConfig, ...]
    output_root: Path
    run_name: str = "reflective_question_dpo_pairs"
    expected_record_counts: tuple[tuple[str, int], ...] = ()

    def expected_counts(self) -> dict[str, int]:
        return dict(self.expected_record_counts)


def load_preference_pair_config(path: Path) -> PreferencePairConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read preference-pair config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Preference-pair config must be a JSON object.")

    base_dir = path.parent
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Config field 'sources' must be a non-empty list.")
    sources = tuple(
        _load_source(item, index=index, base_dir=base_dir)
        for index, item in enumerate(raw_sources)
    )
    expected_payload = payload.get("expected_record_counts", {})
    if not isinstance(expected_payload, dict):
        raise ValueError("Config field 'expected_record_counts' must be an object.")
    expected_counts: list[tuple[str, int]] = []
    for dataset, count in expected_payload.items():
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("Expected-count dataset names must be non-empty strings.")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"Expected record count for {dataset!r} must be a non-negative integer."
            )
        expected_counts.append((dataset, count))

    output_root = _required_path(payload, "output_root", base_dir)
    run_name = payload.get("run_name", "reflective_question_dpo_pairs")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("Config field 'run_name' must be a non-empty string.")
    config = PreferencePairConfig(
        sources=sources,
        output_root=output_root,
        run_name=run_name,
        expected_record_counts=tuple(expected_counts),
    )
    _validate(config)
    return config


def config_to_jsonable(config: PreferencePairConfig) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    for source in config.sources:
        item = asdict(source)
        item["trace_root"] = str(source.trace_root)
        item["manifest_path"] = str(source.manifest_path)
        item["datasets"] = list(source.datasets)
        sources.append(item)
    return {
        "sources": sources,
        "output_root": str(config.output_root),
        "run_name": config.run_name,
        "expected_record_counts": config.expected_counts(),
    }


def _load_source(value: Any, *, index: int, base_dir: Path) -> SourceConfig:
    if not isinstance(value, dict):
        raise ValueError(f"Config sources[{index}] must be an object.")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Config sources[{index}].name must be a non-empty string.")
    datasets = value.get("datasets")
    if (
        not isinstance(datasets, list)
        or not datasets
        or not all(isinstance(item, str) and item.strip() for item in datasets)
    ):
        raise ValueError(
            f"Config sources[{index}].datasets must be a non-empty string list."
        )
    if len(set(datasets)) != len(datasets):
        raise ValueError(f"Config sources[{index}].datasets contains duplicates.")
    context_scope = value.get("context_scope")
    if context_scope not in {"full_interview", "turn_window"}:
        raise ValueError(
            f"Config sources[{index}].context_scope must be "
            "'full_interview' or 'turn_window'."
        )
    before = _optional_nonnegative_int(
        value.get("context_turns_before"),
        f"sources[{index}].context_turns_before",
    )
    after = _optional_nonnegative_int(
        value.get("context_turns_after"),
        f"sources[{index}].context_turns_after",
    )
    if context_scope == "turn_window" and (before is None or after is None):
        raise ValueError(
            f"Config sources[{index}] turn_window requires both context turn counts."
        )
    if context_scope == "full_interview" and (before is not None or after is not None):
        raise ValueError(
            f"Config sources[{index}] full_interview cannot set context turn counts."
        )
    return SourceConfig(
        name=name,
        trace_root=_required_path(value, "trace_root", base_dir),
        manifest_path=_required_path(value, "manifest_path", base_dir),
        datasets=tuple(datasets),
        context_scope=context_scope,
        context_turns_before=before,
        context_turns_after=after,
    )


def _required_path(payload: dict[str, Any], key: str, base_dir: Path) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field {key!r} must be a non-empty path string.")
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Config field {name!r} must be a non-negative integer.")
    return value


def _validate(config: PreferencePairConfig) -> None:
    source_names = [source.name for source in config.sources]
    if len(set(source_names)) != len(source_names):
        raise ValueError("Preference-pair source names must be unique.")
    configured_datasets = [
        dataset for source in config.sources for dataset in source.datasets
    ]
    if len(set(configured_datasets)) != len(configured_datasets):
        raise ValueError("A dataset may be configured under only one source.")
    unknown_expected = set(config.expected_counts()) - set(configured_datasets)
    if unknown_expected:
        raise ValueError(
            "Expected record counts contain unconfigured datasets: "
            f"{sorted(unknown_expected)}"
        )
