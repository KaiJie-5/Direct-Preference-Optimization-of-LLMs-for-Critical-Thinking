from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .builder import build_preference_pairs
from .config import load_preference_pair_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construct evidence-rich and question-only conversational DPO "
            "preference pairs from reflective-question enrichment traces."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_preference_pair_config(args.config)
        run_dir = build_preference_pairs(config)
    except Exception as exc:
        print(f"dpo-build-preferences: error: {exc}", file=sys.stderr)
        return 1
    print(f"Preference-pair manifest: {run_dir / 'run_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
