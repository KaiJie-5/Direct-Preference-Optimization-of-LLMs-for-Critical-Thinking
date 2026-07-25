from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DATASET_VERSIONS, load_training_config
from .runner import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate, split, profile, and train conversational preference pairs "
            "with TRL DPOTrainer."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--dataset-version",
        required=True,
        choices=DATASET_VERSIONS,
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate, split, render, and tokenize without loading model weights.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Explicit existing run directory whose latest checkpoint should resume.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_training_config(args.config)
        run_dir = run_training(
            config,
            dataset_version=args.dataset_version,
            preflight_only=args.preflight_only,
            resume=args.resume,
        )
    except Exception as exc:
        print(f"dpo-train: error: {exc}", file=sys.stderr)
        return 1
    print(f"DPO training run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
