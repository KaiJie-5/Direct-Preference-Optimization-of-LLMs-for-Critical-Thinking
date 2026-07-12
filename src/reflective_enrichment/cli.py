from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_reflective_config
from .runner import run_reflective_enrichment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reflective questions from debate-ranked code examples."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIR",
        help="Resume in an existing reflective-enrichment run directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_reflective_config(args.config)
    run_dir = run_reflective_enrichment(config, resume_dir=args.resume)
    manifest = run_dir / "run_manifest.json"
    import json

    state = json.loads(manifest.read_text(encoding="utf-8"))["run_state"]
    return 0 if state.get("failure_count", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
