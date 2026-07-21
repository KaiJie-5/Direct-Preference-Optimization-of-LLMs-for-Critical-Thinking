from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_reflective_config
from .runner import run_reflective_enrichment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reflective questions from debate-ranked or direct single-pass "
            "code examples."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIR",
        help="Resume in an existing reflective-enrichment run directory.",
    )
    parser.add_argument(
        "--migrate-label-normalization",
        action="store_true",
        help=(
            "Back up and narrowly migrate a v1 resume run to normalized code labels "
            "before retrying unresolved records. Requires --resume."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_reflective_config(args.config)
    if args.migrate_label_normalization and args.resume is None:
        raise SystemExit("--migrate-label-normalization requires --resume")
    if args.migrate_label_normalization and config.input_mode != "ranked":
        raise SystemExit(
            "--migrate-label-normalization applies only to historical ranked-input runs"
        )
    run_dir = run_reflective_enrichment(
        config,
        resume_dir=args.resume,
        migrate_label_normalization=args.migrate_label_normalization,
    )
    manifest = run_dir / "run_manifest.json"
    import json

    state = json.loads(manifest.read_text(encoding="utf-8"))["run_state"]
    return 0 if state.get("failure_count", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
