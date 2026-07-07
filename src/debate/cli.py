from __future__ import annotations

import argparse
from pathlib import Path

from .compare import compare_rankings
from .config import load_debate_config
from .preflight import run_preflight
from .ranking import run_debate_ranking


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multi-agent debate ranking and compare it with reviewer CSVs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rank = subparsers.add_parser("rank", help="Run multi-agent debate ranking.")
    rank.add_argument("--config", required=True, type=Path)

    preflight = subparsers.add_parser(
        "preflight",
        help="Load configured debate agents and report device placement.",
    )
    preflight.add_argument("--config", required=True, type=Path)
    preflight.add_argument(
        "--generate-qwen-json",
        action="store_true",
        help="After loading both agents, ask qwen_72b for a tiny strict JSON response.",
    )

    compare = subparsers.add_parser(
        "compare",
        help="Compare saved debate rankings with qualitative researcher CSV exports.",
    )
    compare.add_argument(
        "--config",
        type=Path,
        help="Optional debate config path for command provenance.",
    )
    compare.add_argument("--model-csv", required=True, type=Path)
    compare.add_argument(
        "--reviewer-csv",
        required=True,
        action="append",
        type=Path,
        help="Reviewer export CSV. May be repeated.",
    )
    compare.add_argument("--output-dir", required=True, type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "rank":
        config = load_debate_config(args.config)
        run_debate_ranking(config)
        return 0
    if args.command == "preflight":
        config = load_debate_config(args.config)
        run_preflight(config, generate_qwen_json=args.generate_qwen_json)
        return 0
    if args.command == "compare":
        compare_rankings(
            model_csv=args.model_csv,
            reviewer_csvs=args.reviewer_csv,
            output_dir=args.output_dir,
        )
        return 0
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
