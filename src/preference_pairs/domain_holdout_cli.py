from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .domain_holdout import build_domain_holdout, load_domain_holdout_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Partition an existing validated DPO preference-pair run into "
            "explicit train and unseen-domain test files."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_domain_holdout_config(args.config)
        run_dir = build_domain_holdout(config)
    except Exception as exc:
        print(f"dpo-build-domain-holdout: error: {exc}", file=sys.stderr)
        return 1
    print(f"Domain-holdout manifest: {run_dir / 'run_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
