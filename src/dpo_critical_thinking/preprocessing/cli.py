from __future__ import annotations

import argparse
import json
from pathlib import Path

from .codebook import convert_xlsx_codebook
from .html import preprocess_html_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess raw qualitative data for segment-level enrichment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    codebook = subparsers.add_parser("codebook", help="Convert an XLSX codebook to JSON.")
    codebook.add_argument("--input-xlsx", required=True, type=Path)
    codebook.add_argument("--output-path", required=True, type=Path)
    codebook.add_argument("--codebook-id", default="example_codes")
    codebook.add_argument("--codebook-version", required=True)
    codebook.add_argument(
        "--description",
        default="Example codes used as guidance for segment-level qualitative enrichment.",
    )
    codebook.add_argument("--overwrite", action="store_true")

    html = subparsers.add_parser("html", help="Create per-interview HTML and segment JSONL.")
    html.add_argument("--input-path", required=True, type=Path)
    html.add_argument("--raw-html-dir", required=True, type=Path)
    html.add_argument("--segments-dir", required=True, type=Path)
    html.add_argument("--manifest-path", required=True, type=Path)
    html.add_argument("--codebook-path", required=True, type=Path)
    html.add_argument("--interview-id-prefix", default="INT")
    html.add_argument("--heading-selector", default="h2")
    html.add_argument("--interviewer-selector", default="p.interviewer")
    html.add_argument("--participant-selector", default="p.participant")
    html.add_argument("--overwrite", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "codebook":
        payload = convert_xlsx_codebook(
            input_xlsx=args.input_xlsx,
            output_path=args.output_path,
            codebook_id=args.codebook_id,
            codebook_version=args.codebook_version,
            description=args.description,
            overwrite=args.overwrite,
        )
        print(json.dumps({"output_path": str(args.output_path), "code_count": len(payload["codes"])}, indent=2))
        return 0

    if args.command == "html":
        manifest = preprocess_html_dataset(
            input_path=args.input_path,
            raw_html_dir=args.raw_html_dir,
            segments_dir=args.segments_dir,
            manifest_path=args.manifest_path,
            codebook_path=args.codebook_path,
            interview_id_prefix=args.interview_id_prefix,
            heading_selector=args.heading_selector,
            interviewer_selector=args.interviewer_selector,
            participant_selector=args.participant_selector,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "manifest_path": str(args.manifest_path),
                    "interview_count": len(manifest["interviews"]),
                    "segment_count": sum(item["segment_count"] for item in manifest["interviews"]),
                },
                indent=2,
            )
        )
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
