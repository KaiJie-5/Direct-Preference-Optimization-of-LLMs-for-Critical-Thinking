from __future__ import annotations

import argparse
import json
from pathlib import Path

from .codebook import convert_xlsx_codebook
from .exclusions import (
    UKDA_4688_REVIEW_PROFILE,
    approve_exclusions,
    generate_target_review,
)
from .html import preprocess_html_dataset
from .rtf import PROFILE_REGISTRY, preprocess_rtf_dataset


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
    html.add_argument("--interview-id-prefix", default="INT")
    html.add_argument(
        "--interview-id-source",
        choices=["generated", "heading"],
        default="generated",
        help=(
            "Use generated IDs from --interview-id-prefix, or derive IDs from "
            "the interview heading text."
        ),
    )
    html.add_argument("--heading-selector", default="h2")
    html.add_argument("--interviewer-selector", default="p.interviewer")
    html.add_argument("--participant-selector", default="p.participant")
    html.add_argument(
        "--text-normalization",
        choices=["none", "mojibake"],
        default="none",
        help="Optional text cleanup to apply before writing raw HTML and JSONL.",
    )
    html.add_argument("--dataset-id")
    html.add_argument("--domain")
    html.add_argument("--overwrite", action="store_true")

    rtf = subparsers.add_parser(
        "rtf", help="Preprocess an RTF interview archive with a dataset profile."
    )
    rtf.add_argument("--profile", required=True, choices=sorted(PROFILE_REGISTRY))
    rtf.add_argument("--input-path", required=True, type=Path)
    rtf.add_argument("--output-dir", required=True, type=Path)
    rtf.add_argument("--strict-inventory", action="store_true")
    rtf.add_argument("--overwrite", action="store_true")

    target_review = subparsers.add_parser(
        "target-review",
        help="Generate a review queue of possible enrichment exclusions.",
    )
    target_review.add_argument(
        "--profile",
        required=True,
        choices=[UKDA_4688_REVIEW_PROFILE],
    )
    target_review.add_argument("--audit-path", required=True, type=Path)
    target_review.add_argument("--output-path", required=True, type=Path)
    target_review.add_argument("--overwrite", action="store_true")

    approve = subparsers.add_parser(
        "approve-exclusions",
        help="Compile resolved target-review decisions into a runtime exclusion list.",
    )
    approve.add_argument("--review-path", required=True, type=Path)
    approve.add_argument("--audit-path", required=True, type=Path)
    approve.add_argument("--output-path", required=True, type=Path)
    approve.add_argument("--overwrite", action="store_true")

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
            interview_id_prefix=args.interview_id_prefix,
            heading_selector=args.heading_selector,
            interviewer_selector=args.interviewer_selector,
            participant_selector=args.participant_selector,
            interview_id_source=args.interview_id_source,
            text_normalization=args.text_normalization,
            dataset_id=args.dataset_id,
            domain=args.domain,
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

    if args.command == "rtf":
        manifest = preprocess_rtf_dataset(
            profile=args.profile,
            input_path=args.input_path,
            output_dir=args.output_dir,
            strict_inventory=args.strict_inventory,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "profile": manifest["profile"],
                    "output_dir": str(args.output_dir),
                    "interview_count": len(manifest["interviews"]),
                    "segment_count": sum(
                        item["segment_count"] for item in manifest["interviews"]
                    ),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "target-review":
        manifest = generate_target_review(
            profile=args.profile,
            audit_path=args.audit_path,
            output_path=args.output_path,
            overwrite=args.overwrite,
        )
        print(json.dumps(manifest, indent=2))
        return 0

    if args.command == "approve-exclusions":
        manifest = approve_exclusions(
            review_path=args.review_path,
            audit_path=args.audit_path,
            output_path=args.output_path,
            overwrite=args.overwrite,
        )
        print(json.dumps(manifest, indent=2))
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
