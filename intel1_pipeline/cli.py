from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import RunOptions, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Intel1 F1 intelligence pipeline.")
    parser.add_argument("--output-dir", default="output/live", type=Path)
    parser.add_argument("--source-registry", default="config/source_registry.json", type=Path)
    parser.add_argument("--force-weekend-id")
    parser.add_argument("--force-stage")
    parser.add_argument("--scheduled", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-ai", action="store_true")
    parser.add_argument("--skip-drive-upload", action="store_true")
    parser.add_argument("--max-items-per-source", type=int, default=16)
    parser.add_argument("--public-base-url", help="Public HTTPS base URL where generated JSON files will be hosted.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        RunOptions(
            output_dir=args.output_dir,
            source_registry=args.source_registry,
            force_weekend_id=args.force_weekend_id,
            force_stage=args.force_stage,
            scheduled=args.scheduled,
            dry_run=args.dry_run,
            skip_ai=args.skip_ai,
            skip_drive_upload=args.skip_drive_upload,
            max_items_per_source=args.max_items_per_source,
            public_base_url=args.public_base_url,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
