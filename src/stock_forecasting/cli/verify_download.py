"""Validate a durable raw-data acquisition checkpoint before preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_forecasting.data.manifest import validate_download_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = validate_download_manifest(args.manifest, input_path=args.raw)
    print(
        json.dumps(
            {
                "state": payload["state"],
                "dataset_profile": payload["dataset_profile"],
                "raw": payload["artifacts"]["raw"],
                "request_log": payload["artifacts"]["request_log"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
