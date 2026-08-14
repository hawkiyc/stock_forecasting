"""CLI for preflight validation."""

from __future__ import annotations

import argparse

from stock_forecasting.config import ExperimentConfig
from stock_forecasting.preflight import run_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    report = run_preflight(config, require_data=not args.skip_data)
    for key, value in report.facts.items():
        print(f"FACT {key}={value}")
    for warning in report.warnings:
        print(f"WARNING {warning}")
    for error in report.errors:
        print(f"ERROR {error}")
    report.require_success()
    print("Preflight checks passed.")


if __name__ == "__main__":
    main()
