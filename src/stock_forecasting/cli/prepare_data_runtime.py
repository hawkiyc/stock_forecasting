"""Stable console entry point for Stage 1 data preparation."""

from stock_forecasting.cli.prepare_data import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
