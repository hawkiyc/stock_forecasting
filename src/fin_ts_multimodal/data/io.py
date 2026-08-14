"""Serialization for processed causal-window records."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

import pandas as pd

_REQUIRED_NESTED_COLUMNS = (
    "context",
    "benchmark_context",
    "label",
    "diagnostics",
    "metadata",
)
_OPTIONAL_NESTED_DEFAULTS: dict[str, Any] = {}
_NESTED_COLUMNS = (*_REQUIRED_NESTED_COLUMNS, *_OPTIONAL_NESTED_DEFAULTS)


def _missing_optional_value(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _optional_default(column: str) -> list[Any] | dict[str, Any]:
    return [] if isinstance(_OPTIONAL_NESTED_DEFAULTS[column], list) else {}


def write_processed_records(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Write nested records to JSONL or portable string-backed Parquet."""

    if not records:
        raise ValueError("No processed records to write.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with destination.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
                output.write("\n")
    elif suffix in {".parquet", ".pq"}:
        rows: list[dict[str, Any]] = []
        for record in records:
            row = dict(record)
            for column in _NESTED_COLUMNS:
                if column not in row:
                    if column in _OPTIONAL_NESTED_DEFAULTS:
                        row[column] = _optional_default(column)
                    else:
                        continue
                elif column in _OPTIONAL_NESTED_DEFAULTS and _missing_optional_value(row[column]):
                    row[column] = _optional_default(column)
                row[column] = json.dumps(
                    row[column],
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            rows.append(row)
        try:
            pd.DataFrame(rows).to_parquet(destination, index=False)
        except ImportError as error:
            raise RuntimeError(
                "Writing Parquet requires pyarrow. Install the parquet dependency group."
            ) from error
    else:
        raise ValueError(
            f"Unsupported processed format: {destination.suffix}. Use JSONL or Parquet."
        )
    return destination


def write_processed_records_streaming(
    records: Iterable[dict[str, Any]],
    path: str | Path,
    *,
    batch_size: int = 128,
) -> Path:
    """Write records with bounded memory using a JSON-envelope Parquet column."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("Streaming processed output must use Parquet")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Writing Parquet requires pyarrow. Install the parquet dependency group."
        ) from error

    writer: Any = None
    buffered: list[str] = []
    written = 0
    try:
        for record in records:
            buffered.append(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            )
            if len(buffered) < batch_size:
                continue
            table = pa.table({"record_json": buffered})
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
            written += len(buffered)
            buffered = []
        if buffered:
            table = pa.table({"record_json": buffered})
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
            written += len(buffered)
    finally:
        if writer is not None:
            writer.close()
    if written == 0:
        raise ValueError("No processed records to write.")
    return destination


def _decode_nested_row(row: dict[str, Any]) -> dict[str, Any]:
    for column in _NESTED_COLUMNS:
        if column in row and isinstance(row[column], str):
            row[column] = json.loads(row[column])
        elif column in _OPTIONAL_NESTED_DEFAULTS and (
            column not in row or _missing_optional_value(row[column])
        ):
            row[column] = _optional_default(column)
    return row


def iter_processed_records(
    path: str | Path,
    *,
    batch_size: int = 64,
) -> Iterator[dict[str, Any]]:
    """Yield processed records in bounded batches from legacy or envelope files."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    source = Path(path)
    suffix = source.suffix.lower()
    yielded = 0
    if suffix in {".jsonl", ".ndjson"}:
        with source.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                if not line.strip():
                    continue
                yielded += 1
                yield cast(dict[str, Any], json.loads(line))
    elif suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError(
                "Reading Parquet requires pyarrow. Install the parquet dependency group."
            ) from error
        parquet_file = pq.ParquetFile(source)
        envelope = parquet_file.schema_arrow.names == ["record_json"]
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            rows = cast(list[dict[str, Any]], batch.to_pylist())
            for row in rows:
                yielded += 1
                if envelope:
                    yield cast(dict[str, Any], json.loads(str(row["record_json"])))
                else:
                    yield _decode_nested_row(row)
    else:
        raise ValueError(f"Unsupported processed format: {source.suffix}. Use JSONL or Parquet.")
    if yielded == 0:
        raise ValueError(f"Processed dataset is empty: {source}")


def read_processed_records(path: str | Path) -> list[dict[str, Any]]:
    """Load records previously written by :func:`write_processed_records`."""

    return list(iter_processed_records(path))
