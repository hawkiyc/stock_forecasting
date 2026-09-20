"""Disk-backed full-population evaluation with bounded numerical reductions."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from stock_forecasting.metrics import POSTPROCESS_SIGNAL_NAMES, cross_sectional_metrics

CHUNK_ROWS = 8192
META_DTYPE = np.dtype(
    [
        ("symbol", "U48"),
        ("date", "U40"),
        ("market", "U24"),
        ("asset_type", "U24"),
        ("provider", "U32"),
    ]
)


class Moments:
    """Accumulate sufficient statistics; never average unequal batch means."""

    def __init__(self, horizons: list[int], scales: list[float]):
        self.horizons = horizons
        self.scales = np.asarray(scales, dtype=np.float64)
        if (
            self.scales.shape != (len(horizons),)
            or not np.isfinite(self.scales).all()
            or (self.scales <= 0).any()
        ):
            raise ValueError("Evaluation requires one finite positive scale per horizon")
        self.count = 0
        self.sums = np.zeros((10, len(horizons)), dtype=np.float64)

    def add(self, target: np.ndarray, prediction: np.ndarray) -> None:
        y, q = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
        if y.shape != (len(y), len(self.horizons)) or q.shape != (*y.shape, 3):
            raise ValueError("Metric inputs have incompatible horizon shapes")
        if not np.isfinite(y).all() or not np.isfinite(q).all():
            raise ValueError("Full evaluation contains nonfinite targets or predictions")
        if (q[..., 0] > q[..., 1]).any() or (q[..., 1] > q[..., 2]).any():
            raise ValueError("Quantile ordering is violated")
        p = q[..., 1]
        error = y[..., None] - q
        pinball = np.maximum(error * [0.1, 0.5, 0.9], error * [-0.9, -0.5, -0.1]).mean(-1)
        values = (
            pinball,
            abs(y - p),
            (y >= q[..., 0]) & (y <= q[..., 2]),
            q[..., 2] - q[..., 0],
            np.sign(y) == np.sign(p),
            y,
            p,
            y * y,
            p * p,
            y * p,
        )
        self.sums += np.stack([value.sum(axis=0) for value in values])
        self.count += len(y)

    def result(self) -> dict[str, Any]:
        if not self.count:
            raise ValueError("Empty evaluation population")
        m = self.sums / self.count
        covariance = m[9] - m[5] * m[6]
        denominator = np.sqrt(np.maximum(0, m[7] - m[5] ** 2) * np.maximum(0, m[8] - m[6] ** 2))
        correlations = np.divide(
            covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-20
        ).clip(-1, 1)
        per_horizon = {}
        for i, h in enumerate(self.horizons):
            per_horizon[f"{h}d"] = {
                "pinball": float(m[0, i]),
                "normalized_pinball": float(m[0, i] / self.scales[i]),
                "median_mae": float(m[1, i]),
                "normalized_median_mae": float(m[1, i] / self.scales[i]),
                "interval_coverage": float(m[2, i]),
                "interval_width": float(m[3, i]),
                "coverage_error": float(abs(m[2, i] - 0.8)),
                "median_direction_agreement": float(m[4, i]),
                "median_correlation": float(correlations[i]),
            }
        aggregate = {
            key: float(np.mean([v[key] for v in per_horizon.values()]))
            for key in (
                "pinball",
                "normalized_pinball",
                "median_mae",
                "interval_coverage",
                "coverage_error",
            )
        }
        aggregate["selection_score"] = aggregate["normalized_pinball"]
        return {
            "aggregate": aggregate,
            "per_horizon": per_horizon,
            "primary_5d": {**per_horizon["5d"], "selection_score": aggregate["selection_score"]},
        }


class EvaluationStore:
    """Persist predictions in canonical dataset order, never on the GPU."""

    def __init__(self, root: Path, count: int, horizons: list[int], scales: list[float]):
        self.root, self.count, self.horizons, self.scales = root, count, horizons, scales
        root.mkdir(parents=True, exist_ok=True)
        required = count * (len(horizons) * 16 + META_DTYPE.itemsize)
        if shutil.disk_usage(root).free < required + 1024**3:
            raise OSError(f"Full evaluation requires {required} bytes plus 1 GiB free disk")
        self.targets = np.lib.format.open_memmap(
            root / "targets.npy", mode="w+", dtype="float32", shape=(count, len(horizons))
        )
        self.predictions = np.lib.format.open_memmap(
            root / "predictions.npy", mode="w+", dtype="float32", shape=(count, len(horizons), 3)
        )
        self.metadata = np.lib.format.open_memmap(
            root / "membership.npy", mode="w+", dtype=META_DTYPE, shape=(count,)
        )
        self.offset = 0

    def append(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        *,
        symbols,
        dates,
        markets,
        asset_types,
        providers,
    ) -> None:
        count = len(targets)
        stop = self.offset + count
        if (
            stop > self.count
            or predictions.shape != (count, len(self.horizons), 3)
            or targets.shape != (count, len(self.horizons))
        ):
            raise ValueError("Evaluation batch does not match the full population contract")
        self.targets[self.offset : stop] = targets
        self.predictions[self.offset : stop] = predictions
        for name, values in (
            ("symbol", symbols),
            ("date", dates),
            ("market", markets),
            ("asset_type", asset_types),
            ("provider", providers),
        ):
            strings = [str(value) for value in values]
            if len(strings) != count or any(
                len(v) > META_DTYPE[name].itemsize // 4 for v in strings
            ):
                raise ValueError(f"Invalid or truncated evaluation metadata: {name}")
            self.metadata[name][self.offset : stop] = strings
        self.offset = stop

    def finish(self, *, signal_threshold: float = 0.0) -> dict[str, Any]:
        if self.offset != self.count:
            raise ValueError(f"Incomplete evaluation: {self.offset}/{self.count}")
        for array in (self.targets, self.predictions, self.metadata):
            array.flush()
        total = Moments(self.horizons, self.scales)
        subgroups: dict[str, dict[str, Moments]] = {
            k: {} for k in ("market", "asset_type", "provider", "month", "year")
        }
        daily: dict[str, tuple[float, int]] = {}
        digest = hashlib.sha256(b"[")
        signal_counts = np.zeros((len(self.horizons), 5), dtype=np.int64)
        for start in range(0, self.count, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, self.count)
            y, q, meta = (
                self.targets[start:stop],
                self.predictions[start:stop],
                self.metadata[start:stop],
            )
            total.add(y, q)
            for name, groups in subgroups.items():
                values = (
                    np.asarray([v[: 7 if name == "month" else 4] for v in meta["date"]])
                    if name in ("month", "year")
                    else meta[name]
                )
                for value in np.unique(values):
                    groups.setdefault(str(value), Moments(self.horizons, self.scales)).add(
                        y[values == value], q[values == value]
                    )
            errors = (y[..., None].astype(np.float64) - q) / np.asarray(self.scales)[None, :, None]
            losses = np.maximum(errors * [0.1, 0.5, 0.9], errors * [-0.9, -0.5, -0.1]).mean((1, 2))
            for day in np.unique(meta["date"]):
                selected = meta["date"] == day
                old_sum, old_count = daily.get(str(day), (0.0, 0))
                daily[str(day)] = (
                    old_sum + float(losses[selected].sum()),
                    old_count + int(selected.sum()),
                )
            codes = np.full(q.shape[:2], 2)
            codes[q[..., 1] < -signal_threshold] = 1
            codes[q[..., 2] < -signal_threshold] = 0
            codes[q[..., 1] > signal_threshold] = 3
            codes[q[..., 0] > signal_threshold] = 4
            signal_counts += np.stack([(codes == i).sum(0) for i in range(5)], axis=-1)
            for index, row in enumerate(meta, start):
                if index:
                    digest.update(b",")
                digest.update(
                    json.dumps(
                        [str(row["symbol"]), str(row["date"])],
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode()
                )
        digest.update(b"]")
        # Detect duplicates using only an integer permutation plus one symbol group.
        dates = self.metadata["date"]
        order = np.argsort(dates, kind="stable")
        boundaries = np.flatnonzero(dates[order][1:] != dates[order][:-1]) + 1
        for group in np.split(order, boundaries):
            symbols = self.metadata["symbol"][group]
            if len(np.unique(symbols)) != len(symbols):
                raise ValueError("Duplicate stock-date in full evaluation")
        cross = {
            f"{h}d": cross_sectional_metrics(
                targets=self.targets[:, i],
                signals=self.predictions[:, i, 1],
                dates=dates,
                symbols=self.metadata["symbol"],
                annualization_horizon=h,
                date_groups=(order, boundaries),
            )
            for i, h in enumerate(self.horizons)
        }
        del order, boundaries
        result = total.result()
        result.update(
            {
                "loss": result["aggregate"]["normalized_pinball"],
                "samples": self.count,
                "sample_membership": {
                    "ordered_symbol_dates_sha256": digest.hexdigest(),
                    "samples": self.count,
                    "unique_dates": len(daily),
                    "cutoff_start": min(daily),
                    "cutoff_end": max(daily),
                },
                "evaluation_robust_scales": self.scales,
                "daily_normalized_pinball": {k: s / n for k, (s, n) in sorted(daily.items())},
                "cross_sectional_by_horizon": cross,
                "cross_sectional_5d": cross["5d"],
                "subgroups": {
                    name: {
                        key: {"samples": value.count, **value.result()}
                        for key, value in sorted(groups.items())
                    }
                    for name, groups in subgroups.items()
                },
                "postprocess_signal_distribution": {
                    f"{h}d": dict(
                        zip(POSTPROCESS_SIGNAL_NAMES, map(int, signal_counts[i]), strict=True)
                    )
                    for i, h in enumerate(self.horizons)
                },
            }
        )
        return result

    def close(self) -> None:
        for array in (self.targets, self.predictions, self.metadata):
            array.flush()
            array._mmap.close()
