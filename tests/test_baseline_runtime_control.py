"""Dependency-free quota and batching contracts runnable on the control host."""

from __future__ import annotations

import ast
import math
import runpy
import tempfile
import unittest
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = runpy.run_path(str(ROOT / "src/stock_forecasting/runtime_resources.py"))


def definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if getattr(node, "name", "") in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class RuntimeTests(unittest.TestCase):
    def test_fractional_container_quota_is_not_host_cpu_count(self):
        select = RESOURCES["select_visible_cpu_count"]
        self.assertEqual(
            select(reported_cpu_count=48, affinity_cpu_count=48, quota_cpu_count=13.6), 13
        )
        self.assertEqual(select(reported_cpu_count=48, quota_cpu_count=0.5), 1)
        with self.assertRaises(ValueError):
            select(reported_cpu_count=48, quota_cpu_count=float("nan"))

    def test_cgroup_v1_v2_ancestors_and_unlimited(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            membership = root / "membership"
            membership.write_text("0::/slice/job\n")
            (root / "slice/job").mkdir(parents=True)
            (root / "cpu.max").write_text("max 100000")
            (root / "slice/cpu.max").write_text("1360000 100000")
            (root / "slice/job/cpu.max").write_text("2000000 100000")
            self.assertEqual(RESOURCES["detect_cpu_quota"](root, membership), 13.6)
            (root / "slice/cpu.max").write_text("max 100000")
            (root / "slice/job/cpu.max").write_text("max 100000")
            self.assertIsNone(RESOURCES["detect_cpu_quota"](root, membership))
            (root / "cpu").mkdir()
            (root / "cpu/cpu.cfs_quota_us").write_text("1360000")
            (root / "cpu/cpu.cfs_period_us").write_text("100000")
            membership.write_text("3:cpu,cpuacct:/\n")
            self.assertEqual(RESOURCES["detect_cpu_quota"](root, membership), 13.6)

    def test_resume_with_another_batch_preserves_dynamic_epoch_order(self):
        from collections.abc import Iterator
        from math import gcd

        namespace = {
            "math": math,
            "gcd": gcd,
            "Iterator": Iterator,
            "Sampler": list,
            "islice": islice,
        }
        definitions(
            "src/stock_forecasting/data/dataset.py", {"BlockwisePermutationSampler"}, namespace
        )
        definitions(
            "src/stock_forecasting/baseline_runtime.py", {"SampleCursorBatchSampler"}, namespace
        )
        sampler_type = namespace["SampleCursorBatchSampler"]
        first = sampler_type(1003, 64, 42, 5)
        batches = list(first)
        original = [index for batch in batches for index in batch]
        cursor = sum(map(len, batches[:7]))
        resumed = sampler_type(1003, 91, 42, 5)
        resumed.set_epoch(0, cursor)
        rest = [index for batch in resumed for index in batch]
        self.assertEqual(original[:cursor] + rest, original)
        self.assertEqual(sorted(original), list(range(1003)))
        first.set_epoch(1)
        self.assertNotEqual([i for batch in first for i in batch], original)
        self.assertEqual(sorted(i for batch in first for i in batch), list(range(1003)))
        self.assertEqual(len(resumed), len(list(resumed)))

    def test_prefetch_budget_accounts_for_both_live_pools(self):
        namespace = definitions(
            "src/stock_forecasting/baseline_runtime.py", {"bounded_prefetch"}, {}
        )
        plan = namespace["bounded_prefetch"]
        self.assertEqual(
            plan(
                workers=2,
                batch_bytes=1024**2,
                host_bytes=1024**3,
                shared_bytes=64 * 1024**2,
                desired=8,
                maximum=4,
            ),
            2,
        )
        with self.assertRaises(MemoryError):
            plan(
                workers=2,
                batch_bytes=1024**2,
                host_bytes=1024**3,
                shared_bytes=1024,
                desired=2,
                maximum=4,
            )


if __name__ == "__main__":
    unittest.main()
