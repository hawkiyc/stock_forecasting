"""Dependency-free regression checks; safe on the local control host."""

from __future__ import annotations

import ast
import copy
import json
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = runpy.run_path(str(ROOT / "src/stock_forecasting/baseline_contract.py"))
POLICY = runpy.run_path(str(ROOT / "src/stock_forecasting/optimization_policy.py"))
READINESS = runpy.run_path(str(ROOT / "scripts/runpod_readiness.py"))
SELECTION = runpy.run_path(str(ROOT / "scripts/runpod_selection.py"))


class FullWorkflowControlPlaneTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for relative in (
            *CONTRACT["BASELINE_SOURCES"],
            "configs/baseline.json",
            "src/stock_forecasting/training.py",
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / relative).read_bytes())
        self.selection = {
            "dataset_request": {
                "date_range": {"start_inclusive": "2016-01-01", "end_exclusive": "2026-06-01"},
                "profile": "us_tw_eodhd",
                "preparation": {"h_start": 1},
            }
        }

    def tearDown(self):
        self.temporary.cleanup()

    def identity(self):
        return CONTRACT["baseline_contract"](self.root, self.selection)

    def test_unrelated_code_never_invalidates_baseline(self):
        before = self.identity()
        path = self.root / "src/stock_forecasting/models/forecast.py"
        path.parent.mkdir(parents=True)
        path.write_text("# A different main model\n")
        (self.root / "README.md").write_text("New documentation")
        self.assertEqual(before, self.identity())

    def test_only_baseline_relevant_shared_training_code_invalidates_cache(self):
        before = self.identity()
        path = self.root / "src/stock_forecasting/training.py"
        content = path.read_text()
        path.write_text(content + "\ndef main_model_only_change():\n    return 2\n")
        self.assertEqual(before, self.identity())
        path.write_text(
            content.replace(
                "ROBUST_SCALE_SELECTION_BLOCK_SIZE = 16", "ROBUST_SCALE_SELECTION_BLOCK_SIZE = 32"
            )
        )
        self.assertNotEqual(before, self.identity())

    def test_data_period_and_baseline_code_invalidate(self):
        before = self.identity()
        self.selection["dataset_request"]["date_range"]["start_inclusive"] = "2021-01-01"
        self.assertNotEqual(before, self.identity())
        before = self.identity()
        path = self.root / "src/stock_forecasting/baseline_build.py"
        path.write_text(path.read_text() + "\n# changed baseline implementation\n")
        self.assertNotEqual(before, self.identity())

    def test_resource_limits_do_not_retrain_models(self):
        before = self.identity()
        path = self.root / "configs/baseline.json"
        parameters = json.loads(path.read_text())
        parameters["resources"]["max_gpu_experiments"] = 3
        path.write_text(json.dumps(parameters))
        self.assertEqual(before, self.identity())
        parameters["learning_rate"] *= 2
        path.write_text(json.dumps(parameters))
        self.assertNotEqual(before, self.identity())

    def test_incomplete_cache_fails_closed(self):
        with self.assertRaises(ValueError):
            CONTRACT["validate_complete"](
                {"state": "complete", "identity": self.identity()}, self.identity()
            )

    def test_cpu_admission_reserves_control_and_input_workers(self):
        # Execute the actual resource planner with device probes stubbed, without
        # importing any local ML package or inspecting a training environment.
        tree = ast.parse((ROOT / "src/stock_forecasting/baseline_build.py").read_text())
        definition = next(
            node for node in tree.body if getattr(node, "name", None) == "resource_plan"
        )
        code = compile(ast.Module(body=[definition], type_ignores=[]), "resource-plan", "exec")
        parameters = json.loads((ROOT / "configs/baseline.json").read_text())
        for cpus in (7, 8, 12, 16, 32):
            with self.subTest(cpus=cpus):
                namespace = {
                    "detect_visible_cpu_count": lambda cpus=cpus: cpus,
                    "detect_available_memory": lambda: SimpleNamespace(
                        available_bytes=64 * 1024**3
                    ),
                    "torch": SimpleNamespace(
                        cuda=SimpleNamespace(mem_get_info=lambda: (24 * 1024**3, 24 * 1024**3))
                    ),
                    "shutil": SimpleNamespace(
                        disk_usage=lambda path: SimpleNamespace(free=16 * 1024**3)
                    ),
                }
                exec(code, namespace)
                plan = namespace["resource_plan"](parameters, 1000, list(range(1, 15)))
                gpu_cores = plan["gpu_slots"] * (2 * plan["loader_workers"] + 1)
                self.assertLessEqual(gpu_cores + 2 + plan["cpu_slots"] * plan["cpu_threads"], cpus)
                self.assertLessEqual(gpu_cores + 2 + plan["input_workers"] + 1, cpus)
                self.assertGreaterEqual(plan["input_workers"], 1)

    def test_local_parameter_gate_reads_both_stage_configs_without_ml_imports(self):
        parameters = json.loads((ROOT / "configs/baseline.json").read_text())
        for stage in ("stage1", "stage2"):
            selection = {
                "stage": {
                    "config_path": f"configs/{stage}_kronos_base_lora.yaml",
                    "feature_mode": "combined",
                }
            }
            CONTRACT["validate_local_configuration"](ROOT, selection, parameters)
            selection["stage"]["feature_mode"] = "baseline"
            with self.assertRaisesRegex(ValueError, "feature mode"):
                CONTRACT["validate_local_configuration"](ROOT, selection, parameters)

    def test_scheduler_trains_at_low_lr_before_stop(self):
        class Optimizer:
            def __init__(self):
                self.param_groups = [{"lr": 1e-4}, {"lr": 1e-5}]

        optimizer = Optimizer()
        schedule = POLICY["ValidationPlateauScheduler"](optimizer, 0)
        schedule.observe(1.0)
        for _ in range(4):
            schedule.step()
            schedule.observe(1.0)
        self.assertAlmostEqual(schedule.ratio, 0.09)
        self.assertFalse(schedule.permits_early_stop)
        for _ in range(2):
            schedule.step()
            schedule.observe(1.0)
        self.assertTrue(schedule.permits_early_stop)
        restored = POLICY["ValidationPlateauScheduler"](Optimizer(), 0)
        restored.load_state_dict(copy.deepcopy(schedule.state_dict()))
        restored.realign(400, 12)
        self.assertAlmostEqual(restored.get_last_lr()[0], 9e-6)
        self.assertAlmostEqual(restored.get_last_lr()[1], 9e-7)

    def test_local_guard_requires_exact_baseline_owner_and_completion(self):
        marker = {
            "schema_version": 1,
            "kind": "stage1-baseline",
            "state": "ready",
            "pod_id": "owned-pod",
            "wandb_run_id": "baseline-run-test",
            "baseline_completed": True,
        }
        kwargs = {
            "expected_kind": "stage1-baseline",
            "expected_pod_id": "owned-pod",
            "active_run_id": "baseline-run-test",
        }
        self.assertEqual(READINESS["_guard_lifecycle_state"](marker, **kwargs), "ready")
        for key, value in (
            ("pod_id", "other-pod"),
            ("wandb_run_id", "other-run"),
            ("baseline_completed", False),
        ):
            with self.assertRaises(ValueError):
                READINESS["_guard_lifecycle_state"]({**marker, key: value}, **kwargs)

    def test_preparation_semantics_unchanged_since_architecture_snapshot(self):
        if subprocess.run(
            ["git", "rev-parse", "--verify", "v0.1.0"], cwd=ROOT, capture_output=True
        ).returncode:
            self.skipTest("Source-only RunPod deployments intentionally omit Git metadata")
        content = runpy.run_path(str(ROOT / "src/stock_forecasting/data/content_identity.py"))
        paths = ["src/stock_forecasting/" + p for p in content["semantic_source_paths"]()]
        paths += [
            "src/stock_forecasting/dataset_identity.py",
            "scripts/runpod_cpu_prepare.sh",
            "scripts/runpod_cpu_finalize.sh",
        ]
        changed = subprocess.check_output(
            ["git", "diff", "v0.1.0", "--name-only", "--", *paths], cwd=ROOT, text=True
        )
        self.assertEqual(changed, "")

    def test_configuration_refresh_preserves_dataset_identity(self):
        parser = SELECTION["build_parser"]()
        previous_args = parser.parse_args(
            [
                "create",
                "--project-root",
                str(ROOT),
                "--stage",
                "stage2",
                "--data-profile",
                "us_tw_eodhd",
                "--start",
                "2016-01-01",
                "--end",
                "2026-06-01",
                "--universe",
                "all",
            ]
        )
        previous = SELECTION["_build_selection"](previous_args, ROOT)
        args = parser.parse_args(["create", "--project-root", str(ROOT), "--reuse-current"])
        captured = {}

        def activate(root, payload):
            captured.update(payload)
            return self.root / "selection.json"

        function = SELECTION["command_create"]
        with patch.dict(
            function.__globals__,
            {
                "_resolve_selection_path": lambda *a, **k: (self.root, previous),
                "_activate_selection": activate,
            },
        ):
            self.assertEqual(function(args), 0)
        self.assertEqual(captured["dataset_request_sha256"], previous["dataset_request_sha256"])
        self.assertEqual(captured["dataset_request"], previous["dataset_request"])


if __name__ == "__main__":
    unittest.main()
