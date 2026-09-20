"""Dependency-free baseline identity shared by the local control host and GPU Pod."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

BASELINE_SOURCES = (
    "src/stock_forecasting/baseline_build.py",
    "src/stock_forecasting/baseline_storage.py",
    "src/stock_forecasting/baselines.py",
    "src/stock_forecasting/evaluation_store.py",
    "src/stock_forecasting/metrics.py",
    "src/stock_forecasting/optimization_policy.py",
    "src/stock_forecasting/data/dataset.py",
    "src/stock_forecasting/data/adjustments.py",
    "src/stock_forecasting/data/horizons.py",
)
SHARED_TRAINING_DEFINITIONS = (
    "ResumableFixedSizeBatchSampler",
    "_RuntimeLabelDataset",
    "estimate_runtime_robust_scales",
    "_robust_scale_identity",
    "_load_cached_robust_scales",
    "resolve_runtime_robust_scales",
)


def shared_training_identity(project: Path) -> str:
    tree = ast.parse((project / "src/stock_forecasting/training.py").read_text())
    selected = [
        node
        for node in tree.body
        if getattr(node, "name", None) in SHARED_TRAINING_DEFINITIONS
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id.startswith("ROBUST_SCALE_")
                for target in node.targets
            )
        )
    ]
    names = {getattr(node, "name", None) for node in selected}
    if not set(SHARED_TRAINING_DEFINITIONS) <= names:
        raise ValueError("Baseline shared calibration/sampling definitions are missing")
    return hashlib.sha256(ast.dump(ast.Module(body=selected, type_ignores=[])).encode()).hexdigest()


def digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def baseline_contract(project: Path, selection: dict) -> dict:
    parameters = json.loads((project / "configs/baseline.json").read_text())
    # Scheduling/resource limits cannot change numerical identity or force a rebuild.
    parameters.pop("resources", None)
    request = selection["dataset_request"]
    core = {
        "schema_version": 1,
        "data": request,
        "parameters": parameters,
        "implementation": {
            name: hashlib.sha256((project / name).read_bytes()).hexdigest()
            for name in BASELINE_SOURCES
        },
        "shared_calibration_sampling": shared_training_identity(project),
        "evaluation": "full-canonical-stock-date-v2",
    }
    return {"baseline_id": "baseline-" + digest(core), "contract": core}


def runtime_contract() -> tuple[Path, dict]:
    project = Path(__file__).resolve().parents[2]
    selection_path = os.environ.get("RUNPOD_SELECTION_FILE") or os.environ.get(
        "RUNPOD_REMOTE_SELECTION_PATH"
    )
    if not selection_path:
        raise ValueError("Baseline workflow requires the immutable RunPod selection")
    selection = json.loads(Path(selection_path).read_text())
    return project, baseline_contract(project, selection)


def validate_complete(payload: dict, expected: dict) -> None:
    if payload.get("state") != "complete" or payload.get("identity") != expected:
        raise ValueError("No complete baseline matches the active data/training contract")
    models = expected["contract"]["parameters"]["models"]
    if set(payload.get("models", {})) != set(models):
        raise ValueError("Baseline completion does not contain every required model")
    if any(v.get("state") != "complete" for v in payload["models"].values()):
        raise ValueError("A baseline model is incomplete")
    if not payload.get("artifacts") or not payload.get("evaluation_membership"):
        raise ValueError("Baseline completion is missing reusable artifacts or full membership")
    for name in models:
        learned = name in {"gbdt", "gru", "dlinear", "patchtst"}
        for seed in expected["contract"]["parameters"]["seeds"] if learned else [None]:
            directory = f"jobs/{name}-{seed}" if learned else f"jobs/rules/{name}"
            weight = "model.pkl" if name == "gbdt" else "model.pt" if learned else "model.json"
            for relative in (
                f"{directory}/{weight}",
                f"{directory}/validation-metrics.json",
                f"{directory}/test/predictions.npy",
                f"{directory}/test/targets.npy",
                f"{directory}/test/membership.npy",
            ):
                if relative not in payload["artifacts"]:
                    raise ValueError(
                        f"Baseline completion lacks required reusable artifact: {relative}"
                    )
    counts = payload.get("sample_counts", {})
    if set(counts) != {"train", "validation", "test"} or any(
        type(v) is not int or v < 1 for v in counts.values()
    ):
        raise ValueError("Baseline completion must record all three full population counts")
    if payload.get("validation_membership", {}).get("samples") != counts["validation"]:
        raise ValueError("Baseline completion is missing the full validation membership")
    for name, result in payload["models"].items():
        scores = result.get("seed_results") or {"single": result.get("metrics", {})}
        if name in {"gbdt", "gru", "dlinear", "patchtst"} and set(scores) != set(
            map(str, expected["contract"]["parameters"]["seeds"])
        ):
            raise ValueError("Baseline completion is missing required random seeds")
        for metrics in scores.values():
            if (
                metrics.get("sample_membership") != payload["evaluation_membership"]
                or metrics.get("samples") != counts["test"]
            ):
                raise ValueError("Baseline metrics do not match the complete test population")
            if metrics.get("evaluation_robust_scales") != payload.get("robust_scales"):
                raise ValueError("Baseline metric calibration differs between experiments")


def validate_optimization_alignment(config, parameters):
    for name in (
        "evaluations_per_epoch",
        "early_stopping_patience_evaluations",
        "early_stopping_min_delta",
        "early_stopping_start_epoch",
        "plateau_patience_evaluations",
        "plateau_factor",
        "plateau_min_ratio",
        "plateau_min_low_lr_evaluations",
    ):
        if getattr(config.training, name) != parameters[name]:
            raise ValueError(f"Main model and baseline optimization policy differ: {name}")
    if config.training.learning_rate_schedule != "validation_plateau":
        raise ValueError("The full baseline workflow requires the validation-plateau schedule")
    if (
        config.training.evaluation_max_samples is not None
        or config.validation.baseline_max_samples_per_split is not None
    ):
        raise ValueError("Prebuilt baselines require full validation and full test")
    if (
        config.data.calibration_seed != parameters["calibration_seed"]
        or config.data.label_scale_calibration_samples
        != parameters["label_scale_calibration_samples"]
    ):
        raise ValueError("Main model and baseline train-only loss calibration must align")
    if (
        set(config.validation.models) - {"kronos_full"} != set(parameters["models"])
        or config.validation.seeds != parameters["seeds"]
    ):
        raise ValueError("Main model and baseline model/seed manifests must align")


def validate_local_configuration(project: Path, selection: dict, parameters: dict) -> None:
    """Read only the repository's small YAML scalar/list contract, without ML imports."""
    sections, section, active_list = {}, None, None
    for line in (project / selection["stage"]["config_path"]).read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        indentation = len(line) - len(line.lstrip())
        if indentation == 0:
            section = text.rstrip(":")
            sections[section] = {}
        elif indentation == 2 and section is not None and ":" in text:
            key, value = text.split(":", 1)
            value = value.strip()
            active_list = key if not value else None
            try:
                parsed = json.loads(value) if value else []
            except ValueError:
                parsed = value.strip("\"'")
            sections[section][key] = parsed
        elif indentation == 4 and text.startswith("- ") and active_list is not None:
            sections[section][active_list].append(text[2:].strip("\"'"))
    config = SimpleNamespace(
        **{name: SimpleNamespace(**values) for name, values in sections.items()}
    )
    validate_optimization_alignment(config, parameters)
    if sections["model"].get("explicit_output_scale") and selection["stage"][
        "feature_mode"
    ] not in ("scales", "combined"):
        raise ValueError("Production output-scale control requires feature mode scales or combined")


def require_baselines(config, *, verify_artifacts: bool = True) -> dict:
    _project, identity = runtime_contract()
    parameters = identity["contract"]["parameters"]
    validate_optimization_alignment(config, parameters)
    root = (
        Path(
            os.environ.get(
                "NETWORK_VOLUME_ROOT", os.environ.get("RUNPOD_VOLUME_MOUNT_PATH", "/runpod-volume")
            )
        )
        / "baselines"
        / identity["baseline_id"]
    )
    path = root / "complete.json"
    if not path.is_file():
        raise ValueError("Build matching baselines first: bash scripts/runpod_workflow.sh baseline")
    payload = json.loads(path.read_text())
    validate_complete(payload, identity)
    from stock_forecasting.data.manifest import sha256_file
    from stock_forecasting.training_paths import resolve_bar_store_path

    manifest = resolve_bar_store_path(config.data.bar_store_path) / "bar-store.json"
    if payload.get("data_identity", {}).get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("Prepared data differs from the baseline's immutable bar store")
    if payload["sample_counts"] != json.loads(manifest.read_text())["split_counts"]:
        raise ValueError("Baseline population counts differ from the complete prepared splits")
    if verify_artifacts:
        for relative, metadata in payload["artifacts"].items():
            artifact = (root / relative).resolve()
            if (
                not artifact.is_relative_to(root.resolve())
                or not artifact.is_file()
                or artifact.stat().st_size != metadata["bytes"]
            ):
                raise ValueError(f"Missing or truncated baseline artifact: {relative}")
    return payload
