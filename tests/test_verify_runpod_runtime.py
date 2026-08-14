from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_verifier() -> ModuleType:
    script = Path(__file__).parents[1] / "scripts" / "verify_runpod_runtime.py"
    spec = importlib.util.spec_from_file_location("verify_runpod_runtime", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata(
    *,
    torch_path: str = "/usr/local/lib/python3.12/site-packages/torch",
) -> dict[str, object]:
    return {
        "python_executable": "/usr/local/bin/python3.12",
        "python_version": "3.12.10",
        "python_prefix": "/usr/local",
        "python_base_prefix": "/usr/local",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "ubuntu_version": "24.04",
        "torch_version": "2.9.1+cu128",
        "torch_public_version": "2.9.1",
        "torch_cuda_version": "12.8",
        "torch_path": torch_path,
        "cuda_available": True,
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "runpod_image": "runpod/pytorch:1.0.7-cu1281-torch291-ubuntu2404",
    }


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "expected_torch": "2.9.1+cu128",
        "expected_cuda_prefix": "12.8",
        "expected_ubuntu": "24.04",
        "network_volume_root": tmp_path / "runpod-volume",
        "role": "image",
        "reference": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_matching_image_runtime_is_accepted(tmp_path: Path) -> None:
    verifier = _load_verifier()

    errors = verifier._validate(_metadata(), args=_args(tmp_path))

    assert errors == []


def test_network_volume_torch_shadow_is_rejected(tmp_path: Path) -> None:
    verifier = _load_verifier()
    network_root = tmp_path / "runpod-volume"
    torch_path = network_root / "project" / ".venv" / "site-packages" / "torch"

    errors = verifier._validate(
        _metadata(torch_path=str(torch_path)),
        args=_args(tmp_path),
    )

    assert any("shadowed by a network-volume package" in error for error in errors)


def test_project_venv_must_match_current_image_runtime(tmp_path: Path) -> None:
    verifier = _load_verifier()
    reference_path = tmp_path / "image-runtime.json"
    reference_path.write_text(json.dumps(_metadata()), encoding="utf-8")
    venv_metadata = _metadata()
    venv_metadata["python_prefix"] = str(tmp_path / "runpod-volume" / "project" / ".venv")
    venv_metadata["torch_version"] = "2.8.0+cu128"

    errors = verifier._validate(
        venv_metadata,
        args=_args(tmp_path, role="venv", reference=reference_path),
    )

    assert any("torch_version does not match" in error for error in errors)
