from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType


def _load_verifier() -> ModuleType:
    script = Path(__file__).parents[1] / "scripts" / "verify_poetry_runtime_ownership.py"
    spec = importlib.util.spec_from_file_location("verify_poetry_runtime_ownership", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_lock_filter() -> ModuleType:
    script = Path(__file__).parents[1] / "scripts" / "filter_runpod_poetry_lock.py"
    spec = importlib.util.spec_from_file_location("filter_runpod_poetry_lock", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_keeps_runpod_runtime_out_of_poetry() -> None:
    verifier = _load_verifier()
    project_root = Path(__file__).parents[1]

    assert (
        verifier.find_violations(
            project_root,
            allow_transitive_image_packages=True,
        )
        == []
    )


def test_locked_torch_is_rejected(tmp_path: Path) -> None:
    verifier = _load_verifier()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "test"\nversion = "0.1.0"\ndependencies = []\n',
        encoding="utf-8",
    )
    (tmp_path / "poetry.lock").write_text(
        '[[package]]\nname = "torch"\nversion = "2.9.1+cu128"\n',
        encoding="utf-8",
    )

    violations = verifier.find_violations(tmp_path)

    assert any("unapproved RunPod image packages: torch" in item for item in violations)


def test_transitive_locked_torch_can_be_allowed_without_allowing_direct_torch(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "test"\nversion = "0.1.0"\ndependencies = ["accelerate>=1.1,<2.0"]\n',
        encoding="utf-8",
    )
    (tmp_path / "poetry.lock").write_text(
        '[[package]]\nname = "accelerate"\nversion = "1.14.0"\n\n'
        '[package.dependencies]\n'
        'torch = ">=2.0.0"\n\n'
        '[[package]]\nname = "torch"\nversion = "2.9.1"\n',
        encoding="utf-8",
    )

    assert verifier.find_violations(tmp_path, allow_transitive_image_packages=True) == []

    (tmp_path / "poetry.lock").write_text(
        '[[package]]\nname = "torch"\nversion = "2.9.1"\n',
        encoding="utf-8",
    )
    violations = verifier.find_violations(tmp_path, allow_transitive_image_packages=True)

    assert any("unapproved RunPod image packages: torch" in item for item in violations)

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "test"\nversion = "0.1.0"\ndependencies = ["torch>=2.0"]\n',
        encoding="utf-8",
    )
    violations = verifier.find_violations(tmp_path, allow_transitive_image_packages=True)

    assert any("project dependency must come from the image: torch" in item for item in violations)


def test_poetry_files_remain_valid_toml() -> None:
    project_root = Path(__file__).parents[1]
    with (project_root / "pyproject.toml").open("rb") as stream:
        assert tomllib.load(stream)["project"]["name"] == "fin-ts-multimodal"


def test_runpod_lint_version_is_pinned() -> None:
    project_root = Path(__file__).parents[1]
    with (project_root / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)

    assert pyproject["tool"]["poetry"]["group"]["dev"]["dependencies"]["ruff"] == "0.15.21"


def test_lock_filter_preserves_authoritative_metadata_and_removes_image_packages() -> None:
    lock_filter = _load_lock_filter()
    lock_text = (
        "# generated\n\n"
        '[[package]]\nname = "accelerate"\nversion = "1.14.0"\n\n'
        "[package.dependencies]\n"
        'torch = ">=2.0.0"\n\n'
        '[[package]]\nname = "torch"\nversion = "2.13.0"\n\n'
        '[[package]]\nname = "numpy"\nversion = "2.0.0"\n\n'
        "[metadata]\nlock-version = \"2.1\"\n"
    )

    filtered_text, removed = lock_filter.filter_lock_text(lock_text)

    assert removed == ["torch"]
    assert 'name = "torch"' not in filtered_text
    assert 'name = "accelerate"' in filtered_text
    assert 'name = "numpy"' in filtered_text
    assert '[metadata]\nlock-version = "2.1"' in filtered_text
    assert tomllib.loads(filtered_text)["metadata"]["lock-version"] == "2.1"
