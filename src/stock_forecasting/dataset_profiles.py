"""Dependency-free dataset profile membership contracts."""

from __future__ import annotations

from typing import Final, Literal

DatasetProfile = Literal[
    "tw_only",
    "us_only_eodhd",
    "us_tw_eodhd",
    "us_tw_massive",
]

PROFILE_DATASETS: Final[dict[DatasetProfile, tuple[str, ...]]] = {
    "tw_only": ("tpex_official", "twse_official"),
    "us_only_eodhd": ("eodhd_us",),
    "us_tw_eodhd": ("eodhd_us", "tpex_official", "twse_official"),
    "us_tw_massive": ("massive_us", "tpex_official", "twse_official"),
}

DATASET_RUNTIME_PROVIDER: Final[dict[str, str]] = {
    "eodhd_us": "eodhd",
    "massive_us": "massive",
    "tpex_official": "tpex_official",
    "twse_official": "twse_official",
}

RUNTIME_PROVIDER_DATASET: Final[dict[str, str]] = {
    provider: dataset for dataset, provider in DATASET_RUNTIME_PROVIDER.items()
}

PROFILE_RUNTIME_PROVIDERS: Final[dict[DatasetProfile, tuple[str, ...]]] = {
    profile: tuple(sorted(DATASET_RUNTIME_PROVIDER[dataset] for dataset in datasets))
    for profile, datasets in PROFILE_DATASETS.items()
}


def selected_datasets(profile: DatasetProfile) -> list[str]:
    """Return canonical durable dataset identifiers for one profile."""

    return sorted(PROFILE_DATASETS[profile])


def runtime_providers(profile: DatasetProfile) -> list[str]:
    """Return canonical acquisition provider identifiers for one profile."""

    return sorted(PROFILE_RUNTIME_PROVIDERS[profile])
