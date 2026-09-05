"""Exact code-digest migrations that preserve compatible interrupted runs."""

from __future__ import annotations

from typing import Final

CHECKPOINT_RETENTION_MIGRATIONS: Final = (
    {
        "id": "adaptive-gpu-resource-planning-v1",
        "from_files": {
            "training.py": (
                "45d495828451462edb0e059c8e0d67700f104cbe0c6b421e4afaa92b5533017a"
            ),
        },
        "to_files": {
            "training.py": (
                "a2521d3b5389c345e2e956e441f7806840e5b721c0c73be3ed1550a6bba4c1ff"
            ),
        },
    },
    {
        "id": "best-five-plus-adaptive-gpu-resource-planning-v1",
        "from_files": {
            "checkpointing.py": (
                "5e73422fe8686366220dc7d07159c39b54228b3913f86ee759b0761d554c6b46"
            ),
            "run_contract.py": (
                "d54028f40a8e1eb5e8a86cd02c3855704c3301fd10c118d2d3e7645e53870ab8"
            ),
            "training.py": (
                "45d495828451462edb0e059c8e0d67700f104cbe0c6b421e4afaa92b5533017a"
            ),
        },
        "to_files": {
            "checkpointing.py": (
                "5c58845e417dcf3aed6381f9136048ee37535586c9cd63617b8b9d0ec45a2194"
            ),
            "run_contract.py": (
                "2b41ead1d3a74040419d393374755aee137d5fbe68a49da88eb7fbe9a1773044"
            ),
            "training.py": (
                "a2521d3b5389c345e2e956e441f7806840e5b721c0c73be3ed1550a6bba4c1ff"
            ),
        },
    },
    {
        "id": "hardware-portable-runtime-plan-v2",
        "from_files": {
            "training.py": (
                "d54aaa84c5adda60051cb3574baa54e4009211629198f540c99ae11b83173437"
            ),
        },
        "to_files": {
            "training.py": (
                "a2521d3b5389c345e2e956e441f7806840e5b721c0c73be3ed1550a6bba4c1ff"
            ),
        },
    },
)
