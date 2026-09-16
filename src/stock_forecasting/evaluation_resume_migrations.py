"""Exact code-digest migrations that preserve completed numerical evaluations."""

from __future__ import annotations

from typing import Final

EVALUATION_RESUME_MIGRATIONS: Final = (
    {
        "id": "checkpoint-runtime-execution-plan-reader-v1",
        "from_files": {
            "cli/evaluate.py": {
                "sha256": "d7a5e3e9edbf43673c447a093959c8fd797af79b4680fa092d00408fb18e2077",
                "size_bytes": 6942,
            },
            "validation_benchmark.py": {
                "sha256": "049f467b57900a823d78f8896f7f9fb44f6dd57e76412a30a7725b263671d71c",
                "size_bytes": 44440,
            },
        },
        "to_files": {
            "cli/evaluate.py": {
                "sha256": "4beb492ecb135edae0f2fc041b7b5f1749a1a57ba6440e6e503b22545d82f5ea",
                "size_bytes": 7149,
            },
            "validation_benchmark.py": {
                "sha256": "30588bb761eedae6469c71bcec756d8c1b1925ad0c57e72efb4e2c3bfd3e61da",
                "size_bytes": 45385,
            },
        },
    },
)
