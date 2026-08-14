"""Minimal LoRA layers for the non-Transformers Kronos predictor."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """Keep one frozen linear layer plus a trainable low-rank update."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank < 1 or alpha <= 0.0 or not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA rank/alpha/dropout are invalid")
        base_layer.requires_grad_(False)
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        factory_kwargs = {
            "device": base_layer.weight.device,
            "dtype": base_layer.weight.dtype,
        }
        self.lora_a = nn.Linear(
            base_layer.in_features,
            rank,
            bias=False,
            **factory_kwargs,
        )
        self.lora_b = nn.Linear(
            rank,
            base_layer.out_features,
            bias=False,
            **factory_kwargs,
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        base = self.base_layer(inputs)
        adapter_inputs = self.dropout(inputs).to(dtype=self.lora_a.weight.dtype)
        update = self.lora_b(self.lora_a(adapter_inputs)) * self.scaling
        return base + update.to(dtype=base.dtype)


def inject_lora(
    module: nn.Module,
    *,
    target_modules: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
    allowed_prefixes: Sequence[str] = ("transformer.",),
) -> tuple[str, ...]:
    """Replace exact predictor linear-module suffixes and fail closed on drift."""

    targets = tuple(target_modules)
    allowed = tuple(allowed_prefixes)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("target_modules must contain unique names")

    replacements: list[tuple[nn.Module, str, str, nn.Linear]] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for child_name, child in parent.named_children():
            path = f"{prefix}.{child_name}" if prefix else child_name
            if (
                isinstance(child, nn.Linear)
                and child_name in targets
                and any(path.startswith(allowed_prefix) for allowed_prefix in allowed)
            ):
                replacements.append((parent, child_name, path, child))
                continue
            visit(child, path)

    visit(module, "")
    matched_targets = {path.rsplit(".", 1)[-1] for _, _, path, _ in replacements}
    missing_targets = sorted(set(targets) - matched_targets)
    if missing_targets:
        raise ValueError(
            "Kronos predictor does not expose configured LoRA targets: "
            + ", ".join(missing_targets)
        )
    if not replacements:
        raise ValueError("No Kronos predictor layers matched the LoRA contract")

    names: list[str] = []
    for parent, child_name, path, base_layer in replacements:
        setattr(
            parent,
            child_name,
            LoRALinear(
                base_layer,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )
        names.append(path)
    return tuple(sorted(names))


def lora_parameter_names(module: nn.Module) -> tuple[str, ...]:
    """Return the exact trainable LoRA parameter names for checkpoint metadata."""

    return tuple(
        sorted(
            name
            for name, parameter in module.named_parameters()
            if parameter.requires_grad and (".lora_a." in name or ".lora_b." in name)
        )
    )


def merge_lora_weights(module: nn.Module) -> nn.Module:
    """Create an explicitly merged copy for optional standalone export."""

    import copy

    merged = copy.deepcopy(module)

    def visit(parent: nn.Module) -> None:
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                base = child.base_layer
                delta = child.lora_b.weight @ child.lora_a.weight
                with torch.no_grad():
                    base.weight.add_(delta.to(dtype=base.weight.dtype), alpha=child.scaling)
                setattr(parent, child_name, base)
            else:
                visit(child)

    visit(merged)
    return merged
