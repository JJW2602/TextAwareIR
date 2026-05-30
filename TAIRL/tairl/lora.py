from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


def _module_weight_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.enabled = True
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        # Keep trainable adapter weights fp32 for AMP optimizer stability.
        self.lora_down.to(device=_module_weight_device(base), dtype=torch.float32)
        self.lora_up.to(device=_module_weight_device(base), dtype=torch.float32)
        nn.init.normal_(self.lora_down.weight, std=1.0 / rank)
        nn.init.zeros_(self.lora_up.weight)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.base(x)
        base_out = self.base(x)
        lora_in = self.dropout(x).to(dtype=self.lora_down.weight.dtype)
        lora_out = self.lora_up(self.lora_down(lora_in)) * self.scale
        return base_out + lora_out.to(dtype=base_out.dtype)


class LoRAConv2d1x1(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if base.kernel_size != (1, 1) or base.groups != 1:
            raise ValueError("LoRAConv2d1x1 only supports dense 1x1 convolutions.")
        self.base = base
        self.rank = rank
        self.enabled = True
        self.scale = alpha / rank
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Conv2d(base.in_channels, rank, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)
        # Keep trainable adapter weights fp32 for AMP optimizer stability.
        self.lora_down.to(device=_module_weight_device(base), dtype=torch.float32)
        self.lora_up.to(device=_module_weight_device(base), dtype=torch.float32)
        nn.init.normal_(self.lora_down.weight, std=1.0 / rank)
        nn.init.zeros_(self.lora_up.weight)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.base(x)
        base_out = self.base(x)
        lora_in = self.dropout(x).to(dtype=self.lora_down.weight.dtype)
        lora_out = self.lora_up(self.lora_down(lora_in)) * self.scale
        return base_out + lora_out.to(dtype=base_out.dtype)


@dataclass(frozen=True)
class LoRAInjectionReport:
    wrapped: int
    trainable_params: int
    total_params: int
    wrapped_names: list[str]


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def _matches(name: str, keywords: Iterable[str]) -> bool:
    keywords = list(keywords)
    return not keywords or any(k in name for k in keywords)


def inject_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_keywords: Iterable[str],
    include_linear: bool,
    include_conv1x1: bool,
) -> LoRAInjectionReport:
    wrapped_names: list[str] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            replacement: nn.Module | None = None
            if _matches(full_name, target_keywords):
                if include_linear and isinstance(child, nn.Linear):
                    replacement = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                elif (
                    include_conv1x1
                    and isinstance(child, nn.Conv2d)
                    and child.kernel_size == (1, 1)
                    and child.groups == 1
                ):
                    replacement = LoRAConv2d1x1(child, rank=rank, alpha=alpha, dropout=dropout)
            if replacement is None:
                visit(child, full_name)
            else:
                setattr(parent, child_name, replacement)
                wrapped_names.append(full_name)

    freeze_module(module)
    visit(module)
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return LoRAInjectionReport(
        wrapped=len(wrapped_names),
        trainable_params=trainable,
        total_params=total,
        wrapped_names=wrapped_names,
    )


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in module.state_dict().items()
        if ".lora_down." in name or ".lora_up." in name
    }


@contextmanager
def temporary_lora_enabled(module: nn.Module, enabled: bool):
    lora_modules = [
        child
        for child in module.modules()
        if isinstance(child, (LoRALinear, LoRAConv2d1x1))
    ]
    previous = [child.enabled for child in lora_modules]
    for child in lora_modules:
        child.enabled = enabled
    try:
        yield
    finally:
        for child, child_enabled in zip(lora_modules, previous):
            child.enabled = child_enabled
