# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class TensorLoRARequest:
    """LoRA request backed by in-memory tensors instead of a file path.

    Used by VeRL and similar RL training loops that update LoRA weights
    in-memory after each gradient step and need to push them to the
    inference engine without writing to disk.

    ``lora_tensors`` maps module names (e.g. ``"self_attn.qkv_proj"``) to
    ``(lora_a, lora_b)`` tensor pairs.
    """

    lora_name: str
    lora_int_id: int
    lora_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]]
    rank: int
    lora_alpha: int
    target_modules: list[str] = field(default_factory=list)
