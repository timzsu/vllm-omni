# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
from vllm.lora.request import LoRARequest


class OmniLoRARequest(LoRARequest):
    """LoRA request that optionally carries in-memory tensors.

    For file-backed adapters, use ``lora_path`` as usual.  For in-memory
    adapters (VeRL RL loops), set ``lora_tensors`` instead of ``lora_path``.

    ``lora_tensors`` maps module names (e.g. ``"self_attn.qkv_proj"``) to
    ``(lora_a, lora_b)`` tensor pairs.

    NOTE: This subclasses vLLM's ``LoRARequest`` (``msgspec.Struct``).
    If upstream changes ``LoRARequest`` fields or ``__post_init__``
    validation, this class may need updating.
    """

    lora_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
    rank: int = 0
    lora_alpha: int = 0
    target_modules: list[str] = []

    def __post_init__(self):
        if self.lora_int_id < 1:
            raise ValueError(f"id must be > 0, got {self.lora_int_id}")
        if not self.lora_path and self.lora_tensors is None:
            raise ValueError("Either lora_path or lora_tensors must be provided")
