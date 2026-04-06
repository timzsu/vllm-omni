# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for BAGEL LoRA support across Stage 0 (Thinker) and Stage 1 (DiT)."""

from __future__ import annotations

import pytest
import torch

from tests.diffusion.lora.conftest import (
    DummyBaseLayerWithLoRA,
    FakeLinearBase,
    fake_replace_submodule,
)
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_FakeLinearBase = FakeLinearBase


# ---------------------------------------------------------------------------
# Stage 0 (Thinker / AR) -- packed_modules_mapping on the AR model class
# ---------------------------------------------------------------------------


class TestStage0ThinkerLoRA:
    """Validate that OmniBagelForConditionalGeneration declares correct LoRA metadata."""

    def test_omni_bagel_supports_lora(self):
        from vllm_omni.model_executor.models.bagel.bagel import (
            OmniBagelForConditionalGeneration,
        )

        assert getattr(OmniBagelForConditionalGeneration, "supports_lora", False) is True

    def test_omni_bagel_packed_modules_mapping_complete(self):
        from vllm_omni.model_executor.models.bagel.bagel import (
            OmniBagelForConditionalGeneration,
        )

        mapping = OmniBagelForConditionalGeneration.packed_modules_mapping
        # Standard Qwen2 projections
        assert mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]
        assert mapping["gate_up_proj"] == ["gate_proj", "up_proj"]
        # MoE generation-mode projections
        assert mapping["qkv_proj_moe_gen"] == [
            "q_proj_moe_gen",
            "k_proj_moe_gen",
            "v_proj_moe_gen",
        ]
        assert mapping["mlp_moe_gen.gate_up_proj"] == [
            "mlp_moe_gen.gate_proj",
            "mlp_moe_gen.up_proj",
        ]


# ---------------------------------------------------------------------------
# Stage 1 (DiT / Diffusion) -- DiffusionLoRAManager with bagel component
# ---------------------------------------------------------------------------


class TestStage1DiTLoRA:
    """Validate DiffusionLoRAManager discovers BAGEL's packed modules."""

    def test_diffusion_lora_manager_discovers_bagel_packed_modules(self):
        """Manager should derive packed→sublayer mapping from stacked_params_mapping."""
        pipeline = torch.nn.Module()
        pipeline.bagel = torch.nn.Module()

        # Simulate a submodule that exposes stacked_params_mapping
        # (as Qwen2MoTForCausalLM does after our load_weights() change)
        language_model = torch.nn.Module()
        language_model.stacked_params_mapping = [
            (".qkv_proj_moe_gen", ".q_proj_moe_gen", "q"),
            (".qkv_proj_moe_gen", ".k_proj_moe_gen", "k"),
            (".qkv_proj_moe_gen", ".v_proj_moe_gen", "v"),
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]
        pipeline.bagel.language_model = language_model

        manager = DiffusionLoRAManager(
            pipeline=pipeline,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            max_cached_adapters=1,
        )

        mapping = manager._packed_modules_mapping
        assert mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]
        assert mapping["qkv_proj_moe_gen"] == [
            "q_proj_moe_gen",
            "k_proj_moe_gen",
            "v_proj_moe_gen",
        ]

    def test_diffusion_lora_manager_replaces_bagel_packed_layer_via_sublayer_target(self, monkeypatch):
        """Targeting sublayer 'q_proj' should replace the fused 'qkv_proj' under bagel."""
        import vllm_omni.diffusion.lora.manager as manager_mod

        monkeypatch.setattr(manager_mod, "BaseLayerWithLoRA", DummyBaseLayerWithLoRA)

        def _fake_from_layer_diffusion(*, layer, **_kwargs):
            return DummyBaseLayerWithLoRA(layer)

        replace_calls: list[str] = []

        monkeypatch.setattr(manager_mod, "from_layer_diffusion", _fake_from_layer_diffusion)
        monkeypatch.setattr(
            manager_mod,
            "replace_submodule",
            lambda root, name, sub: fake_replace_submodule(root, name, sub, replace_calls),
        )

        # Build pipeline with bagel component
        pipeline = torch.nn.Module()
        pipeline.bagel = torch.nn.Module()
        lm = torch.nn.Module()
        lm.stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]
        lm.attn = torch.nn.Module()
        lm.attn.qkv_proj = _FakeLinearBase()
        pipeline.bagel.language_model = lm

        manager = DiffusionLoRAManager(
            pipeline=pipeline,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            max_cached_adapters=1,
        )

        # Treat qkv_proj as 3-slice packed layer
        monkeypatch.setattr(manager, "_get_packed_modules_list", lambda _module: ["q", "k", "v"])

        # Target sublayer "q_proj" -- manager should replace the packed "qkv_proj"
        peft_helper = type("_PH", (), {"r": 1, "target_modules": ["q_proj"]})()
        manager._replace_layers_with_lora(peft_helper)

        assert "language_model.attn.qkv_proj" in replace_calls
        assert "bagel.language_model.attn.qkv_proj" in manager._lora_modules
        # Verify the module was actually replaced in the tree (not just recorded)
        assert isinstance(pipeline.bagel.language_model.attn.qkv_proj, DummyBaseLayerWithLoRA)
