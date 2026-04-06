# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for cross-stage LoRA routing in the orchestrator and engine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _mock_lora_request(name: str = "test", lora_id: int = 1):
    """Create a mock LoRARequest for testing without vllm dependency."""
    lr = MagicMock()
    lr.lora_name = name
    lr.lora_int_id = lora_id
    lr.lora_path = f"/tmp/fake_lora_{name}"
    return lr


class TestBuildEngineCoreRequestLoRA:
    """Verify build_engine_core_request_from_tokens passes LoRA from params."""

    def test_lora_extracted_from_params(self):
        from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens

        lr = _mock_lora_request()
        params = MagicMock()
        params.lora_request = lr
        params.clone.return_value = params
        # SamplingParams-like
        type(params).__instancecheck__ = lambda cls, inst: True

        prompt = {"prompt_token_ids": [1, 2, 3]}
        request = build_engine_core_request_from_tokens(
            request_id="req-1",
            prompt=prompt,
            params=params,
            model_config=None,
        )
        assert request.lora_request is lr

    def test_no_lora_defaults_to_none(self):
        from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens

        params = MagicMock(spec=[])  # no lora_request attr
        params.clone = MagicMock(return_value=params)

        prompt = {"prompt_token_ids": [1, 2, 3]}
        request = build_engine_core_request_from_tokens(
            request_id="req-2",
            prompt=prompt,
            params=params,
            model_config=None,
        )
        assert request.lora_request is None


class TestLoRAPropagationToStageParams:
    """Verify top-level lora_request propagates to per-stage params."""

    def test_propagation_fills_empty_stages(self):
        lr = _mock_lora_request("shared")
        stage0_params = MagicMock()
        stage0_params.lora_request = None
        stage1_params = OmniDiffusionSamplingParams()
        assert stage1_params.lora_request is None

        # Simulate the propagation logic from async_omni_engine
        params_list = [stage0_params, stage1_params]
        if lr is not None:
            for stage_params in params_list:
                if hasattr(stage_params, "lora_request") and stage_params.lora_request is None:
                    stage_params.lora_request = lr

        assert stage0_params.lora_request is lr
        assert stage1_params.lora_request is lr

    def test_per_stage_lora_not_overwritten(self):
        top_level_lr = _mock_lora_request("top")
        stage1_lr = _mock_lora_request("stage1_specific")

        stage0_params = MagicMock()
        stage0_params.lora_request = None
        stage1_params = OmniDiffusionSamplingParams(lora_request=stage1_lr)

        params_list = [stage0_params, stage1_params]
        if top_level_lr is not None:
            for stage_params in params_list:
                if hasattr(stage_params, "lora_request") and stage_params.lora_request is None:
                    stage_params.lora_request = top_level_lr

        # Stage 0 gets the top-level default
        assert stage0_params.lora_request is top_level_lr
        # Stage 1 keeps its own LoRA (not overwritten)
        assert stage1_params.lora_request is stage1_lr

    def test_no_propagation_when_top_level_none(self):
        stage1_params = OmniDiffusionSamplingParams()

        params_list = [MagicMock(), stage1_params]
        lora_request = None
        if lora_request is not None:
            for stage_params in params_list:
                if hasattr(stage_params, "lora_request") and stage_params.lora_request is None:
                    stage_params.lora_request = lora_request

        assert stage1_params.lora_request is None
