# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for cross-stage LoRA routing in the orchestrator and engine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

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
