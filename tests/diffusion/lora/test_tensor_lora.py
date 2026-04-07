# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
from vllm.lora.request import LoRARequest

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.tensor_lora_request import OmniLoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# -- Helpers -------------------------------------------------------------------


class _DummyLoRALayer:
    """Minimal stub matching the set_lora / reset_lora interface."""

    def __init__(self, n_slices: int = 1, output_slices: tuple[int, ...] = (4,)):
        self.n_slices = n_slices
        self.output_slices = output_slices
        self.set_calls: list[tuple] = []
        self.reset_calls: int = 0

    def set_lora(self, index: int, lora_a, lora_b):
        assert index == 0
        self.set_calls.append((lora_a, lora_b))

    def reset_lora(self, index: int):
        assert index == 0
        self.reset_calls += 1


def _make_tensor_request(
    adapter_id: int = 1,
    rank: int = 4,
    lora_alpha: int = 4,
    module_names: list[str] | None = None,
    in_dim: int = 8,
    out_dim: int = 4,
    target_modules: list[str] | None = None,
) -> OmniLoRARequest:
    if module_names is None:
        module_names = ["transformer.foo"]
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in module_names:
        lora_a = torch.randn(rank, in_dim)
        lora_b = torch.randn(out_dim, rank)
        tensors[name] = (lora_a, lora_b)
    return OmniLoRARequest(
        lora_name=f"tensor_adapter_{adapter_id}",
        lora_int_id=adapter_id,
        lora_tensors=tensors,
        rank=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules or [],
    )


# -- Tests ---------------------------------------------------------------------


def test_load_adapter_builds_lora_model():
    """_load_adapter should build LoRAModel + PEFTHelper from tensors."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=2,
    )
    manager._expected_lora_modules = ["foo"]

    rank = 4
    request = _make_tensor_request(adapter_id=1, rank=rank)

    lora_model, peft_helper = manager._load_adapter(request)

    assert lora_model.id == 1
    assert lora_model.rank == rank
    assert "transformer.foo" in lora_model.loras
    assert peft_helper.r == rank
    assert peft_helper.lora_alpha == rank


def test_load_adapter_casts_dtype():
    """Tensors should be cast to the manager's dtype and moved to CPU."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=2,
    )
    manager._expected_lora_modules = ["foo"]

    # Create float32 tensors
    request = OmniLoRARequest(
        lora_name="fp32_adapter",
        lora_int_id=1,
        lora_tensors={
            "transformer.foo": (
                torch.randn(4, 8, dtype=torch.float32),
                torch.randn(4, 4, dtype=torch.float32),
            ),
        },
        rank=4,
        lora_alpha=4,
        target_modules=[],
    )

    lora_model, _ = manager._load_adapter(request)
    lora = lora_model.loras["transformer.foo"]
    assert lora.lora_a.dtype == torch.bfloat16
    assert lora.lora_b.dtype == torch.bfloat16
    assert lora.lora_a.device == torch.device("cpu")


def test_load_adapter_raises_without_expected_modules():
    """Should raise ValueError if pipeline has no supported LoRA modules."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=2,
    )
    manager._expected_lora_modules = []

    request = _make_tensor_request()
    with pytest.raises(ValueError, match="No supported LoRA modules"):
        manager._load_adapter(request)


def test_add_adapter_registers_tensor_request(monkeypatch):
    """add_adapter should load and register an OmniLoRARequest with tensors."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=2,
    )
    manager._expected_lora_modules = {"foo"}
    monkeypatch.setattr(manager, "_replace_layers_with_lora", lambda _peft: None)

    tensor_req = _make_tensor_request(adapter_id=1)
    manager.add_adapter(tensor_req)
    assert 1 in manager._registered_adapters
    assert "transformer.foo" in manager._registered_adapters[1].loras


def test_set_active_adapter_with_tensor_request(monkeypatch):
    """set_active_adapter should work end-to-end with OmniLoRARequest."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=2,
    )

    dummy_layer = _DummyLoRALayer()
    rank = 4
    in_dim = 8
    out_dim = 4

    lora_a = torch.ones(rank, in_dim, dtype=torch.bfloat16)
    lora_b = torch.ones(out_dim, rank, dtype=torch.bfloat16) * 2.0

    request = OmniLoRARequest(
        lora_name="test_tensor",
        lora_int_id=42,
        lora_tensors={"transformer.foo": (lora_a, lora_b)},
        rank=rank,
        lora_alpha=rank,
        target_modules=[],
    )
    manager._expected_lora_modules = ["foo"]
    manager._lora_modules = {"transformer.foo": dummy_layer}

    # Bypass _replace_layers_with_lora since we manually set _lora_modules
    monkeypatch.setattr(manager, "_replace_layers_with_lora", lambda _peft: None)

    scale = 0.5
    manager.set_active_adapter(request, lora_scale=scale)

    assert 42 in manager._registered_adapters
    assert len(dummy_layer.set_calls) == 1
    set_a, set_b = dummy_layer.set_calls[0]
    # lora_b should be scaled by the external scale
    assert torch.allclose(set_b, lora_b * scale)


def test_tensor_request_replaces_existing_adapter(monkeypatch):
    """Loading a new OmniLoRARequest with the same ID should update weights."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=2,
    )

    dummy_layer = _DummyLoRALayer()
    manager._expected_lora_modules = ["foo"]
    manager._lora_modules = {"transformer.foo": dummy_layer}
    monkeypatch.setattr(manager, "_replace_layers_with_lora", lambda _peft: None)

    rank = 4
    in_dim = 8
    out_dim = 4

    # First request
    req1 = OmniLoRARequest(
        lora_name="step_1",
        lora_int_id=1,
        lora_tensors={"transformer.foo": (torch.ones(rank, in_dim), torch.ones(out_dim, rank))},
        rank=rank,
        lora_alpha=rank,
        target_modules=[],
    )
    manager.set_active_adapter(req1, lora_scale=1.0)
    assert len(dummy_layer.set_calls) == 1

    # Remove and re-add with different weights (simulating RL training step update)
    manager.remove_adapter(1)
    req2 = OmniLoRARequest(
        lora_name="step_2",
        lora_int_id=2,
        lora_tensors={"transformer.foo": (torch.ones(rank, in_dim) * 3, torch.ones(out_dim, rank) * 5)},
        rank=rank,
        lora_alpha=rank,
        target_modules=[],
    )
    manager.set_active_adapter(req2, lora_scale=1.0)
    assert len(dummy_layer.set_calls) == 2
    _, set_b_2 = dummy_layer.set_calls[1]
    assert torch.allclose(set_b_2, torch.ones(out_dim, rank, dtype=torch.bfloat16) * 5)


def test_tensor_lora_multiple_modules():
    """_load_adapter should handle multiple module names."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=2,
    )
    manager._expected_lora_modules = ["q_proj", "v_proj"]

    request = _make_tensor_request(
        adapter_id=1,
        module_names=["transformer.q_proj", "transformer.v_proj"],
    )

    lora_model, _ = manager._load_adapter(request)
    assert len(lora_model.loras) == 2
    assert "transformer.q_proj" in lora_model.loras
    assert "transformer.v_proj" in lora_model.loras


def test_omni_lora_request_is_lora_request():
    """OmniLoRARequest should be an instance of vLLM's LoRARequest."""
    request = _make_tensor_request()
    assert isinstance(request, LoRARequest)


def test_omni_lora_request_rejects_invalid_id():
    """OmniLoRARequest should reject lora_int_id < 1."""
    with pytest.raises(ValueError, match="id must be > 0"):
        OmniLoRARequest(
            lora_name="bad",
            lora_int_id=0,
            lora_tensors={"foo": (torch.ones(2, 4), torch.ones(4, 2))},
            rank=2,
            lora_alpha=2,
        )


def test_omni_lora_request_rejects_neither_path_nor_tensors():
    """OmniLoRARequest should reject when both lora_path and lora_tensors are missing."""
    with pytest.raises(ValueError, match="Either lora_path or lora_tensors"):
        OmniLoRARequest(
            lora_name="empty",
            lora_int_id=1,
        )


def test_tensor_lora_deactivation(monkeypatch):
    """Passing None after a tensor LoRA activation should deactivate."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=2,
    )

    dummy_layer = _DummyLoRALayer()
    manager._expected_lora_modules = ["foo"]
    manager._lora_modules = {"transformer.foo": dummy_layer}
    monkeypatch.setattr(manager, "_replace_layers_with_lora", lambda _peft: None)

    request = _make_tensor_request(adapter_id=1)
    manager.set_active_adapter(request, lora_scale=1.0)
    assert manager._active_adapter_id == 1

    manager.set_active_adapter(None)
    assert manager._active_adapter_id is None
    assert dummy_layer.reset_calls == 1
