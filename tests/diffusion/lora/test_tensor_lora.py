# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
from vllm.lora.lora_weights import LoRALayerWeights

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.tensor_lora_request import TensorLoRARequest

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
) -> TensorLoRARequest:
    if module_names is None:
        module_names = ["transformer.foo"]
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in module_names:
        lora_a = torch.randn(rank, in_dim)
        lora_b = torch.randn(out_dim, rank)
        tensors[name] = (lora_a, lora_b)
    return TensorLoRARequest(
        lora_name=f"tensor_adapter_{adapter_id}",
        lora_int_id=adapter_id,
        lora_tensors=tensors,
        rank=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules or [],
    )


# -- Tests ---------------------------------------------------------------------


def test_load_adapter_from_tensors_builds_lora_model():
    """_load_adapter_from_tensors should build LoRAModel + PEFTHelper from tensors."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=2,
    )
    manager._expected_lora_modules = ["foo"]

    rank = 4
    request = _make_tensor_request(adapter_id=1, rank=rank)

    lora_model, peft_helper = manager._load_adapter_from_tensors(request)

    assert lora_model.id == 1
    assert lora_model.rank == rank
    assert "transformer.foo" in lora_model.loras
    assert peft_helper.r == rank
    assert peft_helper.lora_alpha == rank


def test_load_adapter_from_tensors_casts_dtype():
    """Tensors should be cast to the manager's dtype and moved to CPU."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=2,
    )
    manager._expected_lora_modules = ["foo"]

    # Create float32 tensors
    request = TensorLoRARequest(
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

    lora_model, _ = manager._load_adapter_from_tensors(request)
    lora = lora_model.loras["transformer.foo"]
    assert lora.lora_a.dtype == torch.bfloat16
    assert lora.lora_b.dtype == torch.bfloat16
    assert lora.lora_a.device == torch.device("cpu")


def test_load_adapter_from_tensors_raises_without_expected_modules():
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
        manager._load_adapter_from_tensors(request)


def test_add_adapter_dispatches_tensor_request(monkeypatch):
    """add_adapter should use _load_adapter_from_tensors for TensorLoRARequest."""
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_cached_adapters=2,
    )

    load_calls: list[str] = []

    def _fake_load_tensors(req: TensorLoRARequest):
        load_calls.append("tensors")
        lora_model = type("LM", (), {"id": req.lora_int_id})()
        peft_helper = type("PH", (), {"r": req.rank})()
        return lora_model, peft_helper

    def _fake_load_file(req):
        load_calls.append("file")
        lora_model = type("LM", (), {"id": req.lora_int_id})()
        peft_helper = type("PH", (), {"r": 4})()
        return lora_model, peft_helper

    monkeypatch.setattr(manager, "_load_adapter_from_tensors", _fake_load_tensors)
    monkeypatch.setattr(manager, "_load_adapter", _fake_load_file)
    monkeypatch.setattr(manager, "_replace_layers_with_lora", lambda _peft: None)

    tensor_req = _make_tensor_request(adapter_id=1)
    manager.add_adapter(tensor_req)
    assert load_calls == ["tensors"]


def test_set_active_adapter_with_tensor_request(monkeypatch):
    """set_active_adapter should work end-to-end with TensorLoRARequest."""
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

    request = TensorLoRARequest(
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
    """Loading a new TensorLoRARequest with the same ID should update weights."""
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
    req1 = TensorLoRARequest(
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
    req2 = TensorLoRARequest(
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
    """_load_adapter_from_tensors should handle multiple module names."""
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

    lora_model, _ = manager._load_adapter_from_tensors(request)
    assert len(lora_model.loras) == 2
    assert "transformer.q_proj" in lora_model.loras
    assert "transformer.v_proj" in lora_model.loras


def test_tensor_lora_request_is_frozen():
    """TensorLoRARequest should be immutable (frozen dataclass)."""
    request = _make_tensor_request()
    with pytest.raises(AttributeError):
        request.lora_name = "changed"


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
