# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end test for BAGEL LoRA support (Stage 1 / DiT).

Validates that LoRA adapters are correctly loaded, applied with controllable
scale, and cleanly deactivated.  Uses a synthetic rank-1 adapter targeting the
first decoder layer's QKV projection.

Assertions:
  (a) LoRA at scale=1.0 visibly changes the output  (diff > 0.5)
  (b) scale=2.0 produces a larger delta than scale=1.0  (linearity)
  (c) The delta is bounded  (diff < 80, not corrupted)
  (d) Deactivating LoRA exactly restores the baseline  (diff == 0)
"""

import json
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from tests.conftest import modify_stage_config
from tests.utils import hardware_test
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id

MODEL = "ByteDance-Seed/BAGEL-7B-MoT"
BAGEL_STAGE_CONFIG = str(Path(__file__).parent / "stage_configs" / "bagel_sharedmemory_ci.yaml")
DEFAULT_PROMPT = "<|im_start|>A cute cat<|im_end|>"


# ---------------------------------------------------------------------------
# Helpers (reused from test_bagel_text2img.py patterns)
# ---------------------------------------------------------------------------


def _resolve_stage_config(config_path: str, run_level: str) -> str:
    if run_level == "advanced_model":
        return modify_stage_config(
            config_path,
            deletes={
                "stage_args": {
                    0: ["engine_args.load_format"],
                    1: ["engine_args.load_format"],
                }
            },
        )
    return config_path


def _configure_sampling_params(omni: Omni, num_inference_steps: int = 2) -> list:
    params_list = omni.default_sampling_params_list
    if len(params_list) > 1:
        params_list[1].num_inference_steps = num_inference_steps
        params_list[1].extra_args = {
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 1.5,
        }
    return params_list


def _extract_generated_image(omni_outputs: list) -> Image.Image | None:
    for req_output in omni_outputs:
        if images := getattr(req_output, "images", None):
            return images[0]
        if hasattr(req_output, "request_output") and req_output.request_output:
            stage_out = req_output.request_output
            if hasattr(stage_out, "images") and stage_out.images:
                return stage_out.images[0]
    return None


def _generate_bagel_image(omni: Omni) -> Image.Image:
    params_list = _configure_sampling_params(omni)
    outputs = list(
        omni.generate(
            prompts=[{"prompt": DEFAULT_PROMPT, "modalities": ["image"]}],
            sampling_params_list=params_list,
        )
    )
    img = _extract_generated_image(outputs)
    assert img is not None, "No image generated"
    return img


def _generate_bagel_image_with_lora(
    omni: Omni,
    lora_request: LoRARequest,
    lora_scale: float = 1.0,
) -> Image.Image:
    params_list = _configure_sampling_params(omni)
    params_list[1].lora_request = lora_request
    params_list[1].lora_scale = lora_scale
    outputs = list(
        omni.generate(
            prompts=[{"prompt": DEFAULT_PROMPT, "modalities": ["image"]}],
            sampling_params_list=params_list,
        )
    )
    img = _extract_generated_image(outputs)
    assert img is not None, "No image generated with LoRA"
    return img


def _write_bagel_lora(adapter_dir: Path) -> str:
    """Create a synthetic rank-1 LoRA adapter for BAGEL's DiT QKV projection."""
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # BAGEL hidden_size=4096, 32 heads, head_dim=128
    dim = 4096
    module_name = "bagel.language_model.model.layers.0.self_attn.qkv_proj"
    rank = 1

    lora_a = torch.zeros((rank, dim), dtype=torch.float32)
    lora_a[0, 0] = 1.0

    # QKVParallelLinear packs (Q, K, V), total out_dim = 3 * dim
    lora_b = torch.zeros((3 * dim, rank), dtype=torch.float32)
    # Apply a bounded delta to Q slice only
    lora_b[:dim, 0] = 0.1

    save_file(
        {
            f"base_model.model.{module_name}.lora_A.weight": lora_a,
            f"base_model.model.{module_name}.lora_B.weight": lora_b,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": rank, "lora_alpha": rank, "target_modules": [module_name]}),
        encoding="utf-8",
    )
    return str(adapter_dir)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_bagel_lora_scale_and_deactivation(run_level, tmp_path):
    """Validate LoRA effect, scale linearity, bounded perturbation, and clean deactivation."""
    config_path = _resolve_stage_config(BAGEL_STAGE_CONFIG, run_level)
    omni = Omni(model=MODEL, stage_configs_path=config_path, stage_init_timeout=300)
    try:
        lora_dir = _write_bagel_lora(tmp_path / "bagel_lora")
        lora_request = LoRARequest(
            lora_name="test",
            lora_int_id=stable_lora_int_id(lora_dir),
            lora_path=lora_dir,
        )

        # 1) Baseline (no LoRA)
        baseline = _generate_bagel_image(omni)

        # 2) LoRA with scale=1.0
        img_1x = _generate_bagel_image_with_lora(omni, lora_request, lora_scale=1.0)

        # 3) LoRA with scale=2.0
        img_2x = _generate_bagel_image_with_lora(omni, lora_request, lora_scale=2.0)

        # 4) No LoRA again (deactivation)
        restored = _generate_bagel_image(omni)

        baseline_arr = np.array(baseline, dtype=np.int16)
        img_1x_arr = np.array(img_1x, dtype=np.int16)
        img_2x_arr = np.array(img_2x, dtype=np.int16)
        restored_arr = np.array(restored, dtype=np.int16)

        diff_1x = np.abs(baseline_arr - img_1x_arr).mean()
        diff_2x = np.abs(baseline_arr - img_2x_arr).mean()
        diff_restored = np.abs(baseline_arr - restored_arr).mean()

        # (a) Adapter has visible effect
        assert diff_1x > 0.5, f"LoRA scale=1.0 had no visible effect: diff={diff_1x}"

        # (b) Scale parameter works (2x produces larger change)
        assert diff_2x > diff_1x, f"LoRA scale not applied: diff_2x={diff_2x} <= diff_1x={diff_1x}"

        # (c) Output is not corrupted
        assert diff_2x < 80, f"LoRA output looks corrupted: diff_2x={diff_2x}"

        # (d) Deactivation fully restores base model
        assert diff_restored == 0.0, f"Base model not restored after LoRA deactivation: diff={diff_restored}"
    finally:
        omni.close()
