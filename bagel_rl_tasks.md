# BAGEL RL Support -- Task Breakdown

**RFC**: [vllm-project/vllm-omni#1904](https://github.com/vllm-project/vllm-omni/issues/1904)
**Reference**: [Qwen-Image RL integration (verl#4639)](https://github.com/verl-project/verl/issues/4639)
**Design doc**: [bagel_rl_lora_design.md](https://github.com/user-attachments/files/26003442/bagel_rl_lora_design.md)

---

## Status Summary

| PR | Tasks | Branch | Status |
|----|-------|--------|--------|
| **PR 1** | Task 1 (LoRA names + Stage 0/1 support) | `bagel-lora-component-names` | DONE |
| **PR 2** | Task 2 (Trajectory recording) | `bagel-rl-trajectory-recording` | DONE |
| **PR 3** | Tasks 6 + 6b + 7 (Zero-copy IPC, tensor LoRA, cross-stage routing) | `bagel-rl-zero-copy-lora-routing` | IN PROGRESS |
| **PR 4** | Tasks 3 + 4 + 5 (SDE scheduler, log-probs, BagelRLPipeline) | -- | TODO |

## Dependency Graph

```
DONE:
  Task 1 (LoRA Stage 0+1) ──────────────────> Task 7 (cross-stage routing) ✓
  Task 2 (trajectory recording) ──> Task 6 (zero-copy IPC) ✓

IN PROGRESS:
  Task 1 ──> Task 6b (tensor LoRA)   ← PR 3

TODO:
  Task 2 ──> Task 3 (SDE scheduler) ──> Task 4 (log-probs) ──> Task 5 (BagelRLPipeline)
                                                                  ↑ one PR (3+4+5)
```

---

## PR 1: Full BAGEL LoRA Support -- DONE

**Branch**: `bagel-lora-component-names`

### Task 1: LoRA Manager + Stage 0/1 Component Support

**Stage 1 (DiT/Diffusion):**
- Added `"bagel"` to `DiffusionLoRAManager._replace_layers_with_lora()` component tuple
- Saved `stacked_params_mapping` as instance attribute on `Qwen2MoTForCausalLM` for packed QKV→sublayer mapping
- Fixed `RuntimeError` from mutating module dict during `named_modules()` iteration

**Stage 0 (Thinker/AR):**
- Added `packed_modules_mapping` to `OmniBagelForConditionalGeneration` with 4 entries: `qkv_proj`, `gate_up_proj`, `qkv_proj_moe_gen`, `mlp_moe_gen.gate_up_proj`

**Tests**: 5 CPU-only unit tests covering both stages.

---

## PR 2: Trajectory Recording -- DONE

**Branch**: `bagel-rl-trajectory-recording`

### Task 2: BAGEL Denoising Loop Trajectory Capture

- Added `return_trajectory_latents: bool = False` param to `Bagel.generate_image()` and `_generate_image_parallel()`
- All 4 denoising loop variants (batched CFG, SP+CFG, SP-only, CFG-parallel) record `x_t.clone()` and `timesteps[i]` at each step
- `pipeline_bagel.py` reads flags from `OmniDiffusionSamplingParams`, populates `DiffusionOutput.trajectory_latents/timesteps/decoded`
- Zero overhead when disabled (default)

**Tests**: 6 CPU-only unit tests for trajectory recording.

---

## PR 3: Zero-Copy IPC + Tensor LoRA + Cross-Stage Routing -- IN PROGRESS

**Branch**: `bagel-rl-zero-copy-lora-routing`

### Task 6: Zero-Copy Trajectory Buffers -- DONE

Extended `ipc.py` SHM packing to cover all trajectory fields:
- `trajectory_timesteps` (list[Tensor] → stacked → SHM → unstack)
- `trajectory_decoded` (same pattern)
- Tensors inside `custom_output` dict (walk values, pack any above 1MB threshold)

### Task 7: Cross-Stage LoRA Routing -- DONE

- Fixed `build_engine_core_request_from_tokens()` hardcoded `lora_request=None` → extract from per-stage params
- Added top-level `lora_request` propagation to downstream stages in `async_omni_engine.py`; per-stage `lora_request` takes precedence

### Task 6b: In-Memory Tensor LoRA Loading -- TODO

**Problem**: `DiffusionLoRAManager._load_adapter()` requires a file path. VeRL updates LoRA weights in-memory after each training step and needs to push them without disk I/O.

**Changes**:
- New `vllm_omni/lora/tensor_lora_request.py` -- `TensorLoRARequest` dataclass with `lora_tensors: dict[str, tuple[Tensor, Tensor]]`, `rank`, `lora_alpha`, `target_modules`
- `vllm_omni/diffusion/lora/manager.py` -- add `_load_adapter_from_tensors()` (creates `PEFTHelper` + `LoRAModel` programmatically, no disk I/O), dispatch in `add_adapter()` on `isinstance(request, TensorLoRARequest)`
- `vllm_omni/lora/request.py` -- export `TensorLoRARequest`
- Tests in `tests/diffusion/lora/test_tensor_lora.py`

**Reference**: VeRL's `OmniTensorLoRARequest` pattern from `verl#5616`

**Tests**: E2E test `test_bagel_lora.py` validates LoRA scale linearity, bounded perturbation, and clean deactivation.

---

## PR 4: BagelRLPipeline (Tasks 3+4+5) -- TODO

### Task 3: Pluggable Scheduler for Stochastic Sampling

**Problem**: BAGEL hard-codes the deterministic Euler ODE step (`x_t = x_t - v_t * dt`). RL exploration needs stochastic samplers (SDE, CPS) that inject noise and compute log-probs.

**Changes**:
- Promote `FlowMatchSDEDiscreteSchedulerForTest` from `tests/e2e/offline_inference/custom_pipeline/flow_match_sde_scheduler.py` to `vllm_omni/diffusion/schedulers/flow_match_sde.py`
- Abstract the Euler step in all 4 loop variants behind optional scheduler dispatch:
  ```python
  if scheduler is not None:
      out = scheduler.step(v_t, timesteps[i], x_t, dts[i])
      x_t = out.prev_sample
  else:
      x_t = x_t - v_t.to(x_t.device) * dts[i]  # existing ODE path
  ```
- Scheduler params via `extra_args`: `sde_type`, `noise_level`, `sde_window_size`

**Key files**: `bagel_transformer.py`, `pipeline_bagel.py`
**Deps**: Task 2

### Task 4: Diffusion Log-Probabilities

**Problem**: FlowGRPO needs per-step log-probs from the denoising process.

**Changes**:
- SDE scheduler's `step()` returns `log_prob` via `SchedulerStepOutput`
- Collect `log_prob` per step alongside trajectory latents
- Store in `DiffusionOutput.custom_output["all_log_probs"]`
- Gaussian transition: `log_prob = -((sample - mean)^2) / (2 * var) - log(std) - log(sqrt(2π))`

**Key files**: `bagel_transformer.py`, scheduler file
**Deps**: Task 3

### Task 5: BagelRLPipeline -- Custom Pipeline for VeRL

**Problem**: Need a turnkey pipeline subclass for VeRL integration.

**Changes**:
- New `vllm_omni/diffusion/models/bagel/pipeline_bagel_rl.py`
- Extends `BagelPipeline`:
  - Swaps scheduler to `FlowMatchSDEScheduler` in `__init__`
  - Enables trajectory recording + log-prob collection
  - Accepts pre-tokenized prompt IDs via `OmniCustomPrompt` dicts (for VeRL)
  - Returns full RL bundle in `DiffusionOutput.custom_output`:
    `all_latents`, `all_log_probs`, `all_timesteps`, `prompt_embeds`
- Loaded via `custom_pipeline_args`:
  ```python
  custom_pipeline_args={"pipeline_class": "vllm_omni.diffusion.models.bagel.pipeline_bagel_rl.BagelRLPipeline"}
  ```

**Reference**: `tests/e2e/offline_inference/custom_pipeline/qwen_image_pipeline_with_logprob.py`
**Deps**: Tasks 2, 3, 4

---

## VeRL Integration (End State)

After all PRs land, VeRL integration works via:

```python
from vllm_omni import Omni
from vllm_omni.lora.request import TensorLoRARequest

omni = Omni(
    model="ByteDance-Seed/BAGEL-7B-MoT",
    stage_configs_path="bagel_sharedmemory.yaml",
    custom_pipeline_args={
        "pipeline_class": "vllm_omni.diffusion.models.bagel.pipeline_bagel_rl.BagelRLPipeline"
    },
)

params_list = omni.default_sampling_params_list
params_list[1].extra_args = {
    "sde_type": "sde",
    "noise_level": 1.0,
    "logprobs": True,
}

# Per-stage LoRA (in-memory tensors from training loop)
params_list[1].lora_request = TensorLoRARequest(
    lora_name="rl_step_42",
    lora_int_id=42,
    lora_tensors=updated_weights,  # dict[str, tuple[Tensor, Tensor]]
    rank=8,
    lora_alpha=8,
    target_modules=["self_attn.qkv_proj"],
)

outputs = list(omni.generate(
    prompts=[{"prompt": "<|im_start|>A cute cat<|im_end|>", "modalities": ["image"]}],
    sampling_params_list=params_list,
))

# RL rollout data
custom = outputs[0].custom_output
trajectory_latents = custom["all_latents"]      # (steps, C, H, W)
log_probs = custom["all_log_probs"]             # (steps,)
timesteps = custom["all_timesteps"]             # (steps,)
prompt_embeds = custom["prompt_embeds"]          # (seq_len, hidden_dim)
```
