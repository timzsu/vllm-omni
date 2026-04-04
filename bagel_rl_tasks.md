# BAGEL RL Support -- Task Breakdown

**RFC**: [vllm-project/vllm-omni#1904](https://github.com/vllm-project/vllm-omni/issues/1904)  
**Reference**: [Qwen-Image RL integration (verl#4639)](https://github.com/verl-project/verl/issues/4639)

---

## Dependency Graph

```
Task 1 (LoRA names) ───> Task 6 (Tensor LoRA) ───> Task 7 (Cross-stage LoRA)

Task 2 (Trajectory) ───> Task 3 (Scheduler) ───> Task 4 (Log-probs) ───> Task 5 (BagelRLPipeline)
```

Two parallel starting points: **Tasks 1 and 2**.

---

## Task 1: LoRA Manager -- Add Bagel Component Names (Small)

**Problem**: `DiffusionLoRAManager._replace_layers_with_lora()` hard-codes `("transformer", "transformer_2", "dit")` at `manager.py:369`. BAGEL uses `"bagel"` / `"language_model"`, so LoRA layers are never discovered.

**Changes**:
- `vllm_omni/diffusion/lora/manager.py:369` -- add `"bagel"`, `"language_model"` to component tuple
- `vllm_omni/diffusion/models/bagel/bagel_transformer.py` -- add `stacked_params_mapping` for packed QKV
- Tests in `tests/diffusion/lora/`

**Deps**: None

---

## Task 2: Trajectory Recording (Medium) -- DONE (PR #1)

**Problem**: `Bagel.generate_image()` discards intermediate latents/timesteps. `DiffusionOutput` has `trajectory_latents` and `trajectory_timesteps` fields but BAGEL never populates them.

**Implementation**:
- `bagel_transformer.py` -- added `return_trajectory_latents: bool = False` param to `generate_image()` and `_generate_image_parallel()`. All 4 loop variants (batched CFG, SP+CFG, SP-only, CFG-parallel) append `x_t.clone()` and `timesteps[i]` after each Euler step when enabled
- `pipeline_bagel.py` -- reads `return_trajectory_latents` / `return_trajectory_decoded` from `DiffusionSamplingParams`, passes through, populates `DiffusionOutput.trajectory_latents` (stacked tensor) and `DiffusionOutput.trajectory_timesteps`
- Return signature: `(unpacked_latent, traj_latents_or_None, traj_timesteps_or_None)`
- Zero overhead when disabled (default)

**Deps**: None

---

## Task 3: Pluggable Scheduler for Stochastic Sampling (Medium)

**Problem**: BAGEL hard-codes the deterministic Euler ODE step (`x_t = x_t - v_t * dt`). RL exploration needs stochastic samplers (SDE, CPS) that inject noise and compute log-probs.

**Changes**:
- Abstract the `x_t = x_t - v_t * dt` line behind an optional scheduler object passed via `extra_args["scheduler"]` or a new `scheduler` param
- Default: existing inline Euler ODE (backward compatible, no scheduler object needed)
- Scheduler interface: `step(model_output, timestep, sample, dt, ...) -> SchedulerStepOutput(prev_sample, log_prob, mean, std)`
- Must work with all 4 loop variants (each currently has `x_t = x_t - v_t.to(x_t.device) * dts[i]`)
- Support `noise_level`, `sde_window`, `sde_type` params

**Key files**: `bagel_transformer.py`, `pipeline_bagel.py`  
**Deps**: Task 2

---

## Task 4: Diffusion Log-Probabilities (Medium)

**Problem**: FlowGRPO needs per-step log-probs from the denoising process. With the scheduler abstraction from Task 3, we can implement log-prob computation as a scheduler.

**Changes**:
- Implement `FlowMatchSDEScheduler` following `tests/e2e/offline_inference/custom_pipeline/flow_match_sde_scheduler.py` (Gaussian transition: `log_prob = -((sample - mean)^2) / (2 * var) - log(std) - log(sqrt(2*pi))`)
- Collect `log_prob` per step alongside trajectory latents (extend trajectory recording from Task 2)
- Store in `DiffusionOutput` (add `trajectory_log_probs` field)

**Key files**: `bagel_transformer.py`, `pipeline_bagel.py`, new scheduler file  
**Deps**: Task 3

---

## Task 5: BagelRLPipeline -- Custom Pipeline for RL Rollouts (Medium)

**Problem**: Need a turnkey pipeline subclass for verl integration, like `QwenImagePipelineWithLogProbForTest`.

**Changes**:
- New `vllm_omni/diffusion/models/bagel/pipeline_bagel_rl.py`
- Extends `BagelPipeline`:
  - Always enables trajectory recording + SDE scheduler
  - Accepts pre-tokenized prompt IDs via `OmniCustomPrompt` dicts (for verl)
  - Returns full RL bundle in `DiffusionOutput`: `trajectory_latents`, `trajectory_timesteps`, `trajectory_log_probs`, `prompt_embeds`
- Loaded via `CustomPipelineWorkerExtension` / `custom_pipeline_args`

**Reference**: `tests/e2e/offline_inference/custom_pipeline/qwen_image_pipeline_with_logprob.py`  
**Deps**: Tasks 2, 3, 4

---

## Task 6: In-Memory Tensor LoRA Loading (Medium/Large)

**Problem**: `DiffusionLoRAManager._load_adapter()` requires a file path. verl's training loop updates LoRA weights in-memory and needs to load them as tensors without disk I/O.

**Changes**:
- `vllm_omni/diffusion/lora/manager.py` -- add `_load_adapter_from_tensors()`, dispatch in `add_adapter()` based on whether request carries tensors or path
- Extend `LoRARequest` with optional `tensors` field or create `TensorLoRARequest` subclass
- Same activation flow as file-loaded adapters

**Reference**: verl's `OmniTensorLoRARequest` pattern from PR verl#5616  
**Deps**: Task 1

---

## Task 7: Cross-Stage LoRA -- Independent Stage 0 + Stage 1 Adapters (Medium)

**Problem**: Need separate LoRA adapters for Stage 0 (Thinker/LLM) and Stage 1 (DiT). Current manager applies one adapter globally.

**Changes**:
- `vllm_omni/diffusion/lora/manager.py` -- add `target_components` filtering to `_replace_layers_with_lora()` and `_activate_adapter()`
  - `target_components=["bagel.language_model"]` for Stage 0
  - `target_components=["bagel"]` for Stage 1
- `vllm_omni/diffusion/worker/diffusion_worker.py` -- wire per-stage adapter requests

**Deps**: Tasks 1, 6
