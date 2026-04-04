#!/usr/bin/env python3
"""Create sub-issues for RFC #1904 (BAGEL RL Support) on timzsu/vllm-omni.

Usage:
    export GITHUB_TOKEN=ghp_...
    python create_sub_issues.py
"""

import json
import os
import subprocess
import sys

REPO = "timzsu/vllm-omni"
PARENT_ISSUE = 1904

SUB_ISSUES = [
    {
        "title": "[BAGEL RL 1/8] LoRA Manager: Add Bagel component name support",
        "body": """## Summary

`DiffusionLoRAManager._replace_layers_with_lora()` at `vllm_omni/diffusion/lora/manager.py:369` hard-codes component names `("transformer", "transformer_2", "dit")`. BAGEL uses `"bagel"` (which nests `"language_model"`), so the LoRA manager cannot discover linear layers in Bagel's DiT.

## What to do

1. Add `"bagel"` and `"language_model"` to the component name tuple in `_replace_layers_with_lora()`.
2. Consider making the component list configurable or derivable from the pipeline (e.g., a class attribute).
3. Add `stacked_params_mapping` to BAGEL model classes so packed QKV projections are handled correctly (similar to how SD3.5/QwenImage define theirs).
4. Add unit tests verifying LoRA layer discovery in a Bagel-style model.

## Key files
- `vllm_omni/diffusion/lora/manager.py` (line 369)
- `vllm_omni/diffusion/models/bagel/bagel_transformer.py` (add `stacked_params_mapping`)
- New tests in `tests/diffusion/lora/`

## Dependencies
None -- this can start immediately.

## Complexity
Small

## Parent RFC
Part of #1904
""",
    },
    {
        "title": "[BAGEL RL 2/8] Trajectory Recording: Populate trajectory fields in DiffusionOutput",
        "body": """## Summary

BAGEL's `Bagel.generate_image()` in `bagel_transformer.py` runs a denoising loop (`x_t = x_t - v_t * dt`) but discards all intermediate states. `DiffusionOutput` already has `trajectory_latents`, `trajectory_timesteps`, and `custom_output` fields, but BAGEL never populates them.

## What to do

1. Add opt-in trajectory capture controlled via `extra_args["record_trajectory"]=True`.
2. In the denoising loop, when enabled, collect `x_t.clone()` and the current timestep at each step.
3. Return them in `DiffusionOutput.trajectory_latents`, `trajectory_timesteps`, and `custom_output`.
4. Must handle **all** loop variants:
   - `generate_image()` (standard)
   - `_generate_image_parallel()` (CFG parallel)
   - SP+CFG variants
5. Performance target: zero overhead when `record_trajectory=False` (default).

## Key files
- `vllm_omni/diffusion/models/bagel/bagel_transformer.py` -- modify `generate_image()`, `_generate_image_parallel()`, and SP variants
- `vllm_omni/diffusion/models/bagel/pipeline_bagel.py` -- wire trajectory data into `DiffusionOutput`
- New tests verifying trajectory data shapes and content

## Dependencies
None -- this can start immediately.

## Complexity
Medium

## Parent RFC
Part of #1904
""",
    },
    {
        "title": "[BAGEL RL 3/8] Diffusion Log-Probabilities: Compute per-step log-probs",
        "body": """## Summary

FlowGRPO requires per-step log-probabilities from the denoising process. For BAGEL's flow-matching velocity model, this means computing the log-probability of each denoising transition under a stochastic (SDE) formulation.

## What to do

1. Refactor the `x_t` update in the denoising loop to support a scheduler callback that returns `(prev_sample, log_prob)`.
2. Implement log-prob computation following the pattern in `FlowMatchSDEDiscreteSchedulerForTest` at `tests/e2e/offline_inference/custom_pipeline/flow_match_sde_scheduler.py`.
3. The log-prob math for SDE:
   - `prev_sample_mean = sample * (...) + model_output * (...) * dt`
   - `prev_sample = prev_sample_mean + std_dev_t * sqrt(-dt) * noise`
   - `log_prob = -((prev_sample - mean)^2) / (2 * variance) - log(std) - log(sqrt(2*pi))`
4. Store accumulated log-probs in `DiffusionOutput.custom_output["all_log_probs"]`.

## Reference
- `tests/e2e/offline_inference/custom_pipeline/flow_match_sde_scheduler.py` -- SDE/CPS log-prob math
- `tests/e2e/offline_inference/custom_pipeline/qwen_image_pipeline_with_logprob.py` -- how Qwen-Image uses it

## Key files
- `vllm_omni/diffusion/models/bagel/bagel_transformer.py`
- `vllm_omni/diffusion/models/bagel/pipeline_bagel.py`

## Dependencies
- **PR 2** (trajectory recording infrastructure)

## Complexity
Medium

## Parent RFC
Part of #1904
""",
    },
    {
        "title": "[BAGEL RL 4/8] Custom Scheduler: Pluggable scheduler for stochastic sampling",
        "body": """## Summary

BAGEL hard-codes a deterministic Euler ODE update (`x_t = x_t - v_t * dt`). RL exploration requires stochastic samplers (Euler SDE, CPS) that inject noise for exploration.

## What to do

1. Refactor the denoising loop to accept an optional scheduler object/callback for the `x_t` update step, instead of hard-coding the Euler ODE formula.
2. The scheduler is injected via `extra_args["scheduler"]` or `custom_pipeline_args`.
3. When no scheduler is provided, fall back to the existing deterministic Euler update (backward compatible).
4. The scheduler interface should match what `FlowMatchSDEDiscreteSchedulerForTest` provides:
   - `step(model_output, timestep, sample, ...) -> (prev_sample, log_prob, prev_sample_mean, std_dev_t)`
5. Support SDE parameters: `noise_level`, `sde_window`, `sde_type` ("sde" or "cps").

## Reference
- `tests/e2e/offline_inference/custom_pipeline/flow_match_sde_scheduler.py`

## Key files
- `vllm_omni/diffusion/models/bagel/bagel_transformer.py`
- `vllm_omni/diffusion/models/bagel/pipeline_bagel.py`

## Dependencies
- **PR 3** (log-prob computation needs the scheduler)

## Complexity
Medium

## Parent RFC
Part of #1904
""",
    },
    {
        "title": "[BAGEL RL 5/8] BagelRLPipeline: Custom pipeline subclass for RL rollouts",
        "body": """## Summary

Create a `BagelRLPipeline` that extends `BagelPipeline`, mirroring how `QwenImagePipelineWithLogProbForTest` extends `QwenImagePipeline`. This provides a turnkey RL rollout pipeline for verl integration.

## What to do

1. Create `BagelRLPipeline` extending `BagelPipeline`.
2. The RL pipeline should:
   - Always enable trajectory recording
   - Use the SDE scheduler for stochastic sampling
   - Accept pre-tokenized prompt IDs via `OmniCustomPrompt` dicts (for verl integration, bypassing tokenization)
   - Return the full RL output bundle in `custom_output`: `all_latents`, `all_log_probs`, `all_timesteps`, `prompt_embeds`
3. Loaded via `CustomPipelineWorkerExtension` / `custom_pipeline_args` mechanism (existing `diffusion_worker.py:302-326`).
4. Add tests.

## Reference
- `tests/e2e/offline_inference/custom_pipeline/qwen_image_pipeline_with_logprob.py` -- the Qwen-Image equivalent

## Key files
- New: `vllm_omni/diffusion/models/bagel/pipeline_bagel_rl.py`
- Tests

## Dependencies
- **PR 2** (trajectory recording)
- **PR 3** (log-probabilities)
- **PR 4** (pluggable scheduler)

## Complexity
Medium

## Parent RFC
Part of #1904
""",
    },
    {
        "title": "[BAGEL RL 6/8] IPC Serialization: Handle tensors in custom_output across process boundaries",
        "body": """## Summary

`vllm_omni/diffusion/ipc.py` only applies shared-memory optimization to `output.output` and `output.trajectory_latents`. RL outputs put large tensors in the `custom_output` dict (`all_latents`, `all_log_probs`, `prompt_embeds`), which currently get pickled through the MessageQueue -- slow and memory-inefficient.

## What to do

1. Extend `_pack_diffusion_fields()` to walk the `custom_output` dict and apply SHM transfer to any `torch.Tensor` value exceeding the 1MB threshold (`_SHM_TENSOR_THRESHOLD`).
2. Extend `_unpack_diffusion_fields()` to reconstruct tensors from SHM handles in `custom_output`.
3. Handle nested dicts (one level deep should suffice).
4. Add tests with tensor-bearing `custom_output` dicts.

## Key files
- `vllm_omni/diffusion/ipc.py` (lines 81-122)
- New tests

## Dependencies
None -- this can start immediately and is independent of other PRs.

## Complexity
Small

## Parent RFC
Part of #1904
""",
    },
    {
        "title": "[BAGEL RL 7/8] LoRA Manager: In-memory tensor LoRA loading",
        "body": """## Summary

`DiffusionLoRAManager._load_adapter()` uses `LoRAModel.from_local_checkpoint()` which requires a file path. For verl's RL training loop, LoRA weights are updated in-memory after each training step and need to be loaded as tensors directly without writing to disk. The verl side uses `OmniTensorLoRARequest` to achieve this for Qwen-Image.

## What to do

1. Add `_load_adapter_from_tensors()` method alongside the existing `_load_adapter()`.
2. Dispatch in `add_adapter()` based on whether the `LoRARequest` carries in-memory tensors or a file path.
3. Define how tensor payloads are passed -- either extend `LoRARequest` with an optional `tensors` field, or create a `TensorLoRARequest` subclass.
4. Ensure the tensor-loaded adapter goes through the same layer-replacement and activation flow as file-loaded ones.
5. Add tests for in-memory LoRA loading.

## Reference
- verl's `OmniTensorLoRARequest` + `VLLMOmniHijack` pattern from verl PR #5616

## Key files
- `vllm_omni/diffusion/lora/manager.py`
- `vllm_omni/lora/request.py`

## Dependencies
- **PR 1** (Bagel component names, so LoRA can target Bagel layers)

## Complexity
Medium/Large

## Parent RFC
Part of #1904
""",
    },
    {
        "title": "[BAGEL RL 8/8] Cross-Stage LoRA: Independent Stage 0 + Stage 1 adapters",
        "body": """## Summary

Enable loading separate LoRA adapters for BAGEL's Stage 0 (Qwen2MoT LLM / Thinker) and Stage 1 (DiT / image generation). The current manager applies a single adapter globally. Multi-stage alignment requires independent LoRA adapters per stage.

## What to do

1. Add component-scoped adapter support: callers specify `target_components=["bagel.language_model"]` (Stage 0 / Thinker) vs `target_components=["bagel"]` (Stage 1 / DiT) to control which model parts a LoRA applies to.
2. Add component filtering to `_replace_layers_with_lora()` and `_activate_adapter()` in `DiffusionLoRAManager`.
3. Wire per-stage adapter requests through `DiffusionWorker`.
4. Test: load two different LoRA adapters, one for each stage, verify they apply to the correct layers.

## Use case
FlowGRPO multi-stage alignment where:
- Stage 0 LoRA adapts the LLM's text reasoning for better image prompt generation
- Stage 1 LoRA adapts the DiT for improved image quality

## Key files
- `vllm_omni/diffusion/lora/manager.py`
- `vllm_omni/diffusion/worker/diffusion_worker.py`

## Dependencies
- **PR 1** (Bagel component names)
- **PR 7** (tensor LoRA loading, for the RL training loop use case)

## Complexity
Medium

## Parent RFC
Part of #1904
""",
    },
]


def create_issue(title: str, body: str) -> dict:
    """Create a GitHub issue using gh CLI."""
    result = subprocess.run(
        [
            "gh", "issue", "create",
            "--repo", REPO,
            "--title", title,
            "--body", body,
            "--label", "RL",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR creating '{title}': {result.stderr}", file=sys.stderr)
        return {"title": title, "error": result.stderr}
    url = result.stdout.strip()
    print(f"Created: {url}")
    return {"title": title, "url": url}


def add_sub_issue(parent_number: int, child_url: str):
    """Add a sub-issue relationship using gh CLI."""
    # Extract issue number from URL
    child_number = child_url.rstrip("/").split("/")[-1]
    result = subprocess.run(
        [
            "gh", "api",
            "--method", "POST",
            f"/repos/{REPO}/issues/{parent_number}/sub_issues",
            "-f", f"sub_issue_id={child_number}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: Could not add sub-issue link: {result.stderr.strip()}", file=sys.stderr)


def main():
    if not os.environ.get("GITHUB_TOKEN") and not os.path.exists(os.path.expanduser("~/.config/gh/hosts.yml")):
        print("Please set GITHUB_TOKEN or authenticate with `gh auth login`", file=sys.stderr)
        sys.exit(1)

    created = []
    for issue in SUB_ISSUES:
        result = create_issue(issue["title"], issue["body"])
        created.append(result)
        if "url" in result:
            add_sub_issue(PARENT_ISSUE, result["url"])

    print("\n--- Summary ---")
    for c in created:
        if "url" in c:
            print(f"  {c['title']} -> {c['url']}")
        else:
            print(f"  {c['title']} -> FAILED: {c['error']}")


if __name__ == "__main__":
    main()
