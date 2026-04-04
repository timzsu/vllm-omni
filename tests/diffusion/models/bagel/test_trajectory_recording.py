# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for BAGEL trajectory recording in the denoising loop."""

import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.diffusion.models.bagel.bagel_transformer import (
    Bagel,
    NaiveCache,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NUM_TOKENS = 8
HIDDEN_DIM = 16
NUM_TIMESTEPS = 5
# generate_image uses timesteps[:-1], so actual steps = NUM_TIMESTEPS - 1
EXPECTED_STEPS = NUM_TIMESTEPS - 1


def _make_mock_bagel():
    """Create a mock Bagel with _forward_flow returning constant velocity."""
    mock = MagicMock(spec=Bagel)
    mock._sp_size = 1

    # _forward_flow returns a small constant velocity so x_t changes each step
    def fake_forward_flow(self, x_t, **kwargs):
        return torch.ones_like(x_t) * 0.1

    mock._forward_flow = types.MethodType(fake_forward_flow, mock)
    # _merge_naive_caches is called in the batched CFG path
    mock._merge_naive_caches = types.MethodType(lambda self, caches: NaiveCache(1), mock)

    # Bind the real generate_image to our mock
    mock.generate_image = types.MethodType(Bagel.generate_image, mock)
    return mock


def _make_generate_args(num_tokens=NUM_TOKENS, hidden_dim=HIDDEN_DIM):
    """Minimal tensor arguments for generate_image (no-CFG path)."""
    seq_len = num_tokens + 2  # packed_seqlens includes 2 extra tokens
    return dict(
        packed_text_ids=torch.zeros(2, dtype=torch.long),
        packed_text_indexes=torch.tensor([0, 1], dtype=torch.long),
        packed_init_noises=torch.randn(num_tokens, hidden_dim),
        packed_vae_position_ids=torch.arange(num_tokens, dtype=torch.long),
        packed_vae_token_indexes=torch.arange(2, seq_len, dtype=torch.long),
        packed_seqlens=torch.tensor([seq_len], dtype=torch.int),
        packed_position_ids=torch.arange(seq_len, dtype=torch.long),
        packed_indexes=torch.arange(seq_len, dtype=torch.long),
        past_key_values=NaiveCache(1),
        key_values_lens=torch.tensor([0], dtype=torch.int),
        packed_key_value_indexes=torch.zeros(0, dtype=torch.long),
        num_timesteps=NUM_TIMESTEPS,
        timestep_shift=1.0,
        cfg_text_scale=1.0,  # no CFG → simplest code path
        cfg_img_scale=1.0,
    )


@pytest.fixture()
def bagel_and_args():
    """Shared mock Bagel instance and generate_image arguments."""
    with patch(
        "vllm_omni.diffusion.models.bagel.bagel_transformer.get_classifier_free_guidance_world_size",
        return_value=1,
    ):
        yield _make_mock_bagel(), _make_generate_args()


class TestTrajectoryRecording:
    """Tests for trajectory latent/timestep recording in generate_image."""

    def test_trajectory_disabled_returns_none(self, bagel_and_args):
        bagel, args = bagel_and_args

        unpacked, trajectory_latents, trajectory_timesteps = bagel.generate_image(
            **args, return_trajectory_latents=False
        )

        assert isinstance(unpacked, (list, tuple))
        assert len(unpacked) == 1  # one sequence
        assert trajectory_latents is None
        assert trajectory_timesteps is None

    def test_trajectory_enabled_returns_correct_count(self, bagel_and_args):
        bagel, args = bagel_and_args

        _, trajectory_latents, trajectory_timesteps = bagel.generate_image(**args, return_trajectory_latents=True)

        assert trajectory_latents is not None
        assert trajectory_timesteps is not None
        assert len(trajectory_latents) == EXPECTED_STEPS
        assert len(trajectory_timesteps) == EXPECTED_STEPS

    def test_trajectory_latents_shape_matches_input(self, bagel_and_args):
        bagel, args = bagel_and_args
        expected_shape = args["packed_init_noises"].shape

        _, trajectory_latents, _ = bagel.generate_image(**args, return_trajectory_latents=True)

        for i, lat in enumerate(trajectory_latents):
            assert lat.shape == expected_shape, f"Step {i}: expected {expected_shape}, got {lat.shape}"

    def test_trajectory_latents_are_distinct(self, bagel_and_args):
        bagel, args = bagel_and_args

        _, trajectory_latents, _ = bagel.generate_image(**args, return_trajectory_latents=True)

        for i in range(1, len(trajectory_latents)):
            assert not torch.equal(trajectory_latents[i], trajectory_latents[i - 1]), (
                f"Steps {i - 1} and {i} should differ"
            )

    def test_trajectory_timesteps_are_decreasing(self, bagel_and_args):
        bagel, args = bagel_and_args

        _, _, trajectory_timesteps = bagel.generate_image(**args, return_trajectory_latents=True)

        for i in range(1, len(trajectory_timesteps)):
            assert trajectory_timesteps[i] < trajectory_timesteps[i - 1], (
                f"Timestep {i} ({trajectory_timesteps[i]:.4f}) should be less than "
                f"timestep {i - 1} ({trajectory_timesteps[i - 1]:.4f})"
            )

    def test_trajectory_final_latent_matches_output(self, bagel_and_args):
        bagel, args = bagel_and_args

        unpacked, trajectory_latents, _ = bagel.generate_image(**args, return_trajectory_latents=True)

        # Reconstruct the full final latent from unpacked pieces
        final_latent = torch.cat(unpacked, dim=0)
        assert torch.allclose(trajectory_latents[-1], final_latent, atol=1e-6), (
            "Last trajectory latent should match the final output"
        )
