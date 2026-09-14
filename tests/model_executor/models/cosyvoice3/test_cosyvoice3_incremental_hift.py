# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Correctness + boundedness tests for the incremental (windowed) HiFT path.

The streaming vocoder used to re-run over the full cumulative mel every chunk
(O(n) per chunk -> O(n^2) total). This runs it over a bounded window instead,
carrying harmonic phase and noise-buffer offset across chunks.

* **Correctness** — windowed output must match the full cumulative re-run.
  With full history the match is bit-exact (proves the phase/noise carry is
  right). With a bounded window, float64 still agrees to ~1e-16 (the window
  is algebraically exact); float32 differs by ~2e-7 absolute (reassociation
  rounding, not truncation), so tests assert a tight atol/rtol rather than
  PCM16 byte-identity, since a sub-LSB perturbation can flip a boundary sample.
* **Boundedness** — the per-chunk window must not grow with mel history.

Tests are parametrized across REAL_CONFIG (24000 Hz -> SineGen2, what
CosyVoice3 ships) and LEGACY_CONFIG (22050 Hz -> SineGen): the fixture used
to default to 22050 Hz only, so a real ~3.4% divergence bug in SineGen2 went
undetected.
"""

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import (
    CausalConvRNNF0Predictor,
    CausalHiFTGenerator,
    SineGen,
    SineGen2,
)
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CHUNK_LEN = 24
TOTAL_MEL = 96
SPM = 480  # samples per mel frame for the small test HiFT (8*5*3*4), matching the real model

LONG_TOTAL_MEL = 400  # long enough that the 64-frame window actually truncates
PCM16_LSB = 1.0 / 32767.0  # smallest representable PCM16 difference

# Worst observed deviation is 1.97e-7 (~2x float32 eps); atol=1e-6 keeps 5x headroom.
ATOL = 1e-6
RTOL = 1e-5

# What CosyVoice3 ships (-> SineGen2). nb_harmonics=8 is required, not a free
# choice: SineGen2's causal buffers are hardcoded to dim 9 (= harmonic_num+1).
REAL_CONFIG = {"sampling_rate": 24000, "nb_harmonics": 8}
LEGACY_CONFIG = {"sampling_rate": 22050, "nb_harmonics": 4}  # -> SineGen, still shipped code

CONFIGS = [
    pytest.param(REAL_CONFIG, id="sinegen2_real_config"),
    pytest.param(LEGACY_CONFIG, id="sinegen1_legacy_config"),
]


def _make_hift(config: dict = REAL_CONFIG) -> CausalHiFTGenerator:
    """Small CausalHiFTGenerator; defaults to the config CosyVoice3 ships."""
    torch.manual_seed(0)
    f0_predictor = CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=16)
    return CausalHiFTGenerator(
        in_channels=80,
        base_channels=32,
        nb_harmonics=config["nb_harmonics"],
        sampling_rate=config["sampling_rate"],
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        f0_predictor=f0_predictor,
    ).eval()


def test_real_config_resolves_to_sinegen2():
    """Guard against silently testing the wrong SineGen class again."""
    hift = _make_hift(REAL_CONFIG)
    assert isinstance(hift.m_source.l_sin_gen, SineGen2)

    hift_legacy = _make_hift(LEGACY_CONFIG)
    assert isinstance(hift_legacy.m_source.l_sin_gen, SineGen)


def _make_model(hift: CausalHiFTGenerator, window_len: int) -> CosyVoice3Code2Wav:
    """Build a CosyVoice3Code2Wav shell wired to the given HiFT."""
    model = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    model.hift = hift
    model._hift_window_len = window_len
    return model


def _chunks(total_mel: int = TOTAL_MEL, chunk_len: int = CHUNK_LEN) -> list[torch.Tensor]:
    torch.manual_seed(1)
    mel = torch.randn(1, 80, total_mel)
    return [mel[:, :, i : i + chunk_len] for i in range(0, total_mel, chunk_len)]


def _full_reference(model: CosyVoice3Code2Wav, chunks: list[torch.Tensor]) -> torch.Tensor:
    """Re-run HiFT over the full cumulative mel each chunk (pre-windowing behavior).

    Uses non-finalize inference per chunk, matching the old streaming vocoder; a
    single finalize pass would emit the look-right tail streaming holds back.
    """
    torch.manual_seed(0)  # align per-call phase_vec draws with _incremental
    emitted = []
    cache: dict[str, torch.Tensor] | None = None
    for chunk in chunks:
        cached = None if cache is None else cache.get("mel")
        if cached is not None and cached.numel() > 0:
            tts_mel = torch.cat([cached, chunk], dim=-1)
        else:
            tts_mel = chunk
        speech, _, _ = model.hift.inference(speech_feat=tts_mel, finalize=False)
        speech = speech.reshape(speech.shape[0], -1)
        offset = 0 if cache is None else cache.get("speech_offset")
        emitted.append(speech[:, offset:])
        cache = {"mel": tts_mel.detach().cpu(), "speech_offset": int(speech.shape[-1])}
    return torch.cat(emitted, dim=-1)


def _pcm16(x: torch.Tensor) -> torch.Tensor:
    """Quantize float audio to PCM16, the format the server streams to clients."""
    return (x.clamp(-1, 1) * 32767).round().to(torch.int16)


def _incremental(model: CosyVoice3Code2Wav, chunks: list[torch.Tensor]) -> torch.Tensor:
    """Run the new windowed path via _stream_hift_from_feat."""
    torch.manual_seed(0)
    emitted = []
    cache: dict[str, torch.Tensor] | None = None
    for chunk in chunks:
        speech, cache = model._stream_hift_from_feat(chunk, cache_state=cache, finalize=False)
        emitted.append(speech.reshape(speech.shape[0], -1))
    return torch.cat(emitted, dim=-1)


def _rel_divergence(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().mean() / a.abs().mean().clamp_min(1e-6)).item()


@pytest.mark.parametrize("config", CONFIGS)
def test_incremental_hift_matches_full_reference_exactly_with_full_history(config):
    """With a window covering the full history, the windowed path must match the
    full cumulative re-run exactly. This proves the phase/noise carry is right."""
    hift = _make_hift(config)
    model = _make_model(hift, window_len=TOTAL_MEL)  # window >= history
    chunks = _chunks()

    full = _full_reference(model, chunks)
    incr = _incremental(model, chunks)

    assert full.shape == incr.shape, f"{full.shape} vs {incr.shape}"
    assert _rel_divergence(full, incr) < 1e-6


@pytest.mark.parametrize("config", CONFIGS)
def test_incremental_hift_bounded_window_is_close(config):
    """With a bounded window, the windowed path stays close to the full re-run;
    only the truncated deep history diverges, and it is small."""
    hift = _make_hift(config)
    model = _make_model(hift, window_len=48)
    chunks = _chunks()

    full = _full_reference(model, chunks)
    incr = _incremental(model, chunks)

    assert full.shape == incr.shape
    torch.testing.assert_close(incr, full, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("config", CONFIGS)
def test_incremental_hift_matches_streaming_reference_within_tolerance(config):
    """Windowed output matches the full-cumulative reference within float32 rounding.

    The load-bearing correctness test: 400 mel frames far exceeds the 64-frame
    window, so truncation genuinely exercises the phase/noise carry. Float64
    agrees to ~1e-16, confirming the residual here is rounding, not truncation.
    """
    hift = _make_hift(config)
    model = _make_model(hift, window_len=64)  # the real _hift_window_len
    chunks = _chunks(total_mel=LONG_TOTAL_MEL)

    trim = int(hift.f0_predictor.condnet[0].causal_padding)
    assert LONG_TOTAL_MEL > 64 + trim + CHUNK_LEN  # window must actually truncate

    full = _full_reference(model, chunks)
    incr = _incremental(model, chunks)

    assert full.shape == incr.shape, f"{full.shape} vs {incr.shape}"
    torch.testing.assert_close(incr, full, atol=ATOL, rtol=RTOL)

    max_dev = (full - incr).abs().max().item()  # stays far below one PCM16 step
    assert max_dev < PCM16_LSB / 10, f"max deviation {max_dev:.3e} vs PCM16 LSB {PCM16_LSB:.3e}"


def test_incremental_hift_window_is_bounded():
    """Per-chunk window must not grow with cumulative mel history."""
    hift = _make_hift()
    model = _make_model(hift, window_len=32)
    chunks = _chunks(total_mel=200, chunk_len=24)

    trim = int(hift.f0_predictor.condnet[0].causal_padding)  # f0 predictor's own history requirement

    cache = None
    window_sizes = []
    for chunk in chunks:
        _, cache = model._stream_hift_from_feat(chunk, cache_state=cache, finalize=False)
        window_sizes.append(int(cache["mel"].shape[-1]))

    assert max(window_sizes) <= 32 + trim + 24  # bounded regardless of utterance length
    steady = max(window_sizes)
    assert window_sizes.count(steady) >= 2  # steady-state value repeats once history exceeds window


def test_incremental_hift_finalize_releases_tail():
    """Finalize must emit the released look-right tail without crashing."""
    hift = _make_hift()
    model = _make_model(hift, window_len=32)
    chunks = _chunks(total_mel=72, chunk_len=24)

    cache = None
    emitted = []
    for i, chunk in enumerate(chunks):
        finalize = i == len(chunks) - 1
        speech, cache = model._stream_hift_from_feat(chunk, cache_state=cache, finalize=finalize)
        emitted.append(speech.reshape(speech.shape[0], -1))
        if finalize:
            assert cache is None

    full = torch.cat(emitted, dim=-1)
    assert full.shape[-1] > 0
    assert torch.isfinite(full).all()


def test_incremental_hift_emits_full_audio_length():
    """Streamed output covers the full mel history at samples-per-mel resolution.

    Guards against emission arithmetic dropping an upsample stage, which
    silently truncates audio (e.g. 1/3 of expected length with [8,5,3]).
    """
    hift = _make_hift()
    model = _make_model(hift, window_len=32)
    chunks = _chunks(total_mel=72, chunk_len=24)

    cache = None
    emitted = []
    for i, chunk in enumerate(chunks):
        finalize = i == len(chunks) - 1
        speech, cache = model._stream_hift_from_feat(chunk, cache_state=cache, finalize=finalize)
        emitted.append(speech.reshape(speech.shape[0], -1))
        if finalize:
            assert cache is None

    full = torch.cat(emitted, dim=-1)
    # total_mel * SPM, +slack for finalize's look-right tail, -slack for a dropped upsample stage
    total_mel = sum(c.shape[-1] for c in chunks)
    assert full.shape[-1] > total_mel * SPM * 0.8, f"{full.shape[-1]} vs {total_mel * SPM}"
    assert full.shape[-1] < total_mel * SPM * 1.5, f"{full.shape[-1]} vs {total_mel * SPM}"
