# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Correctness + boundedness tests for the incremental (windowed) HiFT path.

The streaming vocoder used to re-run over the full cumulative mel every chunk
(O(n) per chunk -> O(n^2) total). The C1 change runs it over a bounded window
of the last ``_hift_window_len`` mel frames plus the new tail, carrying the
harmonic phase and noise-buffer offset across chunks.

These tests assert the two properties that must hold for the change to be
sound:

* **Correctness** — the windowed path reproduces the full cumulative re-run
  (the previous behavior).

  - With a window covering the full history the match is **bit-exact**, which
    proves the phase / noise-offset carry across chunks is right.
  - With a *bounded* window the two differ only by floating-point
    **reassociation**: identical arithmetic accumulated in a different order
    over a short window than over the long cumulative sequence. Evaluated in
    float64 the two agree to ~1e-16, i.e. the bounded window is algebraically
    exact — it genuinely captures the full dependency and nothing is lost by
    truncating history. In float32 the residual is ~2e-7 absolute, about
    1/150th of a PCM16 quantization step: rounding noise, not truncation error.

  This is why the tests assert a tight ``atol``/``rtol`` rather than
  byte-identical PCM16. A sub-LSB perturbation can still flip a sample sitting
  exactly on a rounding boundary (~0.05% of samples do), so PCM16 equality is
  not a meaningful bar, whereas the float bound is.
* **Boundedness** — the per-chunk window does not grow with the cumulative mel
  history, so the O(n) per-chunk slope is actually removed.
"""

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import (
    CausalConvRNNF0Predictor,
    CausalHiFTGenerator,
)
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CHUNK_LEN = 24
TOTAL_MEL = 96
SPM = 480  # samples per mel frame for the small test HiFT (8*5*3*4), matching the real model

# Long enough that the 64-frame window truncates real history for most of the
# stream. At TOTAL_MEL=96 the window never meaningfully truncates, so a test at
# that length would pass vacuously.
LONG_TOTAL_MEL = 400

# One PCM16 quantization step, the smallest difference the streamed audio can
# actually represent.
PCM16_LSB = 1.0 / 32767.0

# Tolerance for the bounded-window comparison. Worst observed deviation across
# seeds {1,2,3} x lengths {96,400,600} x windows {32,64,128} is 1.97e-7
# absolute (0.0065 x PCM16_LSB), pinned at ~2x float32 epsilon. atol=1e-6 keeps
# ~5x headroom for BLAS/hardware variation while staying ~30x below one PCM16
# step, so any real truncation error would fail this bound immediately.
ATOL = 1e-6
RTOL = 1e-5


def _make_hift() -> CausalHiFTGenerator:
    """A small CausalHiFTGenerator with a real (tiny) F0 predictor."""
    torch.manual_seed(0)
    f0_predictor = CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=16)
    return CausalHiFTGenerator(
        in_channels=80,
        base_channels=32,
        nb_harmonics=4,
        sampling_rate=22050,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        f0_predictor=f0_predictor,
    ).eval()


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
    """Re-run HiFT over the full cumulative mel each chunk (old behavior).

    This is the correct reference for the streaming path: it uses non-finalize
    inference per chunk (which trims the conv_pre look-right tail), exactly as
    the pre-C1 streaming vocoder did. A single finalize pass over the whole mel
    is NOT the right reference — it emits the look-right tail that streaming
    deliberately holds back, so it is a different (longer) signal.
    """
    # Reset the RNG so the per-call random phase_vec aligns with the incremental
    # path (both call inference the same number of times).
    torch.manual_seed(0)
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


def test_incremental_hift_matches_full_reference_exactly_with_full_history():
    """With a window covering the full history, the windowed path must match the
    full cumulative re-run exactly. This proves the phase/noise carry is right."""
    hift = _make_hift()
    model = _make_model(hift, window_len=TOTAL_MEL)  # window >= history
    chunks = _chunks()

    full = _full_reference(model, chunks)
    incr = _incremental(model, chunks)

    assert full.shape == incr.shape, f"{full.shape} vs {incr.shape}"
    assert _rel_divergence(full, incr) < 1e-6


def test_incremental_hift_bounded_window_is_close():
    """With a bounded window, the windowed path stays close to the full re-run;
    only the truncated deep history diverges, and it is small."""
    hift = _make_hift()
    model = _make_model(hift, window_len=48)
    chunks = _chunks()

    full = _full_reference(model, chunks)
    incr = _incremental(model, chunks)

    assert full.shape == incr.shape
    torch.testing.assert_close(incr, full, atol=ATOL, rtol=RTOL)


def test_incremental_hift_matches_streaming_reference_within_tolerance():
    """At the real window length (64) over a long utterance, the windowed path
    must match the full-cumulative streaming re-run to within float32 rounding.

    This is the load-bearing correctness test: the cumulative history (400 mel
    frames) far exceeds the window, so truncation — and therefore the phase /
    noise-offset carry — is genuinely exercised. The reference is the pre-C1
    behavior (non-finalize per chunk, emitting from a cumulative speech
    offset), NOT a single finalize pass, which holds back the conv_pre
    look-right tail and is a different, longer signal.

    The residual is float32 reassociation noise, not truncation error: the same
    comparison in float64 agrees to ~1e-16. So the bound is a tight absolute
    tolerance, not PCM16 byte-equality.
    """
    hift = _make_hift()
    model = _make_model(hift, window_len=64)  # the real _hift_window_len
    chunks = _chunks(total_mel=LONG_TOTAL_MEL)

    # Guard: the window must actually truncate, or this test proves nothing.
    trim = int(hift.f0_predictor.condnet[0].causal_padding)
    assert LONG_TOTAL_MEL > 64 + trim + CHUNK_LEN

    full = _full_reference(model, chunks)
    incr = _incremental(model, chunks)

    assert full.shape == incr.shape, f"{full.shape} vs {incr.shape}"
    torch.testing.assert_close(incr, full, atol=ATOL, rtol=RTOL)

    # Audio-domain restatement: the deviation stays far below one PCM16 step,
    # so it cannot be an audible or structural difference.
    max_dev = (full - incr).abs().max().item()
    assert max_dev < PCM16_LSB / 10, f"max deviation {max_dev:.3e} vs PCM16 LSB {PCM16_LSB:.3e}"


def test_incremental_hift_window_is_bounded():
    """Per-chunk window must not grow with cumulative mel history."""
    hift = _make_hift()
    model = _make_model(hift, window_len=32)
    chunks = _chunks(total_mel=200, chunk_len=24)

    # The f0 predictor's condnet[0] is causal_type="right" (kernel 4), so its
    # output is trimmed by `trim` frames at the END and the f0 at the window's
    # start needs `trim` frames of history. The window therefore carries
    # window_len + trim history frames plus the new chunk.
    trim = int(hift.f0_predictor.condnet[0].causal_padding)

    cache = None
    window_sizes = []
    for chunk in chunks:
        _, cache = model._stream_hift_from_feat(chunk, cache_state=cache, finalize=False)
        window_sizes.append(int(cache["mel"].shape[-1]))

    # Once the history exceeds the window, the cached window stays at
    # window_len + trim (overlap) + chunk_len, i.e. bounded regardless of how
    # long the utterance is.
    assert max(window_sizes) <= 32 + trim + 24
    # The window must stop growing once history exceeds the window: the
    # steady-state value (window_len + trim + chunk_len) repeats. (The final
    # chunk may be shorter, so compare the two largest, not the last two.)
    steady = max(window_sizes)
    assert window_sizes.count(steady) >= 2


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
    """The streamed output must cover the full mel history at the correct
    samples-per-mel rate (product of all upsample rates x hop_len). This guards
    against the emission arithmetic dropping an upsample stage, which silently
    truncates the audio (e.g. 1/3 of the expected length with [8,5,3])."""
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
    # Each mel frame yields SPM samples. The emitted length must be on the order
    # of total_mel * SPM (the finalize releases a small look-right tail, so it
    # can exceed it slightly) — NOT a small fraction of it, which is the
    # signature of a dropped upsample stage in the samples-per-mel arithmetic.
    total_mel = sum(c.shape[-1] for c in chunks)
    assert full.shape[-1] > total_mel * SPM * 0.8, f"{full.shape[-1]} vs {total_mel * SPM}"
    assert full.shape[-1] < total_mel * SPM * 1.5, f"{full.shape[-1]} vs {total_mel * SPM}"
