# SPDX-License-Identifier: Apache-2.0
"""Streaming entry point for the ZONOS2 DAC vocoder.

Thin wrapper that decodes the full delayed code sequence in one shot;
incremental overlap-add decode is a later optimization.
"""

from __future__ import annotations

import torch

from .audio_codec import Zonos2DACVocoder

# Process-wide cache: load the DAC checkpoint at most once per device.
_vocoder: Zonos2DACVocoder | None = None
_vocoder_device: str | None = None


def _get_vocoder(device: str) -> Zonos2DACVocoder:
    global _vocoder, _vocoder_device
    if _vocoder is None or _vocoder_device != device:
        _vocoder = Zonos2DACVocoder(device=device)
        _vocoder_device = device
    return _vocoder


def decode_to_pcm(
    audio_codes: torch.Tensor,
    eos_frame: int | None = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Decode delayed ``[T, 9]`` AR codes to 1-D float32 PCM @ 44.1 kHz.

    Args:
        audio_codes: delayed per-frame codes ``[T, 9]`` (or batched ``[B, T, 9]``)
            straight from AR decode, before the delay is sheared out.
        eos_frame: number of aligned frames to keep before EOS, if known.
        device: torch device the DAC model runs on.

    Returns:
        1-D ``float32`` PCM tensor at 44.1 kHz on CPU.
    """
    return _get_vocoder(device).decode(audio_codes, eos_frame=eos_frame)
