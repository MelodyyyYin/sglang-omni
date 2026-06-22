# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local (v1.5) pipeline package."""

from sglang_omni.models.stage_capabilities import StageCapabilities

CAPABILITIES = StageCapabilities(
    supports_cuda_graph=True,
    supports_async_decode=True,
    supports_torch_compile=True,
    supports_streaming_vocoder=True,
    supports_reference_audio=True,
)

__all__ = ["CAPABILITIES"]
