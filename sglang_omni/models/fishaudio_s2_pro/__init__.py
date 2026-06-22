# SPDX-License-Identifier: Apache-2.0
"""FishAudio S2-Pro model support for sglang-omni."""

from sglang_omni.models.stage_capabilities import StageCapabilities

from . import config

CAPABILITIES = StageCapabilities(
    supports_cuda_graph=True,
    supports_torch_compile=True,
    supports_streaming_vocoder=True,
    supports_reference_audio=True,
)

__all__ = ["config", "CAPABILITIES"]
