# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS Base model support for sglang-omni."""

from sglang_omni.models.stage_capabilities import StageCapabilities

from . import config

CAPABILITIES = StageCapabilities(
    supports_cuda_graph=True,
    supports_torch_compile=True,
    supports_reference_audio=True,
)

__all__ = ["config", "CAPABILITIES"]
