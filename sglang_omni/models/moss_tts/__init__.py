# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS support for SGLang Omni."""

from sglang_omni.models.stage_capabilities import StageCapabilities

CAPABILITIES = StageCapabilities(
    supports_cuda_graph=True,
    supports_reference_audio=True,
)

__all__ = ["CAPABILITIES"]
