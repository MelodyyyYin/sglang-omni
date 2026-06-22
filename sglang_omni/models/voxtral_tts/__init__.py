"""Voxtral-4B-TTS model support for sglang-omni."""

from sglang_omni.models.stage_capabilities import StageCapabilities

from . import config

CAPABILITIES = StageCapabilities(
    supports_cuda_graph=True,
    supports_torch_compile=True,
)

__all__ = ["config", "CAPABILITIES"]
