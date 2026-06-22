# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the T6 stage capability vocabulary (RFC #661).

CPU-only. A model whose optional runtime deps are missing in this environment
is skipped (the codebase tolerates missing per-model deps, cf. PR #680); the
presence guarantee is enforced in the full-dependency model CI.
"""

import dataclasses
import importlib

import pytest

from sglang_omni.models.stage_capabilities import (
    StageCapabilities,
    collect_capabilities,
    format_capability_table,
)

# In-scope TTS models that must declare CAPABILITIES.
TTS_MODELS = [
    "higgs_tts",
    "moss_tts",
    "moss_tts_local",
    "qwen3_tts",
    "fishaudio_s2_pro",
    "voxtral_tts",
]


def _try_import(model):
    try:
        return importlib.import_module(f"sglang_omni.models.{model}"), None
    except Exception as exc:  # optional dep missing in this env
        return None, exc


@pytest.mark.parametrize("model", TTS_MODELS)
def test_model_declares_capabilities(model):
    module, exc = _try_import(model)
    if module is None:
        pytest.skip(f"{model} cannot be imported in this env: {exc}")
    caps = getattr(module, "CAPABILITIES", None)
    assert caps is not None, f"{model} must export a CAPABILITIES constant"
    assert isinstance(caps, StageCapabilities)


def test_defaults_all_false():
    caps = StageCapabilities()
    for f in dataclasses.fields(StageCapabilities):
        assert getattr(caps, f.name) is False, f"{f.name} should default to False"


def test_capabilities_is_frozen():
    caps = StageCapabilities()
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.supports_cuda_graph = True


def test_collect_capabilities_covers_importable_models():
    table = collect_capabilities()
    assert table, "no model declared CAPABILITIES"
    for model in TTS_MODELS:
        module, _ = _try_import(model)
        if module is None:
            continue  # optional dep missing in this env; enforced in full CI
        assert model in table, f"{model} imported but missing from collect_capabilities()"
    rendered = format_capability_table(table)
    assert "model" in rendered and "cuda_graph" in rendered
