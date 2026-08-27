# SPDX-License-Identifier: Apache-2.0
"""PD placement is addressed through the canonical config paths.

The two halves are declared under a stage's ``pd_disaggregation`` block, so
they are reachable by the same dotted paths every other per-stage setting
uses. That is what gives them precedence, duplicate detection, and
``config explain`` provenance without a flag of their own.
"""

from __future__ import annotations

import pytest

from sglang_omni.config.path import iter_schema_paths
from sglang_omni.config.schema import PipelineConfig


def _pd_paths() -> list[str]:
    return [p for p in iter_schema_paths(PipelineConfig) if "pd_disaggregation" in p]


@pytest.mark.parametrize(
    "path",
    [
        "stages.*.pd_disaggregation.prefill.gpu",
        "stages.*.pd_disaggregation.prefill.memory_fraction",
        "stages.*.pd_disaggregation.decode.gpu",
        "stages.*.pd_disaggregation.decode.memory_fraction",
    ],
)
def test_each_half_is_addressable(path: str) -> None:
    assert path in _pd_paths()


def test_the_per_half_engine_block_is_addressable() -> None:
    """A half may carry engine args the other does not."""
    paths = _pd_paths()

    assert "stages.*.pd_disaggregation.prefill.engine" in paths
    assert "stages.*.pd_disaggregation.decode.engine" in paths


def test_a_dotted_flag_writes_the_placement() -> None:
    """This is the surface that replaced a flag of PD's own."""
    from sglang_omni.config.manager import ConfigManager
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig

    manager = ConfigManager(config=Qwen3OmniPipelineConfig(model_path="dummy"))
    extra = manager.parse_extra_args(
        [
            "--thinker.pd_disaggregation.prefill.gpu",
            "0",
            "--thinker.pd_disaggregation.prefill.memory_fraction",
            "0.30",
            "--thinker.pd_disaggregation.decode.gpu",
            "0",
            "--thinker.pd_disaggregation.decode.memory_fraction",
            "0.62",
        ]
    )

    merged = manager.merge_config(extra)
    thinker = next(s for s in merged.stages if s.name == "thinker")

    assert thinker.pd_disaggregation is not None
    assert thinker.pd_disaggregation.prefill.gpu == 0
    assert thinker.pd_disaggregation.prefill.memory_fraction == 0.30
    assert thinker.pd_disaggregation.decode.gpu == 0
    assert thinker.pd_disaggregation.decode.memory_fraction == 0.62
