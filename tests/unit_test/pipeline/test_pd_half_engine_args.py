# SPDX-License-Identifier: Apache-2.0
"""Each PD half can carry its own SGLang server args."""

from __future__ import annotations

from typing import ClassVar

import pytest

from sglang_omni.config import expand_pd_stages
from sglang_omni.config.pd_capability import apply_pd_required_engine_args
from sglang_omni.config.schema import (
    PDConfig,
    PDStagePlacement,
    PipelineConfig,
    StageConfig,
)
from tests.unit_test.pipeline.helpers import FACTORY


class _EngineStageConfig(StageConfig):
    """A stage type that may carry an ``engine`` block, as a real one does."""

    engine_stage: ClassVar[bool] = True


def _engine_stage(name: str, **kwargs):
    kwargs.setdefault("factory_path", FACTORY)
    kwargs.setdefault("process", "pipeline")
    return _EngineStageConfig(name=name, **kwargs)


def _halves(pd: PDConfig, **stage_kwargs):
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            _engine_stage(
                "thinker", terminal=True, pd_disaggregation=pd, **stage_kwargs
            )
        ],
    )
    expansion = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )
    return {s.name: s for s in expansion.stages}


def _overrides(stage_config) -> dict:
    return stage_config.engine.overrides() if stage_config.engine else {}


def _required(halves, name):
    """Run the pass that sets what PD requires, and return that half."""
    applied = apply_pd_required_engine_args(halves.values())
    return {s.name: s for s in applied}[name]


def test_each_half_receives_only_its_own_engine_args() -> None:
    halves = _halves(
        PDConfig(
            prefill=PDStagePlacement(gpu=0, engine={"chunked_prefill_size": 4096}),
            decode=PDStagePlacement(gpu=1, engine={"max_running_requests": 32}),
        )
    )

    assert _overrides(halves["thinker_prefill"]) == {"chunked_prefill_size": 4096}
    assert _overrides(halves["thinker_decode"]) == {"max_running_requests": 32}


def test_a_half_value_wins_over_the_stage_value() -> None:
    """The half's args were written for that half, so they take precedence."""
    halves = _halves(
        PDConfig(
            prefill=PDStagePlacement(gpu=0, engine={"max_running_requests": 8}),
            decode=PDStagePlacement(gpu=1),
        ),
        engine={"max_running_requests": 64},
    )

    assert _overrides(halves["thinker_prefill"])["max_running_requests"] == 8
    # The decode half sets nothing, so it keeps the stage value.
    assert _overrides(halves["thinker_decode"])["max_running_requests"] == 64


def test_no_half_engine_args_leaves_the_stage_engine_untouched() -> None:
    original = {"max_running_requests": 64}
    halves = _halves(
        PDConfig(prefill=PDStagePlacement(gpu=0), decode=PDStagePlacement(gpu=1)),
        engine=original,
    )

    assert _overrides(halves["thinker_prefill"]) == original
    assert _overrides(halves["thinker_decode"]) == original


def test_a_half_cannot_contradict_what_pd_requires() -> None:
    """Setting page_size on a half still fails, and as a configuration error."""
    halves = _halves(
        PDConfig(
            prefill=PDStagePlacement(gpu=0, engine={"page_size": 16}),
            decode=PDStagePlacement(gpu=1),
        )
    )

    with pytest.raises(ValueError, match="requires engine.page_size=1"):
        apply_pd_required_engine_args(halves.values())


def test_a_half_may_set_args_pd_does_not_constrain() -> None:
    halves = _halves(
        PDConfig(
            prefill=PDStagePlacement(gpu=0),
            decode=PDStagePlacement(gpu=1, engine={"enable_mixed_chunk": False}),
        )
    )

    overrides = _overrides(_required(halves, "thinker_decode"))

    assert overrides["enable_mixed_chunk"] is False
    assert overrides["page_size"] == 1
    assert overrides["disable_radix_cache"] is True
