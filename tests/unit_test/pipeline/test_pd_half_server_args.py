# SPDX-License-Identifier: Apache-2.0
"""Each PD half can carry its own SGLang server args."""

from __future__ import annotations

import pytest

from sglang_omni.config import expand_pd_stages
from sglang_omni.config.pd_rewrite import pd_required_factory_args
from sglang_omni.config.schema import PDConfig, PDStagePlacement, PipelineConfig
from tests.unit_test.pipeline.helpers import stage


def _halves(pd: PDConfig, **stage_kwargs):
    config = PipelineConfig(
        model_path="dummy",
        stages=[stage("thinker", terminal=True, pd_disaggregation=pd, **stage_kwargs)],
    )
    expansion = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )
    return {s.name: s for s in expansion.stages}


def _overrides(stage_config) -> dict:
    return dict(stage_config.factory_args.get("server_args_overrides") or {})


def test_each_half_receives_only_its_own_server_args() -> None:
    halves = _halves(
        PDConfig(
            prefill=PDStagePlacement(gpu=0, server_args={"chunked_prefill_size": 4096}),
            decode=PDStagePlacement(gpu=1, server_args={"max_running_requests": 32}),
        )
    )

    assert _overrides(halves["thinker_prefill"]) == {"chunked_prefill_size": 4096}
    assert _overrides(halves["thinker_decode"]) == {"max_running_requests": 32}


def test_a_half_value_wins_over_the_stage_value() -> None:
    """The half's args were written for that half, so they take precedence."""
    halves = _halves(
        PDConfig(
            prefill=PDStagePlacement(gpu=0, server_args={"max_running_requests": 8}),
            decode=PDStagePlacement(gpu=1),
        ),
        factory_args={"server_args_overrides": {"max_running_requests": 64}},
    )

    assert _overrides(halves["thinker_prefill"])["max_running_requests"] == 8
    # The decode half sets nothing, so it keeps the stage value.
    assert _overrides(halves["thinker_decode"])["max_running_requests"] == 64


def test_no_half_server_args_leaves_factory_args_untouched() -> None:
    original = {"server_args_overrides": {"max_running_requests": 64}}
    halves = _halves(
        PDConfig(prefill=PDStagePlacement(gpu=0), decode=PDStagePlacement(gpu=1)),
        factory_args=original,
    )

    assert halves["thinker_prefill"].factory_args == original
    assert halves["thinker_decode"].factory_args == original


def test_a_half_cannot_contradict_what_pd_requires() -> None:
    """Setting page_size on a half still fails, and as a configuration error."""
    halves = _halves(
        PDConfig(
            prefill=PDStagePlacement(gpu=0, server_args={"page_size": 16}),
            decode=PDStagePlacement(gpu=1),
        )
    )

    with pytest.raises(ValueError, match="requires page_size=1"):
        pd_required_factory_args(
            "thinker_prefill", halves["thinker_prefill"].factory_args
        )


def test_a_half_may_set_args_pd_does_not_constrain() -> None:
    halves = _halves(
        PDConfig(
            prefill=PDStagePlacement(gpu=0),
            decode=PDStagePlacement(gpu=1, server_args={"enable_mixed_chunk": False}),
        )
    )

    resolved = pd_required_factory_args(
        "thinker_decode", halves["thinker_decode"].factory_args
    )
    overrides = resolved["server_args_overrides"]

    assert overrides["enable_mixed_chunk"] is False
    assert overrides["page_size"] == 1
    assert overrides["disable_radix_cache"] is True
