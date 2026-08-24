# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the --pd-stage placement surface."""

from __future__ import annotations

import pytest

from sglang_omni.config import apply_pd_stage_overrides, parse_pd_stage_assignment
from sglang_omni.config.schema import EndpointsConfig, PipelineConfig
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from tests.unit_test.fixtures.pipeline_fakes import fake_factory_path
from tests.unit_test.pipeline.helpers import stage


@pytest.mark.parametrize(
    "value,expected",
    [
        ("thinker=0:1", ("thinker", 0, 1)),
        ("thinker=0,1:2,3", ("thinker", [0, 1], [2, 3])),
        (" thinker = 0 : 1 ", ("thinker", 0, 1)),
    ],
)
def test_parse_accepts_supported_forms(value, expected) -> None:
    assert parse_pd_stage_assignment(value) == expected


@pytest.mark.parametrize(
    "value,message",
    [
        ("thinker", "expected STAGE="),
        ("thinker=0", "expected STAGE="),
        ("thinker=:1", "expected STAGE="),
        ("=0:1", "expected STAGE="),
        ("thinker=a:1", "must be integers"),
        ("thinker=-1:1", "must be non-negative"),
        ("thinker=0:0,0", "repeat a device"),
    ],
)
def test_parse_rejects_malformed_input(value, message) -> None:
    with pytest.raises(ValueError, match=message):
        parse_pd_stage_assignment(value)


def _pipeline(tmp_path) -> PipelineConfig:
    return PipelineConfig(
        model_path="dummy",
        name="pd-cli",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        entry_stage="pre",
        stages=[
            stage("pre", next="thinker"),
            stage(
                "thinker",
                factory=fake_factory_path("pd_capable_factory"),
                next="post",
            ),
            stage("post", terminal=True),
        ],
    )


def test_override_compiles_into_prefill_and_decode_halves(tmp_path) -> None:
    config = apply_pd_stage_overrides(_pipeline(tmp_path), pd_stages=["thinker=1:2"])

    prep = prepare_pipeline_runtime(config)
    with prep.runtime_dir:
        names = [s.name for s in prep.stages_cfg]
        placement = {n: s.gpu_ids for n, s in prep.placement_plan.stages.items()}
        roles = {
            s.name: (s.pd_execution.role, s.pd_execution.partner)
            for s in prep.stages_cfg
            if s.pd_execution is not None
        }

    assert "thinker_prefill" in names and "thinker_decode" in names
    assert "thinker" not in names
    assert prep.name_map["thinker"] == "thinker_prefill"
    assert prep.terminal_name_map == {"thinker": "thinker_decode"}
    assert placement["thinker_prefill"] == (1,)
    assert placement["thinker_decode"] == (2,)
    assert roles == {
        "thinker_prefill": ("prefill", "thinker_decode"),
        "thinker_decode": ("decode", "thinker_prefill"),
    }


def test_no_override_leaves_the_pipeline_untouched(tmp_path) -> None:
    config = _pipeline(tmp_path)
    assert apply_pd_stage_overrides(config, pd_stages=None) is config

    prep = prepare_pipeline_runtime(config)
    with prep.runtime_dir:
        assert [s.name for s in prep.stages_cfg] == ["pre", "thinker", "post"]
        assert all(s.pd_execution is None for s in prep.stages_cfg)


def test_same_gpu_is_rejected_by_schema_validation(tmp_path) -> None:
    # model_copy does not re-enter model_post_init, so the override must re-run
    # _validate_pd itself; without that this placement would reach expansion.
    with pytest.raises(ValueError, match="cannot share the same GPU"):
        apply_pd_stage_overrides(_pipeline(tmp_path), pd_stages=["thinker=1:1"])


def test_unknown_stage_names_the_known_stages(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown stage 'talker'"):
        apply_pd_stage_overrides(_pipeline(tmp_path), pd_stages=["talker=0:1"])


def test_duplicate_assignment_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="multiple --pd-stage"):
        apply_pd_stage_overrides(
            _pipeline(tmp_path), pd_stages=["thinker=0:1", "thinker=2:3"]
        )


def test_pipeline_declared_pd_is_not_silently_overridden(tmp_path) -> None:
    config = apply_pd_stage_overrides(_pipeline(tmp_path), pd_stages=["thinker=1:2"])
    with pytest.raises(ValueError, match="already declares pd_disaggregation"):
        apply_pd_stage_overrides(config, pd_stages=["thinker=3:4"])
