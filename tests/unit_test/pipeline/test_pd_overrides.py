# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the --pd-stage placement surface."""

from __future__ import annotations

from typing import ClassVar

import pytest

from sglang_omni.config import apply_pd_stage_overrides, parse_pd_stage_assignment
from sglang_omni.config.pd_capability import apply_pd_required_server_args
from sglang_omni.config.pd_rewrite import expand_pd_stages
from sglang_omni.config.schema import (
    EndpointsConfig,
    PDConfig,
    PDStagePlacement,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.pipeline.runtime_config import prepare_pipeline_runtime
from tests.unit_test.fixtures.pipeline_fakes import fake_factory_path
from tests.unit_test.pipeline.helpers import stage


class _EngineStageConfig(StageConfig):
    """A stage type that may carry an ``engine`` block, as a real one does."""

    engine_stage: ClassVar[bool] = True


def _engine_stage(name: str, **kwargs):
    kwargs.setdefault("factory_path", fake_factory_path("pd_capable_factory"))
    kwargs.setdefault("process", "pipeline")
    return _EngineStageConfig(name=name, **kwargs)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("thinker=0:1", ("thinker", 0, 1, None, None)),
        ("thinker=0,1:2,3", ("thinker", [0, 1], [2, 3], None, None)),
        (" thinker = 0 : 1 ", ("thinker", 0, 1, None, None)),
        ("thinker=0@0.25:0@0.65", ("thinker", 0, 0, 0.25, 0.65)),
        ("thinker=0@0.3:1", ("thinker", 0, 1, 0.3, None)),
    ],
)
def test_parse_accepts_supported_forms(value, expected) -> None:
    parsed = parse_pd_stage_assignment(value)

    assert (
        parsed.stage_name,
        parsed.prefill_gpu,
        parsed.decode_gpu,
        parsed.prefill_fraction,
        parsed.decode_fraction,
    ) == expected


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
                factory_path=fake_factory_path("pd_capable_factory"),
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


def test_the_flag_can_place_both_halves_on_one_gpu(tmp_path) -> None:
    """Sharing a card is a placement choice, and the shares go with it."""
    config = apply_pd_stage_overrides(
        _pipeline(tmp_path), pd_stages=["thinker=1@0.3:1@0.6"]
    )
    pd = {s.name: s for s in config.stages}["thinker"].pd_disaggregation

    assert pd.prefill.gpu == 1
    assert pd.decode.gpu == 1
    assert (pd.prefill.memory_fraction, pd.decode.memory_fraction) == (0.3, 0.6)


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


# --- PD-required server args -------------------------------------------------


def _pd(prefill_gpu: int, decode_gpu: int) -> PDConfig:
    return PDConfig(
        prefill=PDStagePlacement(gpu=prefill_gpu),
        decode=PDStagePlacement(gpu=decode_gpu),
    )


def _pd_stages(tmp_path, *, factory: str, server_args=None):
    engine = dict(server_args) if server_args else None
    config = PipelineConfig(
        model_path="dummy",
        name="pd",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        entry_stage="thinker",
        stages=[
            _engine_stage(
                "thinker",
                factory_path=factory,
                terminal=True,
                engine=engine,
                pd_disaggregation=_pd(1, 2),
            )
        ],
    )
    prep = prepare_pipeline_runtime(config)
    with prep.runtime_dir:
        return {s.name: s for s in prep.stages_cfg}


def test_pd_halves_receive_the_server_args_pd_requires(tmp_path) -> None:
    from sglang_omni.config.pd_rewrite import PD_REQUIRED_SERVER_ARGS

    stages = _pd_stages(tmp_path, factory=fake_factory_path("pd_capable_factory"))

    for name in ("thinker_prefill", "thinker_decode"):
        overrides = stages[name].engine.overrides()
        for key, required in PD_REQUIRED_SERVER_ARGS.items():
            assert overrides[key] == required, (name, key)


def test_pd_injection_keeps_unrelated_server_args(tmp_path) -> None:
    stages = _pd_stages(
        tmp_path,
        factory=fake_factory_path("pd_capable_factory"),
        server_args={"enable_mixed_chunk": False},
    )

    overrides = stages["thinker_decode"].engine.overrides()
    assert overrides["enable_mixed_chunk"] is False
    assert overrides["disable_radix_cache"] is True
    assert overrides["page_size"] == 1


def test_pd_injection_rejects_a_contradicting_server_arg(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires engine.page_size=1"):
        _pd_stages(
            tmp_path,
            factory=fake_factory_path("pd_capable_factory"),
            server_args={"page_size": 16},
        )


def test_a_non_engine_stage_gets_no_engine_block(tmp_path) -> None:
    """The engine block only exists on stage types that declare engine_stage."""
    config = PipelineConfig(
        model_path="dummy",
        name="pd",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        entry_stage="thinker",
        stages=[
            stage(
                "thinker",
                factory_path=fake_factory_path("pd_capable_factory"),
                terminal=True,
                pd_disaggregation=PDConfig(
                    prefill=PDStagePlacement(gpu=0),
                    decode=PDStagePlacement(gpu=1),
                ),
            )
        ],
    )
    halves = {
        s.name: s
        for s in apply_pd_required_server_args(
            expand_pd_stages(
                list(config.stages), entry_stage=config.resolved_entry_stage
            ).stages
        )
    }

    for name in ("thinker_prefill", "thinker_decode"):
        assert halves[name].engine is None


def test_non_pd_pipeline_gets_no_server_args_injected(tmp_path) -> None:
    config = PipelineConfig(
        model_path="dummy",
        name="plain",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        stages=[stage("a", next="b"), stage("b", terminal=True)],
    )
    prep = prepare_pipeline_runtime(config)
    with prep.runtime_dir:
        for unchanged in prep.stages_cfg:
            assert (
                unchanged.engine is None
                or "page_size" not in unchanged.engine.overrides()
            )


def test_a_half_share_reaches_the_placement(tmp_path) -> None:
    """`thinker=0@0.25:0@0.65` is how a shared card is expressed on the CLI."""
    config = apply_pd_stage_overrides(
        _pipeline(tmp_path), pd_stages=["thinker=0@0.25:0@0.65"]
    )
    pd = {s.name: s for s in config.stages}["thinker"].pd_disaggregation

    assert pd.prefill.memory_fraction == 0.25
    assert pd.decode.memory_fraction == 0.65


def test_a_share_is_optional(tmp_path) -> None:
    config = apply_pd_stage_overrides(_pipeline(tmp_path), pd_stages=["thinker=0:1"])
    pd = {s.name: s for s in config.stages}["thinker"].pd_disaggregation

    assert pd.prefill.memory_fraction is None
    assert pd.decode.memory_fraction is None


def test_a_share_outside_the_unit_interval_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="must be in"):
        apply_pd_stage_overrides(_pipeline(tmp_path), pd_stages=["thinker=0@1.5:1"])


def test_a_share_that_is_not_a_number_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="is not a number"):
        apply_pd_stage_overrides(_pipeline(tmp_path), pd_stages=["thinker=0@half:1"])
