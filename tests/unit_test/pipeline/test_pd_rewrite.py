# SPDX-License-Identifier: Apache-2.0
"""Note: (Yue Yin) Structural tests for the PD graph rewrite.

These cover only the compiler graph rewrite plus its effect on placement and
process topology. No scheduler/KV/request wiring is exercised.
"""
from __future__ import annotations

import pytest

from sglang_omni.config import (
    PipelineConfig,
    StageConfig,
    build_process_topology_plan,
    build_stage_placement_plan,
    expand_pd_stages,
)
from sglang_omni.config.schema import PDConfig, PDStagePlacement

_FACTORY = "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"


def _pd(prefill_gpu: int, decode_gpu: int) -> PDConfig:
    return PDConfig(
        prefill=PDStagePlacement(gpu=prefill_gpu),
        decode=PDStagePlacement(gpu=decode_gpu),
    )


def _expand(config: PipelineConfig) -> list[StageConfig]:
    return expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    ).stages


def _by_name(stages: list[StageConfig]) -> dict[str, StageConfig]:
    return {s.name: s for s in stages}


def _linear_pd_pipeline() -> PipelineConfig:
    return PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="pre", factory=_FACTORY, gpu=0, process="pre", next="thinker"
            ),
            StageConfig(
                name="thinker",
                factory=_FACTORY,
                gpu=1,
                process="thinker",
                next="post",
                pd_disaggregation=_pd(1, 2),
            ),
            StageConfig(
                name="post", factory=_FACTORY, gpu=3, process="post", terminal=True
            ),
        ],
    )


def test_expand_splits_stage_and_preserves_order() -> None:
    stages = _expand(_linear_pd_pipeline())

    assert [s.name for s in stages] == [
        "pre",
        "thinker_prefill",
        "thinker_decode",
        "post",
    ]


def test_inbound_edge_terminates_at_prefill() -> None:
    stages = _by_name(_expand(_linear_pd_pipeline()))

    assert stages["pre"].next == "thinker_prefill"


def test_downstream_edge_originates_from_decode() -> None:
    stages = _by_name(_expand(_linear_pd_pipeline()))

    prefill = stages["thinker_prefill"]
    decode = stages["thinker_decode"]

    # Note: (Yue Yin) Prefill owns no ordinary downstream payload edge.
    assert prefill.next is None
    assert prefill.terminal is False
    assert prefill.route_fn is None
    assert prefill.stream_to == []

    # Note: (Yue Yin) Decode owns the logical stage's real downstream edge.
    assert decode.next == "post"


def test_pd_placement_and_roles_recorded() -> None:
    stages = _by_name(_expand(_linear_pd_pipeline()))

    prefill = stages["thinker_prefill"]
    decode = stages["thinker_decode"]

    assert prefill.gpu == 1
    assert prefill.process == "thinker_prefill"
    assert prefill.pd_execution is not None
    assert prefill.pd_execution.role == "prefill"
    assert prefill.pd_execution.partner == "thinker_decode"

    assert decode.gpu == 2
    assert decode.process == "thinker_decode"
    assert decode.pd_execution is not None
    assert decode.pd_execution.role == "decode"
    assert decode.pd_execution.partner == "thinker_prefill"

    # Note: (Yue Yin) PD metadata is typed, never smuggled through factory_args
    # (which would leak into the factory constructor).
    assert "pd_role" not in prefill.factory_args
    assert "pd_partner" not in prefill.factory_args
    assert "pd_role" not in decode.factory_args
    assert "pd_partner" not in decode.factory_args

    # Note: (Yue Yin) Partner identity is symmetric.
    assert prefill.pd_execution.partner == decode.name
    assert decode.pd_execution.partner == prefill.name

    # Note: (Yue Yin) The handoff is a PD control/KV relationship, not a next
    # payload edge, so no stage routes to the decode half.
    assert all(s.next != "thinker_decode" for s in stages.values())


def test_placement_puts_halves_on_distinct_gpus() -> None:
    config = _linear_pd_pipeline()
    stages = _expand(config)

    plan = build_stage_placement_plan(config, stages_cfg=stages)

    assert plan.stages["thinker_prefill"].gpu_ids == (1,)
    assert plan.stages["thinker_decode"].gpu_ids == (2,)


def test_topology_puts_halves_in_distinct_processes() -> None:
    config = _linear_pd_pipeline()
    stages = _expand(config)
    plan = build_stage_placement_plan(config, stages_cfg=stages)

    topology = build_process_topology_plan(config, plan, stages_cfg=stages)

    assert topology.stage_to_process["thinker_prefill"] == "thinker_prefill"
    assert topology.stage_to_process["thinker_decode"] == "thinker_decode"
    assert (
        topology.stage_to_process["thinker_prefill"]
        != topology.stage_to_process["thinker_decode"]
    )


def test_entry_stage_rewritten_to_prefill_when_pd_is_entry() -> None:
    config = PipelineConfig(
        model_path="dummy",
        entry_stage="thinker",
        stages=[
            StageConfig(
                name="thinker",
                factory=_FACTORY,
                gpu=1,
                process="thinker",
                next="post",
                pd_disaggregation=_pd(1, 2),
            ),
            StageConfig(
                name="post", factory=_FACTORY, gpu=0, process="post", terminal=True
            ),
        ],
    )

    expansion = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )

    assert expansion.entry_stage == "thinker_prefill"
    assert [s.name for s in expansion.stages][:2] == [
        "thinker_prefill",
        "thinker_decode",
    ]
    # Note: (Yue Yin) Routing map sends the logical entry name to the prefill
    # half; terminal map is empty because the PD stage is not terminal.
    assert expansion.routing_map["thinker"] == "thinker_prefill"
    assert expansion.terminal_map["thinker"] == "thinker_decode"


def test_fan_in_stays_on_prefill_and_decode_is_cleared() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(name="a", factory=_FACTORY, gpu=0, process="a", next="thinker"),
            StageConfig(name="b", factory=_FACTORY, gpu=3, process="b", next="thinker"),
            StageConfig(
                name="thinker",
                factory=_FACTORY,
                gpu=1,
                process="thinker",
                next="post",
                wait_for=["a", "b"],
                merge_fn="tests.unit_test.fixtures.pipeline_fakes.merge_payloads",
                pd_disaggregation=_pd(1, 2),
            ),
            StageConfig(
                name="post", factory=_FACTORY, gpu=0, process="post", terminal=True
            ),
        ],
    )

    stages = _by_name(_expand(config))

    # Note: (Yue Yin) Inbound producers now target the prefill half.
    assert stages["a"].next == "thinker_prefill"
    assert stages["b"].next == "thinker_prefill"

    # Note: (Yue Yin) Fan-in stays where inputs arrive (prefill); decode
    # admits a prebuilt request and carries no fan-in.
    prefill = stages["thinker_prefill"]
    decode = stages["thinker_decode"]
    assert prefill.wait_for == ["a", "b"]
    assert prefill.merge_fn is not None
    assert decode.wait_for is None
    assert decode.merge_fn is None


def test_stream_to_moves_to_decode() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(name="pre", factory=_FACTORY, gpu=0, process="pre", next="thinker"),
            StageConfig(
                name="thinker",
                factory=_FACTORY,
                gpu=1,
                process="thinker",
                next="post",
                stream_to=["sink"],
                pd_disaggregation=_pd(1, 2),
            ),
            StageConfig(name="post", factory=_FACTORY, gpu=0, process="post", terminal=True),
            StageConfig(name="sink", factory=_FACTORY, gpu=0, process="sink", terminal=True),
        ],
    )

    stages = _by_name(_expand(config))

    assert stages["thinker_prefill"].stream_to == []
    assert stages["thinker_decode"].stream_to == ["sink"]


def test_pipeline_without_pd_is_unchanged() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(name="pre", factory=_FACTORY, gpu=0, process="pre", next="post"),
            StageConfig(name="post", factory=_FACTORY, gpu=0, process="post", terminal=True),
        ],
    )

    expansion = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )

    assert [s.name for s in expansion.stages] == ["pre", "post"]
    assert expansion.entry_stage == "pre"
    assert expansion.routing_map == {}
    assert expansion.terminal_map == {}


def test_validation_rejects_shared_gpu() -> None:
    with pytest.raises(ValueError, match="cannot share the same GPU"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                StageConfig(
                    name="thinker",
                    factory=_FACTORY,
                    gpu=1,
                    process="thinker",
                    terminal=True,
                    pd_disaggregation=_pd(1, 1),
                )
            ],
        )


def test_validation_requires_explicit_gpus() -> None:
    with pytest.raises(ValueError, match="requires explicit"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                StageConfig(
                    name="thinker",
                    factory=_FACTORY,
                    gpu=1,
                    process="thinker",
                    terminal=True,
                    pd_disaggregation=PDConfig(
                        prefill=PDStagePlacement(gpu=1),
                        decode=PDStagePlacement(gpu=None),
                    ),
                )
            ],
        )


def test_validation_rejects_pd_in_fused_group() -> None:
    with pytest.raises(ValueError, match="cannot set pd_disaggregation"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                StageConfig(
                    name="thinker",
                    factory=_FACTORY,
                    gpu=1,
                    process="pipeline",
                    next="post",
                    pd_disaggregation=_pd(1, 2),
                ),
                StageConfig(
                    name="post",
                    factory=_FACTORY,
                    gpu=1,
                    process="pipeline",
                    terminal=True,
                ),
            ],
            fused_stages=[["thinker", "post"]],
        )


def _terminal_pd_pipeline() -> PipelineConfig:
    # Note: (Yue Yin) The PD stage is itself the terminal stage.
    return PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="pre", factory=_FACTORY, gpu=0, process="pre", next="thinker"
            ),
            StageConfig(
                name="thinker",
                factory=_FACTORY,
                gpu=1,
                process="thinker",
                terminal=True,
                pd_disaggregation=_pd(1, 2),
            ),
        ],
    )


def test_terminal_flag_moves_to_decode_only() -> None:
    stages = _by_name(_expand(_terminal_pd_pipeline()))

    # Note: (Yue Yin) Only the decode half is terminal; prefill is not.
    assert stages["thinker_decode"].terminal is True
    assert stages["thinker_prefill"].terminal is False


def test_terminal_map_points_terminal_stage_to_decode() -> None:
    config = _terminal_pd_pipeline()
    expansion = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )

    assert expansion.terminal_map["thinker"] == "thinker_decode"
    physical_terminal = [s.name for s in expansion.stages if s.terminal]
    assert physical_terminal == ["thinker_decode"]


def test_expand_twice_is_idempotent() -> None:
    config = _linear_pd_pipeline()
    once = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )
    twice = expand_pd_stages(once.stages, entry_stage=once.entry_stage)

    # Note: (Yue Yin) The generated halves clear pd_disaggregation, so a second
    # expansion is a safe no-op with no further renaming.
    assert [s.name for s in twice.stages] == [s.name for s in once.stages]
    assert twice.entry_stage == once.entry_stage
    assert twice.routing_map == {}
    assert twice.terminal_map == {}


def test_prefill_has_no_next_edge_to_decode() -> None:
    stages = _by_name(_expand(_linear_pd_pipeline()))
    prefill = stages["thinker_prefill"]

    # Note: (Yue Yin) The prefill->decode handoff is never an ordinary next edge.
    assert prefill.next is None
    assert all(s.next != "thinker_decode" for s in stages.values())


def test_validation_rejects_generated_name_collision() -> None:
    with pytest.raises(ValueError, match="collides with an existing stage"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                StageConfig(
                    name="thinker",
                    factory=_FACTORY,
                    gpu=1,
                    process="thinker",
                    next="thinker_decode",
                    pd_disaggregation=_pd(1, 2),
                ),
                StageConfig(
                    name="thinker_decode",
                    factory=_FACTORY,
                    gpu=0,
                    process="thinker_decode",
                    terminal=True,
                ),
            ],
        )
