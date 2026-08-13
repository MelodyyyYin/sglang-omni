# SPDX-License-Identifier: Apache-2.0
"""Runtime-layer tests for PD stage compilation.

Exercise the physical bookkeeping after PD expansion: typed metadata,
terminal/completion accounting, composed routing, and abort fan-out. No
scheduler/KV/request handoff logic is tested here (PR 2).
"""
from __future__ import annotations

import asyncio

import pytest

from sglang_omni.config.runtime import resolve_factory_signature_args
from sglang_omni.config.schema import (
    EndpointsConfig,
    PDConfig,
    PDStagePlacement,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.mp_runner import _build_stage_groups
from sglang_omni.pipeline.runtime_config import _compose_name_map, prepare_pipeline_runtime
from sglang_omni.proto import CompleteMessage, OmniRequest
from sglang_omni.utils.imports import import_string
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeMpContext,
    RecordingCoordinatorControlPlane,
    fake_factory_path,
)

PD_FACTORY = fake_factory_path("pd_capable_factory")
STRICT_PD_FACTORY = fake_factory_path("strict_pd_capable_factory")
PLAIN_FACTORY = fake_factory_path("dummy_factory")


def _pd(prefill_gpu: int, decode_gpu: int) -> PDConfig:
    return PDConfig(
        prefill=PDStagePlacement(gpu=prefill_gpu),
        decode=PDStagePlacement(gpu=decode_gpu),
    )


def _pd_pipeline(
    tmp_path, *, factory: str = PD_FACTORY, terminal: bool = False
) -> PipelineConfig:
    stages = [
        StageConfig(name="pre", factory=factory, gpu=0, process="pre", next="thinker"),
        StageConfig(
            name="thinker",
            factory=factory,
            gpu=1,
            process="thinker",
            terminal=terminal,
            next=None if terminal else "post",
            pd_disaggregation=_pd(1, 2),
        ),
    ]
    if not terminal:
        stages.append(
            StageConfig(name="post", factory=factory, gpu=3, process="post", terminal=True)
        )
    return PipelineConfig(
        model_path="dummy",
        name="pd",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        stages=stages,
    )


def _specs_by_name(config, prep):
    groups = _build_stage_groups(
        config,
        ctx=FakeMpContext(),
        stages_cfg=prep.stages_cfg,
        name_map=prep.name_map,
        endpoints=prep.endpoints,
        placement_plan=prep.placement_plan,
        process_plan=prep.process_plan,
    )
    return {spec.stage_name: spec for group in groups for spec in group.specs}


# --- Part A: typed PD metadata / capability -------------------------------


def test_launch_specs_carry_typed_pd_execution_not_factory_args(tmp_path) -> None:
    config = _pd_pipeline(tmp_path)
    prep = prepare_pipeline_runtime(config)
    with prep.runtime_dir:
        specs = _specs_by_name(config, prep)

    prefill = specs["thinker_prefill"]
    decode = specs["thinker_decode"]

    assert (prefill.pd_execution.role, prefill.pd_execution.partner) == (
        "prefill",
        "thinker_decode",
    )
    assert (decode.pd_execution.role, decode.pd_execution.partner) == (
        "decode",
        "thinker_prefill",
    )

    # PD metadata must never leak into factory kwargs.
    for spec in (prefill, decode):
        assert "pd_role" not in spec.factory_args
        assert "pd_partner" not in spec.factory_args


def test_non_pd_stage_has_no_pd_execution(tmp_path) -> None:
    config = _pd_pipeline(tmp_path)
    prep = prepare_pipeline_runtime(config)
    with prep.runtime_dir:
        specs = _specs_by_name(config, prep)

    assert specs["pre"].pd_execution is None
    assert specs["post"].pd_execution is None
    assert specs["pre"].factory_args == {}


def test_strict_factory_never_receives_pd_role_or_partner(tmp_path) -> None:
    config = _pd_pipeline(tmp_path, factory=STRICT_PD_FACTORY)
    prep = prepare_pipeline_runtime(config)
    with prep.runtime_dir:
        specs = _specs_by_name(config, prep)

    decode = specs["thinker_decode"]
    factory = import_string(decode.factory)
    resolved = resolve_factory_signature_args(
        factory,
        decode.factory_args,
        defaults=decode.factory_arg_defaults,
    )
    assert "pd_role" not in resolved
    assert "pd_partner" not in resolved
    assert factory(**resolved) == {"model_path": "dummy", "gpu_id": resolved["gpu_id"]}


def test_pd_stage_with_non_capable_factory_is_rejected_before_launch(
    tmp_path,
) -> None:
    config = _pd_pipeline(tmp_path, factory=PLAIN_FACTORY)
    with pytest.raises(ValueError, match="not PD-capable"):
        prepare_pipeline_runtime(config)


def test_ordinary_non_pd_pipeline_is_unaffected_by_capability_check(
    tmp_path,
) -> None:
    config = PipelineConfig(
        model_path="dummy",
        name="plain",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        stages=[
            StageConfig(name="a", factory=PLAIN_FACTORY, process="a", next="b"),
            StageConfig(name="b", factory=PLAIN_FACTORY, process="b", terminal=True),
        ],
    )
    prep = prepare_pipeline_runtime(config)
    with prep.runtime_dir:
        specs = _specs_by_name(config, prep)
    assert all(spec.pd_execution is None for spec in specs.values())


# --- Part B: physical terminal / completion accounting --------------------


def test_prep_exposes_physical_terminal_identity(tmp_path) -> None:
    config = _pd_pipeline(tmp_path, terminal=True)
    prep = prepare_pipeline_runtime(config)
    with prep.runtime_dir:
        assert config.terminal_stages == ["thinker"]  # logical (stale) view
        assert prep.terminal_stages == ["thinker_decode"]
        assert prep.terminal_name_map["thinker"] == "thinker_decode"


def test_coordinator_completes_on_decode_not_prefill() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="thinker_prefill",
            terminal_stages=["thinker_decode"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("thinker_prefill", "inproc://prefill")
        coordinator.register_stage("thinker_decode", "inproc://decode")

        await coordinator._submit_request("req-1", OmniRequest(inputs="hi"))
        future = coordinator._completion_futures["req-1"]

        # Prefill completion must not finish the request.
        await coordinator._handle_completion(
            CompleteMessage("req-1", "thinker_prefill", True, result={"stale": 1})
        )
        assert not future.done()
        assert "req-1" in coordinator._requests

        # Decode completion finishes the request.
        await coordinator._handle_completion(
            CompleteMessage("req-1", "thinker_decode", True, result={"text": "ok"})
        )
        assert future.result() == {"text": "ok"}

    asyncio.run(_run())


def test_terminal_resolver_logical_name_maps_to_decode() -> None:
    async def _run() -> None:
        def resolver(request: OmniRequest):
            return ["thinker"]

        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="thinker_prefill",
            terminal_stages=["thinker_decode"],
            terminal_stages_resolver=resolver,
            terminal_name_map={"thinker": "thinker_decode"},
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("thinker_prefill", "inproc://prefill")
        coordinator.register_stage("thinker_decode", "inproc://decode")

        await coordinator._submit_request("req-1", OmniRequest(inputs="hi"))
        await coordinator._handle_completion(
            CompleteMessage("req-1", "thinker_decode", True, result={"text": "ok"})
        )
        assert coordinator._completion_futures["req-1"].result() == {"text": "ok"}

    asyncio.run(_run())


# --- Part C: composed logical->physical routing ---------------------------


def test_prep_entry_stage_is_prefill_when_pd_is_entry(tmp_path) -> None:
    config = PipelineConfig(
        model_path="dummy",
        name="pd-entry",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        entry_stage="thinker",
        stages=[
            StageConfig(
                name="thinker",
                factory=PD_FACTORY,
                gpu=1,
                process="thinker",
                terminal=True,
                pd_disaggregation=_pd(1, 2),
            ),
        ],
    )
    prep = prepare_pipeline_runtime(config)
    with prep.runtime_dir:
        assert prep.entry_stage == "thinker_prefill"


def test_name_map_resolves_logical_pd_target_to_prefill(tmp_path) -> None:
    config = _pd_pipeline(tmp_path)
    prep = prepare_pipeline_runtime(config)
    with prep.runtime_dir:
        specs = _specs_by_name(config, prep)

    name_map = specs["pre"].name_map
    assert name_map["thinker"] == "thinker_prefill"
    assert name_map["post"] == "post"
    assert name_map["thinker_prefill"] == "thinker_prefill"
    assert name_map["thinker_decode"] == "thinker_decode"


def test_compose_name_map_composes_fusion_then_pd() -> None:
    fusion_map = {"raw": "thinker", "thinker": "thinker", "sink": "sink"}
    routing_map = {"thinker": "thinker_prefill"}
    stages = [
        StageConfig(name="thinker_prefill", factory=PD_FACTORY, process="p", terminal=True),
        StageConfig(name="thinker_decode", factory=PD_FACTORY, process="d", terminal=True),
        StageConfig(name="sink", factory=PD_FACTORY, process="s", terminal=True),
    ]
    composed = _compose_name_map(fusion_map, routing_map, stages)

    assert composed["raw"] == "thinker_prefill"
    assert composed["thinker"] == "thinker_prefill"
    assert composed["sink"] == "sink"
    # Generated physical names resolve to themselves.
    assert composed["thinker_prefill"] == "thinker_prefill"
    assert composed["thinker_decode"] == "thinker_decode"


# --- Part D/E: abort fan-out + decode registration without inbound edge ----


def test_both_pd_halves_are_registrable_and_share_abort_endpoint(tmp_path) -> None:
    config = _pd_pipeline(tmp_path)
    prep = prepare_pipeline_runtime(config)
    with prep.runtime_dir:
        groups = _build_stage_groups(
            config,
            ctx=FakeMpContext(),
            stages_cfg=prep.stages_cfg,
            name_map=prep.name_map,
            endpoints=prep.endpoints,
            placement_plan=prep.placement_plan,
            process_plan=prep.process_plan,
        )
        registered = {
            stage_name
            for group in groups
            for stage_name in group.stage_control_endpoints
        }
        specs = {spec.stage_name: spec for group in groups for spec in group.specs}

    # Decode has no inbound edge but still launches and subscribes to abort.
    assert {"thinker_prefill", "thinker_decode"} <= registered
    assert all(s.next != "thinker_decode" for s in prep.stages_cfg)
    assert (
        specs["thinker_prefill"].abort_endpoint
        == specs["thinker_decode"].abort_endpoint
        == prep.endpoints["abort"]
    )


def test_abort_broadcasts_to_all_registered_pd_stages() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="thinker_prefill",
            terminal_stages=["thinker_decode"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("thinker_prefill", "inproc://prefill")
        coordinator.register_stage("thinker_decode", "inproc://decode")

        await coordinator._submit_request("req-1", OmniRequest(inputs="hi"))
        assert await coordinator.abort("req-1") is True

        # One PUB broadcast fans out to both halves through the shared abort endpoint.
        assert [msg.request_id for msg in control_plane.aborts] == ["req-1"]
        assert {"thinker_prefill", "thinker_decode"} <= set(coordinator._stages)

    asyncio.run(_run())
