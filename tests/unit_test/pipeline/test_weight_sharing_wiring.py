# SPDX-License-Identifier: Apache-2.0
"""How a stage receives its weight-sharing plan.

A factory opts in by declaring ``weight_sharing_plan``. Nothing registers it,
because ``resolve_factory_signature_args`` injects a default only when the
factory names the parameter.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from pathlib import Path
from types import SimpleNamespace

from sglang_omni.pipeline.stage_workers import _weight_sharing_plan


def _spec(**overrides):
    base = {
        "stage_name": "thinker_prefill",
        "recv_endpoint": "ipc:///tmp/sglang_omni/run-ab12/stage_thinker_prefill.sock",
        "pd_execution": SimpleNamespace(role="prefill", partner="thinker_decode"),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_a_pd_half_gets_a_plan_naming_its_peer() -> None:
    plan = _weight_sharing_plan(_spec(), 0)

    assert plan.stage_name == "thinker_prefill"
    assert plan.peer_stage == "thinker_decode"
    assert plan.gpu_id == 0


def test_the_run_directory_comes_from_the_stage_endpoint() -> None:
    """allocate_endpoints puts every stage socket in the run's own directory."""
    plan = _weight_sharing_plan(_spec(), 0)

    assert plan.rendezvous_dir == Path("/tmp/sglang_omni/run-ab12")


def test_a_stage_that_is_not_a_pd_half_gets_no_plan() -> None:
    assert _weight_sharing_plan(_spec(pd_execution=None), 0) is None


def test_a_stage_without_a_gpu_gets_no_plan() -> None:
    """Sharing names device memory, so a CPU stage has nothing to share."""
    assert _weight_sharing_plan(_spec(), None) is None


def test_a_non_ipc_endpoint_yields_no_plan() -> None:
    """A TCP endpoint has no run directory, and guessing one would misplace it."""
    assert _weight_sharing_plan(_spec(recv_endpoint="tcp://127.0.0.1:5555"), 0) is None


def test_unsupported_factory_fails_before_waiting_for_publisher(monkeypatch) -> None:
    import pytest

    from sglang_omni.pipeline import stage_workers

    spec = _spec()
    spec.factory = "tests.unit_test.fixtures.pipeline_fakes.strict_pd_capable_factory"
    spec.factory_kwargs = {}
    spec.typed_kwargs = {}
    spec.factory_arg_defaults = {}
    spec.require_factory_gpu_id = False
    waited = False

    def should_not_wait(*_args, **_kwargs):
        nonlocal waited
        waited = True

    monkeypatch.setattr(stage_workers, "_adopt_peer_weights", should_not_wait)
    with pytest.raises(RuntimeError, match="does not support weight sharing"):
        stage_workers._construct_scheduler(spec, 0, logging.getLogger(__name__))

    assert waited is False


def test_publisher_process_death_is_pipeline_fatal() -> None:
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

    async def exercise():
        runner = object.__new__(MultiProcessPipelineRunner)
        runner._started = True
        runner._groups = [
            SimpleNamespace(
                any_dead=lambda: True,
                dead_summary=lambda: "thinker_prefill publisher exited",
            )
        ]
        failures = []

        async def fail_runtime(error):
            failures.append(error)
            runner._started = False

        runner._fail_runtime = fail_runtime
        await runner._monitor_children()
        return failures

    failures = asyncio.run(exercise())
    assert len(failures) == 1
    assert "thinker_prefill publisher exited" in str(failures[0])


def test_the_thinker_factory_declares_the_parameter() -> None:
    """Opting in is the declaration; without it the default is never injected."""
    from sglang_omni.models.qwen3_omni.stages import (
        create_sglang_thinker_executor_from_config,
    )

    assert (
        "weight_sharing_plan"
        in inspect.signature(create_sglang_thinker_executor_from_config).parameters
    )


def test_the_infrastructure_builder_accepts_it() -> None:
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure

    assert (
        "weight_sharing_plan"
        in inspect.signature(create_sglang_infrastructure).parameters
    )


def test_turning_sharing_off_yields_no_plan() -> None:
    """The escape hatch has to stop the plan, not just the swap."""
    spec = _spec(
        pd_execution=SimpleNamespace(
            role="prefill", partner="thinker_decode", share_weights=False
        )
    )

    assert _weight_sharing_plan(spec, 0) is None


def test_sharing_is_on_by_default() -> None:
    """Two copies of the same static tensors buy nothing."""
    from sglang_omni.config.schema import PDConfig

    assert PDConfig().share_weights is True


def test_the_setting_reaches_both_halves() -> None:
    from sglang_omni.config import expand_pd_stages
    from sglang_omni.config.schema import PDConfig, PDStagePlacement
    from tests.unit_test.pipeline.helpers import stage

    stages = [
        stage(
            "thinker",
            terminal=True,
            pd_disaggregation=PDConfig(
                prefill=PDStagePlacement(gpu=0),
                decode=PDStagePlacement(gpu=0),
                share_weights=False,
            ),
        )
    ]

    halves = {s.name: s for s in expand_pd_stages(stages, entry_stage="thinker").stages}

    assert halves["thinker_prefill"].pd_execution.share_weights is False
    assert halves["thinker_decode"].pd_execution.share_weights is False


def _plan(tmp_path, publishes: bool):
    from sglang_omni.model_runner.weight_sharing import WeightSharingPlan

    return WeightSharingPlan(
        stage_name="thinker_decode",
        peer_stage="thinker_prefill",
        rendezvous_dir=tmp_path,
        gpu_id=0,
        publishes=publishes,
    )


def test_an_adopter_without_room_is_rejected_by_name(tmp_path, monkeypatch) -> None:
    """A CUDA out-of-memory in the loader does not say the shares are wrong."""
    import pytest

    from sglang_omni.model_runner.weight_rendezvous import publish_parameter_handles
    from sglang_omni.pipeline.stage_workers import _check_room_to_load_before_swapping

    publish_parameter_handles(
        {"w": ("rebuild", ())},
        rendezvous_dir=tmp_path,
        stage_name="thinker_prefill",
        gpu_id=0,
        weight_bytes=60 * 1024**3,
    )
    monkeypatch.setattr(
        "torch.cuda.mem_get_info", lambda _d: (10 * 1024**3, 140 * 1024**3)
    )

    with pytest.raises(RuntimeError, match="one spare copy free"):
        _check_room_to_load_before_swapping(_plan(tmp_path, publishes=False), 0)


def test_an_adopter_with_room_proceeds(tmp_path, monkeypatch) -> None:
    from sglang_omni.model_runner.weight_rendezvous import publish_parameter_handles
    from sglang_omni.pipeline.stage_workers import _check_room_to_load_before_swapping

    publish_parameter_handles(
        {"w": ("rebuild", ())},
        rendezvous_dir=tmp_path,
        stage_name="thinker_prefill",
        gpu_id=0,
        weight_bytes=60 * 1024**3,
    )
    monkeypatch.setattr(
        "torch.cuda.mem_get_info", lambda _d: (80 * 1024**3, 140 * 1024**3)
    )

    _check_room_to_load_before_swapping(_plan(tmp_path, publishes=False), 0)


def test_a_publisher_without_a_recorded_size_is_not_blocked(tmp_path) -> None:
    """An older peer records no size, and guessing one would reject good runs."""
    from sglang_omni.pipeline.stage_workers import _check_room_to_load_before_swapping

    _check_room_to_load_before_swapping(_plan(tmp_path, publishes=False), 0)
