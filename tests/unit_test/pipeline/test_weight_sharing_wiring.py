# SPDX-License-Identifier: Apache-2.0
"""How a stage receives its weight-sharing plan.

A factory opts in by declaring ``weight_sharing_plan``. Nothing registers it,
because ``resolve_factory_signature_args`` injects a default only when the
factory names the parameter.
"""

from __future__ import annotations

import inspect
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
