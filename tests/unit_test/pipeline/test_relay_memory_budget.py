# SPDX-License-Identifier: Apache-2.0
"""Startup accounting for CUDA-IPC payload relay memory."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import sglang_omni.platforms as platforms
from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.comm.router import CommRouter
from sglang_omni.pipeline import stage_workers
from sglang_omni.pipeline.stage_workers import (
    StageLaunchConfig,
    _adjust_factory_budget_for_cuda_relay,
    _plan_cuda_ipc_relay_reservation,
)


@pytest.fixture(autouse=True)
def _cuda_ipc_platform(monkeypatch):
    monkeypatch.setattr(
        platforms.current_platform,
        "get_intra_node_transport",
        lambda: TransportKind.CUDA_IPC,
    )


def _spec(**overrides) -> StageLaunchConfig:
    base = dict(
        stage_name="image_encoder",
        gpu_id=0,
        placement_gpu_id=0,
        next_stages="mm_aggregate",
        gpu_stage_names={"image_encoder", "mm_aggregate"},
        stage_gpu_ids={"image_encoder": (0,), "mm_aggregate": (0,)},
        comm_config={
            "cuda_ipc_preallocate_pool": True,
            "cuda_ipc_pool_size_mb": 64,
            "cuda_ipc_slot_size_kb": 64,
        },
    )
    base.update(overrides)
    return StageLaunchConfig(**base)


def test_image_capable_route_has_a_pre_ready_relay_reservation() -> None:
    reservation = _plan_cuda_ipc_relay_reservation(_spec())

    assert reservation is not None
    assert reservation.pool_size_bytes == 64 * 1024**2


def test_relay_is_reserved_before_the_stage_factory_runs(monkeypatch) -> None:
    events = []
    spec = _spec(
        factory="fake.factory",
        factory_kwargs={},
        typed_kwargs={},
        factory_arg_defaults={"total_gpu_memory_fraction": 0.5},
    )
    reservation = _plan_cuda_ipc_relay_reservation(spec)

    def factory():
        events.append("factory")
        return object()

    @contextmanager
    def startup_lock(_gpu_id):
        yield "/tmp/fake-lock"

    monkeypatch.setattr(stage_workers, "import_string", lambda _path: factory)
    monkeypatch.setattr(stage_workers, "gpu_startup_lock", startup_lock)
    monkeypatch.setattr(
        stage_workers.torch.cuda,
        "get_device_properties",
        lambda _gpu_id: SimpleNamespace(total_memory=1024 * 1024**2),
    )

    stage_workers._construct_scheduler(
        spec,
        0,
        logging.getLogger(__name__),
        cuda_relay_reservation=reservation,
        before_factory=lambda: events.append("relay_reserved"),
    )

    assert events == ["relay_reserved", "factory"]


def test_relay_budget_is_charged_inside_stage_gpu_fraction() -> None:
    reservation = _plan_cuda_ipc_relay_reservation(_spec())
    defaults = {"total_gpu_memory_fraction": 0.50}

    adjusted = _adjust_factory_budget_for_cuda_relay(
        defaults,
        reservation,
        total_gpu_memory_bytes=1024 * 1024**2,
    )

    assert adjusted["total_gpu_memory_fraction"] == pytest.approx(0.4375)
    assert defaults == {"total_gpu_memory_fraction": 0.50}


def test_insufficient_stage_budget_fails_before_factory_startup() -> None:
    reservation = _plan_cuda_ipc_relay_reservation(_spec())

    with pytest.raises(RuntimeError, match="relay reservation.*stage GPU budget"):
        _adjust_factory_budget_for_cuda_relay(
            {"total_gpu_memory_fraction": 0.05},
            reservation,
            total_gpu_memory_bytes=1024 * 1024**2,
        )


def test_text_only_route_has_no_relay_reservation() -> None:
    spec = _spec(
        stage_name="text_decoder",
        next_stages="detokenizer",
        gpu_stage_names={"text_decoder"},
        stage_gpu_ids={"text_decoder": (0,)},
        comm_config={
            "cuda_ipc_preallocate_pool": False,
            "cuda_ipc_pool_size_mb": 64,
        },
    )

    assert _plan_cuda_ipc_relay_reservation(spec) is None


def test_first_image_payload_reuses_the_preallocated_pool(monkeypatch) -> None:
    allocations = []

    class FakeRelay:
        def __init__(self, **kwargs):
            allocations.append(kwargs)
            self.pool_size = kwargs["pool_size_mb"] * 1024**2
            self._pool_tensor = object() if kwargs["preallocate_pool"] else None

    monkeypatch.setattr("sglang_omni.pipeline.stage_workers.CudaIpcRelay", FakeRelay)
    spec = _spec()
    reservation = _plan_cuda_ipc_relay_reservation(spec)
    relay = reservation.build(spec)

    assert relay._pool_tensor is not None
    assert len(allocations) == 1
    assert allocations[0]["preallocate_pool"] is True

    router = CommRouter(
        stage_name="image_encoder",
        gpu_id=0,
        placement_gpu_id=0,
        same_process_targets=set(),
        gpu_stage_names={"mm_aggregate"},
        stage_gpu_ids={"mm_aggregate": (0,)},
        prebuilt_relays={TransportKind.CUDA_IPC: relay},
    )
    assert router.relay_for("mm_aggregate")[1] is relay
    assert router.relay_for("mm_aggregate")[1] is relay
    assert len(allocations) == 1
