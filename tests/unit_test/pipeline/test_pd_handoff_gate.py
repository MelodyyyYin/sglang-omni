# SPDX-License-Identifier: Apache-2.0
"""Bounding how many PD handoffs are in flight at once.

Each handoff holds that request's prompt KV on the Prefill card until Decode
acknowledges it. Without a bound the count lands on Prefill's
max_running_requests, which sizes batches rather than leases.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from sglang_omni.config.schema import PDConfig, PDExecution
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.pd_continuation import DecodeContinuation
from sglang_omni.scheduling.pd_handoff_capacity import HandoffCapacity
from sglang_omni.scheduling.pd_kv_adapter import SGLangKVPageLease
from tests.unit_test.pipeline.helpers import stage


def _runtime(limit: int | None) -> Stage:
    runtime = Stage.__new__(Stage)
    runtime.pd_execution = PDExecution(
        role="prefill", partner="thinker_decode", max_inflight_handoffs=limit
    )
    return runtime


def test_limit_two_creates_only_two_kv_leases_for_six_requests(monkeypatch) -> None:
    """The bound protects ownership, not just running send coroutines."""

    released: list[str] = []
    monkeypatch.setattr(
        "sglang.srt.mem_cache.common.release_kv_cache",
        lambda req, _cache: released.append(req.rid),
    )
    capacity = HandoffCapacity(max_requests=2, max_tokens=12)
    leases = []
    for index in range(6):
        permit = capacity.try_acquire(6)
        if permit is not None:
            leases.append(
                SGLangKVPageLease(
                    SimpleNamespace(rid=f"r{index}"), object(), capacity_lease=permit
                )
            )

    assert len(leases) == 2
    assert capacity.snapshot() == (2, 12)
    assert released == []

    for lease in leases:
        lease.release()
    assert capacity.snapshot() == (0, 0)


def test_waiting_before_acquire_holds_no_capacity_or_kv() -> None:
    capacity = HandoffCapacity(max_requests=1, max_tokens=8)
    held = capacity.try_acquire(8)

    assert held is not None
    assert capacity.try_acquire(1) is None
    assert capacity.snapshot() == (1, 8)


def test_six_waiters_admit_only_two_before_prefill_kv_exists() -> None:
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler._pd_handoff_capacity = HandoffCapacity(max_requests=2, max_tokens=12)
    scheduler._pd_handoff_capacity_waiting = [
        SimpleNamespace(rid=f"r{index}", origin_input_ids=list(range(6)))
        for index in range(6)
    ]
    scheduler._aborted_request_ids = set()
    scheduler.waiting_queue = []

    scheduler._drain_pd_handoff_capacity_waiters()

    assert [req.rid for req in scheduler.waiting_queue] == ["r0", "r1"]
    assert [req.rid for req in scheduler._pd_handoff_capacity_waiting] == [
        "r2",
        "r3",
        "r4",
        "r5",
    ]
    assert scheduler._pd_handoff_capacity.snapshot() == (2, 12)
    assert all(
        not hasattr(req, "_pd_handoff_capacity_lease")
        for req in scheduler._pd_handoff_capacity_waiting
    )


def test_cancel_before_acquire_never_claims_capacity() -> None:
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler._pd_handoff_capacity = HandoffCapacity(max_requests=1, max_tokens=4)
    scheduler._pd_handoff_capacity_waiting = [
        SimpleNamespace(rid="cancelled", origin_input_ids=[1])
    ]
    scheduler._aborted_request_ids = {"cancelled"}
    scheduler.waiting_queue = []

    scheduler._drain_pd_handoff_capacity_waiters()

    assert scheduler._pd_handoff_capacity_waiting == []
    assert scheduler.waiting_queue == []
    assert scheduler._pd_handoff_capacity.snapshot() == (0, 0)


def test_capacity_and_lease_release_once_during_ack_shutdown_race(monkeypatch) -> None:
    capacity = HandoffCapacity(max_requests=1, max_tokens=4)
    permit = capacity.try_acquire(4)
    assert permit is not None
    releases = []
    monkeypatch.setattr(
        "sglang.srt.mem_cache.common.release_kv_cache",
        lambda req, _cache: releases.append(req.rid),
    )
    lease = SGLangKVPageLease(
        SimpleNamespace(rid="request-1"), object(), capacity_lease=permit
    )

    threads = [threading.Thread(target=lease.release) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert releases == ["request-1"]
    assert capacity.snapshot() == (0, 0)


def test_send_failure_returns_kv_and_weighted_capacity(monkeypatch) -> None:
    async def scenario() -> None:
        capacity = HandoffCapacity(max_requests=1, max_tokens=7)
        permit = capacity.try_acquire(7)
        assert permit is not None
        releases = []
        monkeypatch.setattr(
            "sglang.srt.mem_cache.common.release_kv_cache",
            lambda req, _cache: releases.append(req.rid),
        )
        lease = SGLangKVPageLease(
            SimpleNamespace(rid="request-1"), object(), capacity_lease=permit
        )

        class FailingComm:
            async def send_kv_pages(self, **_kwargs):
                lease.release()
                raise RuntimeError("send failed")

        runtime = _runtime(1)
        runtime._comm = FailingComm()
        runtime._clear_request_state = lambda _rid: None
        failures = []

        async def record_failure(*args):
            failures.append(args)

        runtime._send_failure = record_failure
        handoff = SimpleNamespace(
            continuation=DecodeContinuation(
                request_id="request-1",
                transfer_id="transfer-1",
                origin_input_ids=[1],
                output_ids=[2],
                vocab_size=16,
                sampling_params={},
                cached_tokens=0,
            ),
            source_pool_id="prefill:kv",
            source_page_indices=tuple(range(7)),
            target_pool_id="decode:kv",
            to_stage="decode",
            lease=lease,
        )

        await runtime._send_pd_handoff("request-1", handoff)

        assert releases == ["request-1"]
        assert capacity.snapshot() == (0, 0)
        assert len(failures) == 1

    asyncio.run(scenario())


def test_task_cancelled_before_start_releases_lease_and_capacity(monkeypatch) -> None:
    async def scenario() -> None:
        capacity = HandoffCapacity(max_requests=1, max_tokens=1)
        permit = capacity.try_acquire(1)
        assert permit is not None
        releases = []
        monkeypatch.setattr(
            "sglang.srt.mem_cache.common.release_kv_cache",
            lambda req, _cache: releases.append(req.rid),
        )
        lease = SGLangKVPageLease(
            SimpleNamespace(rid="request-1"), object(), capacity_lease=permit
        )
        runtime = _runtime(1)
        runtime._receive_tasks = set()
        runtime._on_background_task_done = lambda *_args: None
        handoff = SimpleNamespace(lease=lease)

        runtime._launch_pd_handoff("request-1", handoff)
        task = next(iter(runtime._receive_tasks))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

        assert releases == ["request-1"]
        assert capacity.snapshot() == (0, 0)

    asyncio.run(scenario())


def test_the_bound_reaches_both_halves_from_one_setting() -> None:
    """It is a property of the pair, so the rewrite copies it to each half."""
    from sglang_omni.config import expand_pd_stages
    from sglang_omni.config.schema import PDStagePlacement

    stages = [
        stage(
            "thinker",
            terminal=True,
            pd_disaggregation=PDConfig(
                prefill=PDStagePlacement(gpu=0),
                decode=PDStagePlacement(gpu=1),
                max_inflight_handoffs=8,
                max_inflight_handoff_tokens=16384,
            ),
        )
    ]

    halves = {s.name: s for s in expand_pd_stages(stages, entry_stage="thinker").stages}

    assert halves["thinker_prefill"].pd_execution.max_inflight_handoffs == 8
    assert halves["thinker_decode"].pd_execution.max_inflight_handoffs == 8
    assert halves["thinker_prefill"].pd_execution.max_inflight_handoff_tokens == 16384


def test_an_unset_bound_stays_unset_through_the_rewrite() -> None:
    from sglang_omni.config import expand_pd_stages
    from sglang_omni.config.schema import PDStagePlacement

    stages = [
        stage(
            "thinker",
            terminal=True,
            pd_disaggregation=PDConfig(
                prefill=PDStagePlacement(gpu=0), decode=PDStagePlacement(gpu=1)
            ),
        )
    ]

    halves = {s.name: s for s in expand_pd_stages(stages, entry_stage="thinker").stages}

    assert halves["thinker_prefill"].pd_execution.max_inflight_handoffs is None


def test_the_decode_bound_reaches_both_halves() -> None:
    """Prefill needs it to throttle; Decode carries it so the pair agrees."""
    from sglang_omni.config import expand_pd_stages
    from sglang_omni.config.schema import PDStagePlacement

    stages = [
        stage(
            "thinker",
            terminal=True,
            pd_disaggregation=PDConfig(
                prefill=PDStagePlacement(gpu=0),
                decode=PDStagePlacement(gpu=1),
                decode_pending_limit=64,
            ),
        )
    ]

    halves = {s.name: s for s in expand_pd_stages(stages, entry_stage="thinker").stages}

    assert halves["thinker_prefill"].pd_execution.decode_pending_limit == 64
    assert halves["thinker_decode"].pd_execution.decode_pending_limit == 64


def test_an_unset_decode_bound_leaves_prefill_unthrottled() -> None:
    from sglang_omni.config import expand_pd_stages
    from sglang_omni.config.schema import PDStagePlacement

    stages = [
        stage(
            "thinker",
            terminal=True,
            pd_disaggregation=PDConfig(
                prefill=PDStagePlacement(gpu=0), decode=PDStagePlacement(gpu=1)
            ),
        )
    ]

    halves = {s.name: s for s in expand_pd_stages(stages, entry_stage="thinker").stages}

    assert halves["thinker_prefill"].pd_execution.decode_pending_limit is None
