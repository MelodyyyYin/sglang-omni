# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from sglang_omni.config.schema import PDConfig, PDExecution
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.pd_continuation import DecodeContinuation
from sglang_omni.scheduling.pd_handoff_capacity import HandoffKVBudget
from sglang_omni.scheduling.pd_kv_adapter import SGLangKVPageLease
from tests.unit_test.pipeline.helpers import stage


def _req(request_id: str, tokens: int, *, cached: int = 0):
    return SimpleNamespace(
        rid=request_id,
        origin_input_ids=list(range(tokens)),
        prefix_indices=list(range(cached)),
        req_pool_idx=None,
        mamba_pool_idx=None,
    )


def _runtime(limit: int | None) -> Stage:
    runtime = Stage.__new__(Stage)
    runtime.pd_execution = PDExecution(
        role="prefill", partner="thinker_decode", max_inflight_handoffs=limit
    )
    return runtime


def _handoff(lease):
    return SimpleNamespace(
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
        source_page_indices=(1,),
        target_pool_id="decode:kv",
        to_stage="decode",
        lease=lease,
    )


def test_budget_uses_page_rounded_source_slots() -> None:
    budget = HandoffKVBudget(max_tokens=64, page_size=8)

    assert budget.weight(_req("below", 7)) == 8
    assert budget.weight(_req("boundary", 8)) == 8
    assert budget.weight(_req("above", 9)) == 16


def test_cached_prefix_still_counts_every_transferred_source_page() -> None:
    budget = HandoffKVBudget(max_tokens=16, page_size=8)
    req = _req("cached", 9, cached=7)

    assert budget.weight(req) == 16
    assert budget.submit(req)
    assert budget.snapshot() == (1, 16, 0)


def test_weighted_budget_cannot_be_exceeded_and_waiter_owns_no_kv() -> None:
    budget = HandoffKVBudget(max_tokens=16, page_size=8)
    held = _req("held", 9)
    waiter = _req("waiter", 1)

    assert budget.submit(held)
    assert not budget.submit(waiter)
    assert budget.snapshot() == (1, 16, 1)
    assert waiter.req_pool_idx is None

    assert budget.cancel_before_transfer("held") is held
    assert budget.admit_waiters() == [waiter]
    assert budget.snapshot() == (1, 8, 0)


def test_request_gate_does_not_limit_prefill_kv_admission() -> None:
    runtime = _runtime(1)
    budget = HandoffKVBudget(max_tokens=48, page_size=1)

    assert all(budget.submit(_req(f"r{i}", 8)) for i in range(6))
    assert runtime._pd_handoff_gate()._value == 1
    assert budget.snapshot() == (6, 48, 0)


def test_unset_request_bound_installs_no_gate() -> None:
    assert _runtime(None)._pd_handoff_gate() is None


def test_request_gate_is_reused() -> None:
    runtime = _runtime(4)

    assert runtime._pd_handoff_gate() is runtime._pd_handoff_gate()


def test_registry_shutdown_claims_only_pretransfer_requests() -> None:
    budget = HandoffKVBudget(max_tokens=16, page_size=8)
    pretransfer = _req("pretransfer", 8)
    transferred = _req("transferred", 8)
    waiter = _req("waiter", 1)
    assert budget.submit(pretransfer)
    assert budget.submit(transferred)
    assert not budget.submit(waiter)
    budget.mark_transferred("transferred")

    assert set(r.rid for r in budget.shutdown_pretransfer()) == {
        "pretransfer",
        "waiter",
    }
    assert budget.snapshot() == (1, 8, 0)
    assert budget.release_transferred("transferred")
    assert not budget.release_transferred("transferred")
    assert budget.snapshot() == (0, 0, 0)


def test_capacity_and_kv_release_once_during_terminal_race(monkeypatch) -> None:
    budget = HandoffKVBudget(max_tokens=8, page_size=1)
    req = _req("request-1", 8)
    assert budget.submit(req)
    budget.mark_transferred(req.rid)
    releases = []
    monkeypatch.setattr(
        "sglang.srt.mem_cache.common.release_kv_cache",
        lambda released, _cache: releases.append(released.rid),
    )
    lease = SGLangKVPageLease(
        req,
        object(),
        on_release=lambda: budget.release_transferred(req.rid),
    )

    threads = [threading.Thread(target=lease.release) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert releases == ["request-1"]
    assert budget.snapshot() == (0, 0, 0)


def test_request_count_gate_serializes_transfers_not_prefill() -> None:
    async def scenario() -> None:
        runtime = _runtime(1)
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        active = 0
        peak = 0

        async def send(_request_id, handoff):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if not first_entered.is_set():
                first_entered.set()
                await release_first.wait()
            handoff.lease.release()
            active -= 1

        runtime._send_pd_handoff_now = send
        leases = [SimpleNamespace(release=lambda: None) for _ in range(2)]
        tasks = [
            asyncio.create_task(runtime._send_pd_handoff(str(i), _handoff(lease)))
            for i, lease in enumerate(leases)
        ]
        await first_entered.wait()
        await asyncio.sleep(0)
        assert peak == 1
        release_first.set()
        await asyncio.gather(*tasks)
        assert peak == 1

    asyncio.run(scenario())


def test_cancel_while_waiting_for_transfer_gate_releases_lease() -> None:
    async def scenario() -> None:
        runtime = _runtime(1)
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        release_counts = [0, 0]

        async def send(request_id, _handoff):
            if request_id == "first":
                first_entered.set()
                await release_first.wait()

        runtime._send_pd_handoff_now = send
        handoffs = [
            _handoff(
                SimpleNamespace(
                    release=lambda index=index: release_counts.__setitem__(
                        index, release_counts[index] + 1
                    )
                )
            )
            for index in range(2)
        ]
        first = asyncio.create_task(runtime._send_pd_handoff("first", handoffs[0]))
        await first_entered.wait()
        second = asyncio.create_task(runtime._send_pd_handoff("second", handoffs[1]))
        await asyncio.sleep(0)
        second.cancel()
        await asyncio.gather(second, return_exceptions=True)
        assert release_counts == [0, 1]
        release_first.set()
        await first

    asyncio.run(scenario())


def test_task_cancelled_before_coroutine_start_releases_kv_and_budget(
    monkeypatch,
) -> None:
    async def scenario() -> None:
        budget = HandoffKVBudget(max_tokens=1, page_size=1)
        req = _req("request-1", 1)
        assert budget.submit(req)
        budget.mark_transferred(req.rid)
        released = []
        monkeypatch.setattr(
            "sglang.srt.mem_cache.common.release_kv_cache",
            lambda released_req, _cache: released.append(released_req.rid),
        )
        runtime = _runtime(1)
        runtime._receive_tasks = set()
        runtime._on_background_task_done = lambda *_args: None
        handoff = _handoff(
            SGLangKVPageLease(
                req,
                object(),
                on_release=lambda: budget.release_transferred(req.rid),
            )
        )

        runtime._launch_pd_handoff("request-1", handoff)
        task = next(iter(runtime._receive_tasks))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

        assert released == ["request-1"]
        assert budget.snapshot() == (0, 0, 0)

    asyncio.run(scenario())


def test_cancel_after_comm_ownership_does_not_release_source_early() -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        released = []

        class RetainingComm:
            async def send_kv_pages(self, **_kwargs):
                entered.set()
                await asyncio.Future()

        runtime = _runtime(1)
        runtime._comm = RetainingComm()
        runtime._clear_request_state = lambda _rid: None

        async def ignore_failure(*_args):
            pass

        runtime._send_failure = ignore_failure
        handoff = _handoff(SimpleNamespace(release=lambda: released.append(True)))
        task = asyncio.create_task(runtime._send_pd_handoff("request-1", handoff))
        await entered.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert released == []
        handoff.lease.release()
        assert released == [True]

    asyncio.run(scenario())


def test_send_failure_returns_kv_and_weighted_budget(monkeypatch) -> None:
    async def scenario() -> None:
        budget = HandoffKVBudget(max_tokens=1, page_size=1)
        req = _req("request-1", 1)
        assert budget.submit(req)
        budget.mark_transferred(req.rid)
        releases = []
        monkeypatch.setattr(
            "sglang.srt.mem_cache.common.release_kv_cache",
            lambda released, _cache: releases.append(released.rid),
        )
        lease = SGLangKVPageLease(
            req,
            object(),
            on_release=lambda: budget.release_transferred(req.rid),
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
        await runtime._send_pd_handoff("request-1", _handoff(lease))

        assert releases == ["request-1"]
        assert budget.snapshot() == (0, 0, 0)
        assert len(failures) == 1

    asyncio.run(scenario())


def test_scheduler_drains_budget_waiters_after_release() -> None:
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler._pd_handoff_kv_budget = HandoffKVBudget(max_tokens=8, page_size=8)
    scheduler.waiting_queue = []
    held = _req("held", 8)
    waiter = _req("waiter", 8)
    assert scheduler._pd_handoff_kv_budget.submit(held)
    assert not scheduler._pd_handoff_kv_budget.submit(waiter)

    scheduler._release_pd_handoff_budget(held)
    scheduler._drain_pd_handoff_budget_waiters()

    assert scheduler.waiting_queue == [waiter]
    assert scheduler._pd_handoff_kv_budget.snapshot() == (1, 8, 0)


def test_the_two_bounds_reach_both_halves() -> None:
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

    for half in halves.values():
        assert half.pd_execution.max_inflight_handoffs == 8
        assert half.pd_execution.max_inflight_handoff_tokens == 16384


def test_unset_bounds_stay_unset_through_rewrite() -> None:
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

    for half in halves.values():
        assert half.pd_execution.max_inflight_handoffs is None
        assert half.pd_execution.max_inflight_handoff_tokens is None


def test_decode_bound_reaches_both_halves() -> None:
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

    for half in halves.values():
        assert half.pd_execution.decode_pending_limit == 64


def test_unset_decode_bound_leaves_prefill_unthrottled() -> None:
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

    for half in halves.values():
        assert half.pd_execution.decode_pending_limit is None
