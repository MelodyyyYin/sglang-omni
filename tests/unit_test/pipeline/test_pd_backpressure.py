# SPDX-License-Identifier: Apache-2.0
"""Prefill stops admitting when the Decode half is holding too much."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from sglang_omni.comm.engine import CommEngine
from sglang_omni.proto.messages import CapacityUpdateMessage, DataAckMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

_LOGGER = "sglang_omni.scheduling.omni_scheduler"


def _prefill(limit, depth) -> OmniScheduler:
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler._pd_role = "prefill"
    scheduler._pd_decode_pending_limit = limit
    scheduler._pd_peer_pending_fn = (lambda: depth) if depth is not None else None
    return scheduler


def _capacity_prefill(limit, capacity) -> OmniScheduler:
    scheduler = _prefill(limit, None)
    scheduler._pd_peer_capacity_fn = lambda: capacity
    return scheduler


def test_no_limit_reads_as_no_backpressure() -> None:
    """The default keeps the previous behaviour: nothing bounds Decode."""
    assert _prefill(None, 500)._pd_decode_pending_limit is None


def test_the_depth_is_read_through_the_installed_reader() -> None:
    assert _prefill(64, 437)._pd_peer_pending() == 437


def test_a_reader_that_raises_reports_no_reading() -> None:
    """A failed read must not stop admission; it is a pacing hint, not a gate."""

    def boom() -> int:
        raise RuntimeError("no peer yet")

    scheduler = _prefill(64, 0)
    scheduler._pd_peer_pending_fn = boom

    assert scheduler._pd_peer_pending() is None


def test_a_peer_that_never_reported_reads_as_none() -> None:
    assert _prefill(64, None)._pd_peer_pending() is None


def test_no_capacity_report_holds_a_hard_bound_closed() -> None:
    assert _capacity_prefill(2, None)._pd_decode_has_capacity() is False


def test_positive_capacity_allows_prefill_and_zero_holds_it() -> None:
    assert _capacity_prefill(2, 1)._pd_decode_has_capacity() is True
    assert _capacity_prefill(2, 0)._pd_decode_has_capacity() is False


def test_capacity_update_rejects_a_publisher_generation_change() -> None:
    source = CommEngine(SimpleNamespace(stage_name="prefill", comm_config={}))
    first = CapacityUpdateMessage("decode", "prefill", "generation-a", 1, 2, 2, 0)
    source.record_capacity_update(first)

    with pytest.raises(RuntimeError, match="changed generation"):
        source.record_capacity_update(
            CapacityUpdateMessage("decode", "prefill", "generation-b", 1, 2, 0, 2)
        )


def test_decode_drain_returns_capacity_without_another_handoff(monkeypatch) -> None:
    """reach limit -> stop -> Decode drains -> an independent update resumes."""

    async def scenario() -> None:
        source = CommEngine(
            SimpleNamespace(stage_name="thinker_prefill", comm_config={}),
            rank_endpoints={
                "thinker_prefill": ("ipc://prefill",),
                "thinker_decode": ("ipc://decode",),
            },
        )
        destination = CommEngine(
            SimpleNamespace(stage_name="thinker_decode", comm_config={}),
            rank_endpoints={
                "thinker_prefill": ("ipc://prefill",),
                "thinker_decode": ("ipc://decode",),
            },
        )
        depth = 2
        receiver = SimpleNamespace(pending_depth=lambda: depth)
        destination.register_kv_receiver("decode:kv", receiver)
        destination.configure_capacity_updates(
            to_stage="thinker_prefill", limit=2, interval_s=0.001
        )

        updates: asyncio.Queue[CapacityUpdateMessage] = asyncio.Queue()

        async def deliver(_sockets, _endpoint, message) -> None:
            assert isinstance(message, CapacityUpdateMessage)
            source.record_capacity_update(message)
            await updates.put(message)

        monkeypatch.setattr("sglang_omni.comm.engine.send_to_endpoint", deliver)
        task = asyncio.create_task(destination._run_capacity_updates())
        scheduler = _capacity_prefill(2, None)
        scheduler._pd_peer_capacity_fn = lambda: source.peer_capacity("thinker_decode")
        try:
            first = await asyncio.wait_for(updates.get(), timeout=1)
            assert first.receiver_pending == 2
            assert first.available_capacity == 0
            assert scheduler._pd_decode_has_capacity() is False

            # No DataReady/DataAck or new handoff occurs after Prefill stops.
            depth = 0
            resumed = await asyncio.wait_for(updates.get(), timeout=1)
            assert resumed.receiver_pending == 0
            assert resumed.available_capacity == 2
            assert scheduler._pd_decode_has_capacity() is True
        finally:
            destination._closed = True
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_the_decode_depth_counts_every_accepted_request() -> None:
    """Committed-but-unadmitted work counts too; the half has accepted it."""
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler._pd_ready_queue = SimpleNamespace(qsize=lambda: 3)
    scheduler._pd_deferred_admission = object()
    scheduler.waiting_queue = [object(), object()]
    scheduler.running_batch = SimpleNamespace(
        is_empty=lambda: False, reqs=[object()] * 4
    )

    assert scheduler._pd_decode_depth() == 3 + 1 + 2 + 4


def test_an_idle_decode_half_reports_zero() -> None:
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler._pd_ready_queue = SimpleNamespace(qsize=lambda: 0)
    scheduler._pd_deferred_admission = None
    scheduler.waiting_queue = []
    scheduler.running_batch = SimpleNamespace(is_empty=lambda: True, reqs=[])

    assert scheduler._pd_decode_depth() == 0


def test_the_ack_carries_the_depth_across_the_wire() -> None:
    ack = DataAckMessage(
        request_id="r",
        from_stage="thinker_decode",
        to_stage="thinker_prefill",
        object_id="o",
        receiver_pending=437,
    )

    assert DataAckMessage.from_dict(ack.to_dict()).receiver_pending == 437


def test_a_peer_that_omits_the_depth_stays_compatible() -> None:
    """An older peer sends no such field, and that is not an error."""
    ack = DataAckMessage(request_id="r", from_stage="d", to_stage="p", object_id="o")
    wire = ack.to_dict()

    assert "receiver_pending" not in wire
    assert DataAckMessage.from_dict(wire).receiver_pending is None


def _half(role: str, max_queued: int) -> OmniScheduler:
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler.server_args = SimpleNamespace(max_queued_requests=max_queued)
    return scheduler


def test_an_unbounded_decode_queue_is_stated_at_bind(caplog) -> None:
    """Otherwise the operator sees 100% admission and 41-second requests."""
    scheduler = _half("decode", 0)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        scheduler._warn_if_decode_queue_unbounded("thinker_decode", "decode")

    assert len(caplog.records) == 1
    assert "thinker_decode" in caplog.records[0].getMessage()


def test_the_notice_points_at_a_precedent(caplog) -> None:
    """ "Set something" without a reference leaves the operator guessing."""
    scheduler = _half("decode", 0)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        scheduler._warn_if_decode_queue_unbounded("thinker_decode", "decode")

    assert "qwen3_tts" in caplog.records[0].getMessage()


def test_a_bounded_queue_says_nothing(caplog) -> None:
    scheduler = _half("decode", 16)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        scheduler._warn_if_decode_queue_unbounded("thinker_decode", "decode")

    assert caplog.records == []


def test_the_prefill_half_says_nothing(caplog) -> None:
    """The queue this describes is the Decode half's."""
    scheduler = _half("prefill", 0)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        scheduler._warn_if_decode_queue_unbounded("thinker_prefill", "prefill")

    assert caplog.records == []


def test_priority_survives_the_handoff() -> None:
    """Decode ranks its waiting queue by priority; a dropped one ranks by arrival."""
    from sglang_omni.scheduling.pd_continuation import DecodeContinuation

    continuation = DecodeContinuation(
        request_id="req-1",
        transfer_id="t-1",
        origin_input_ids=[1, 2],
        output_ids=[3],
        vocab_size=32,
        sampling_params={},
        cached_tokens=0,
        priority=7,
    )

    assert DecodeContinuation.from_dict(continuation.to_dict()).priority == 7


def test_a_continuation_without_a_priority_stays_valid() -> None:
    """Priority is optional upstream, so its absence is not a schema error."""
    from sglang_omni.scheduling.pd_continuation import DecodeContinuation

    payload = DecodeContinuation(
        request_id="req-1",
        transfer_id="t-1",
        origin_input_ids=[1],
        output_ids=[2],
        vocab_size=32,
        sampling_params={},
        cached_tokens=0,
    ).to_dict()
    payload.pop("priority")

    assert DecodeContinuation.from_dict(payload).priority is None
