# SPDX-License-Identifier: Apache-2.0
"""Prefill stops admitting when the Decode half is holding too much."""

from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.proto.messages import DataAckMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def _prefill(limit, depth) -> OmniScheduler:
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler._pd_role = "prefill"
    scheduler._pd_decode_pending_limit = limit
    scheduler._pd_peer_pending_fn = (lambda: depth) if depth is not None else None
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
    ack = DataAckMessage(
        request_id="r", from_stage="d", to_stage="p", object_id="o"
    )
    wire = ack.to_dict()

    assert "receiver_pending" not in wire
    assert DataAckMessage.from_dict(wire).receiver_pending is None
