# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the T7 PreparedRequestQueue (RFC #661). CPU-only."""

from sglang_omni.scheduling.prepared_request_queue import PreparedRequestQueue


def _active_queue() -> PreparedRequestQueue:
    q: PreparedRequestQueue = PreparedRequestQueue()
    q.set_context(object())
    return q


def test_begin_then_publish_stores_and_pop_returns():
    q = _active_queue()
    assert q.begin("a") is not None
    assert "a" in q.inflight
    assert q.publish("a", "PREP-a") is True
    assert "a" not in q.inflight
    assert q.pop("a") == "PREP-a"
    assert q.pop("a") is None


def test_abort_while_inflight_then_publish_drops():
    q = _active_queue()
    q.begin("a")
    q.abort("a")
    assert "a" in q.aborted
    assert q.publish("a", "PREP-a") is False
    assert "a" not in q.prepared
    assert "a" not in q.aborted


def test_abort_published_drops_it():
    q = _active_queue()
    q.begin("a")
    q.publish("a", "PREP-a")
    assert "a" in q.prepared
    q.abort("a")
    assert "a" not in q.prepared
    assert "a" not in q.aborted


def test_abort_unknown_request_is_noop():
    q = _active_queue()
    q.abort("ghost")
    assert not q.aborted
    assert not q.inflight
    assert not q.prepared


def test_fail_inflight_discards_and_stores_nothing():
    q = _active_queue()
    q.begin("a")
    q.fail_inflight("a")
    assert not q.inflight
    assert not q.aborted
    assert "a" not in q.prepared


def test_set_and_clear_context_reset_state():
    q = _active_queue()
    q.begin("a")
    q.publish("a", "PREP-a")
    q.begin("b")
    q.set_context(object())
    assert not q.prepared and not q.inflight and not q.aborted
    q.begin("c")
    q.clear_context()
    assert q.context is None
    assert not q.prepared and not q.inflight and not q.aborted


def test_begin_without_context_returns_none_and_no_inflight():
    q: PreparedRequestQueue = PreparedRequestQueue()
    assert q.begin("a") is None
    assert "a" not in q.inflight
