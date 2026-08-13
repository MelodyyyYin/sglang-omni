# SPDX-License-Identifier: Apache-2.0
"""Tests for the generic PR 2 Prefill-Decode continuation handoff."""

from __future__ import annotations

import asyncio
import dataclasses
import time

import msgspec
import pytest

from sglang_omni.comm.kv_transfer import (
    KVPageDestination,
    KVTransferPrepareMessage,
)
from sglang_omni.proto import KVBufferSpec, KVPoolLayout
from sglang_omni.scheduling.pd_capabilities import PDCapabilityError, PDCapabilityPolicy
from sglang_omni.scheduling.pd_continuation import (
    CONTINUATION_VERSION,
    ContinuationAwareKVReceiver,
    ContinuationSchemaError,
    ContinuationSerializer,
    DecodeContinuation,
    PDHandoffController,
    PendingHandoff,
    PrefillContinuationProducer,
)


def _sample_continuation(**overrides) -> DecodeContinuation:
    defaults = dict(
        request_id="req-1",
        transfer_id="xfer-1",
        origin_input_ids=[1, 2, 3, 4],
        output_ids=[42],
        vocab_size=32000,
        sampling_params={"temperature": 0.7, "max_new_tokens": 128},
        cached_tokens=4,
    )
    defaults.update(overrides)
    return DecodeContinuation(**defaults)


class _FakeReceiver:
    """In-memory KVReceiver that records call history."""

    def __init__(self) -> None:
        self.reserves: list[KVTransferPrepareMessage] = []
        self.commits: list[tuple[KVTransferPrepareMessage, KVPageDestination]] = []
        self.aborts: list[tuple[KVTransferPrepareMessage, KVPageDestination | None, BaseException]] = []

    def reserve(self, request: KVTransferPrepareMessage) -> KVPageDestination:
        self.reserves.append(request)
        return KVPageDestination(pool_id="pool", page_indices=(0, 1))

    def commit(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination,
    ) -> None:
        self.commits.append((request, destination))

    def abort(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination | None,
        error: BaseException,
    ) -> None:
        self.aborts.append((request, destination, error))


def test_continuation_round_trip():
    cont = _sample_continuation()
    data = ContinuationSerializer.encode(cont)
    decoded = ContinuationSerializer.decode(data)
    assert decoded == cont


def test_schema_version_rejection():
    cont = _sample_continuation()
    d = cont.to_dict()
    d["version"] = "pd-continuation-v0"
    bad = msgspec.msgpack.encode(d)
    with pytest.raises(ContinuationSchemaError):
        ContinuationSerializer.decode(bad)


def test_unknown_key_rejection():
    cont = _sample_continuation()
    d = cont.to_dict()
    d["extra_field"] = 123
    bad = msgspec.msgpack.encode(d)
    with pytest.raises(ContinuationSchemaError):
        ContinuationSerializer.decode(bad)


def test_exactly_once_admission():
    admitted: list[PendingHandoff] = []

    def admit(pending: PendingHandoff) -> None:
        admitted.append(pending)

    ctrl = PDHandoffController(admit_callback=admit)
    cont = _sample_continuation()

    ctrl.start_handoff(cont.request_id, cont.transfer_id)
    ctrl.set_continuation(cont.request_id, cont)
    assert not ctrl.is_admitted(cont.request_id)

    ctrl.set_kv_committed(cont.request_id)
    assert ctrl.is_admitted(cont.request_id)
    assert len(admitted) == 1
    assert admitted[0].continuation is cont

    # Duplicate commit/continuation must not trigger another admission.
    ctrl.set_kv_committed(cont.request_id)
    ctrl.set_continuation(cont.request_id, cont)
    assert len(admitted) == 1


def test_kv_first_then_continuation_still_admits():
    admitted: list[PendingHandoff] = []
    ctrl = PDHandoffController(admit_callback=lambda p: admitted.append(p))
    cont = _sample_continuation()

    ctrl.start_handoff(cont.request_id, cont.transfer_id)
    ctrl.set_kv_committed(cont.request_id)
    assert not ctrl.is_admitted(cont.request_id)
    ctrl.set_continuation(cont.request_id, cont)
    assert ctrl.is_admitted(cont.request_id)
    assert len(admitted) == 1


def test_ack_not_required_for_admission():
    """Admission is continuation + KV commit; no ACK is modelled here."""
    admitted: list[PendingHandoff] = []
    ctrl = PDHandoffController(admit_callback=lambda p: admitted.append(p))
    cont = _sample_continuation()

    ctrl.start_handoff(cont.request_id, cont.transfer_id)
    ctrl.set_continuation(cont.request_id, cont)
    ctrl.set_kv_committed(cont.request_id)
    assert admitted


def test_duplicate_and_stale_messages():
    admitted: list[PendingHandoff] = []
    ctrl = PDHandoffController(admit_callback=lambda p: admitted.append(p))
    cont = _sample_continuation()

    ctrl.start_handoff(cont.request_id, cont.transfer_id)
    ctrl.set_continuation(cont.request_id, cont)
    ctrl.set_kv_committed(cont.request_id)

    # Stale request id should raise or be ignored.
    with pytest.raises(KeyError):
        ctrl.set_continuation("unknown-req", cont)

    # After admission, extra commits are ignored.
    ctrl.set_kv_committed(cont.request_id)
    assert len(admitted) == 1


def test_abort_cleanup_before_admission():
    cleanups: list[tuple[PendingHandoff, str]] = []

    def cleanup(pending: PendingHandoff, reason: str) -> None:
        cleanups.append((pending, reason))

    ctrl = PDHandoffController(cleanup_callback=cleanup)
    cont = _sample_continuation()

    ctrl.start_handoff(cont.request_id, cont.transfer_id)
    ctrl.set_continuation(cont.request_id, cont)
    ctrl.abort(cont.request_id, reason="user-cancel")

    assert ctrl.get_pending(cont.request_id).aborted
    assert cleanups and cleanups[0][1] == "user-cancel"

    # Subsequent commit/aborts must not re-trigger cleanup.
    ctrl.set_kv_committed(cont.request_id)
    ctrl.abort(cont.request_id, reason="again")
    assert len(cleanups) == 1


def test_abort_after_commit_before_admission():
    cleanups: list[tuple[PendingHandoff, str]] = []
    ctrl = PDHandoffController(
        cleanup_callback=lambda p, r: cleanups.append((p, r))
    )
    cont = _sample_continuation()

    ctrl.start_handoff(cont.request_id, cont.transfer_id)
    # KV is committed before continuation arrives; request is then aborted.
    ctrl.set_kv_committed(cont.request_id)
    ctrl.abort(cont.request_id, reason="dropped")
    ctrl.set_continuation(cont.request_id, cont)

    assert ctrl.get_pending(cont.request_id).aborted
    assert not ctrl.is_admitted(cont.request_id)
    assert len(cleanups) == 1


def test_abort_after_admission_is_ignored():
    admitted: list[PendingHandoff] = []
    cleanups: list[tuple[PendingHandoff, str]] = []
    ctrl = PDHandoffController(
        admit_callback=lambda p: admitted.append(p),
        cleanup_callback=lambda p, r: cleanups.append((p, r)),
    )
    cont = _sample_continuation()

    ctrl.start_handoff(cont.request_id, cont.transfer_id)
    ctrl.set_continuation(cont.request_id, cont)
    ctrl.set_kv_committed(cont.request_id)
    assert admitted

    ctrl.abort(cont.request_id, reason="late")
    assert not ctrl.get_pending(cont.request_id).aborted
    assert not cleanups


@pytest.mark.asyncio
async def test_timeout_aborts_unfinished_handoff():
    cleanups: list[tuple[PendingHandoff, str]] = []
    ctrl = PDHandoffController(
        default_timeout_s=0.05,
        cleanup_callback=lambda p, r: cleanups.append((p, r)),
    )
    cont = _sample_continuation()

    ctrl.start_handoff(cont.request_id, cont.transfer_id)
    assert not ctrl.get_pending(cont.request_id).aborted
    await asyncio.sleep(0.15)
    assert ctrl.get_pending(cont.request_id).aborted
    assert cleanups and cleanups[0][1] == "timeout"


def test_timeout_check_timeouts_sync():
    cleanups: list[tuple[PendingHandoff, str]] = []
    ctrl = PDHandoffController(
        default_timeout_s=0.01,
        cleanup_callback=lambda p, r: cleanups.append((p, r)),
    )
    cont = _sample_continuation()

    ctrl.start_handoff(cont.request_id, cont.transfer_id)
    assert not ctrl.get_pending(cont.request_id).aborted
    time.sleep(0.05)
    ctrl.check_timeouts()
    assert ctrl.get_pending(cont.request_id).aborted
    assert cleanups


def test_unsupported_mode_rejections():
    base = _sample_continuation()
    policy = PDCapabilityPolicy()
    ctrl = PDHandoffController(policy=policy)

    ctrl.start_handoff(base.request_id, base.transfer_id)

    # projected input embeddings
    cont = dataclasses.replace(base, input_embeds_are_projected=True)
    with pytest.raises(PDCapabilityError, match="projected input embeddings"):
        ctrl.set_continuation(cont.request_id, cont)

    # speculative decoding
    cont = dataclasses.replace(base, speculative=True)
    with pytest.raises(PDCapabilityError, match="speculative decoding"):
        ctrl.set_continuation(cont.request_id, cont)

    # unknown multimodal schema
    cont = dataclasses.replace(
        base,
        multimodal_resume={"schema": "qwen3-omni-v1", "data": {"foo": 1}},
    )
    with pytest.raises(PDCapabilityError, match="multimodal resume schema"):
        ctrl.set_continuation(cont.request_id, cont)

    # grammar in sampling params
    cont = dataclasses.replace(
        base,
        sampling_params={"temperature": 0.0, "grammar": "..."},
    )
    with pytest.raises(PDCapabilityError, match="grammar"):
        ctrl.set_continuation(cont.request_id, cont)

    # TP size mismatch
    with pytest.raises(PDCapabilityError, match="TP size mismatch"):
        ctrl.set_continuation(cont.request_id, base, source_tp_size=2, target_tp_size=4)


def test_tp_metadata_duplication():
    cont = _sample_continuation(origin_input_ids=list(range(1000)))
    producer = PrefillContinuationProducer(tp_size=4)

    rank0 = producer.prepare_rank_metadata(cont, 0)
    assert rank0["pd_continuation_present"] is True
    assert isinstance(rank0["pd_continuation"], bytes)

    non_rank0_total = 0
    for rank in range(1, 4):
        meta = producer.prepare_rank_metadata(cont, rank)
        assert meta["pd_continuation_present"] is False
        assert "pd_continuation" not in meta
        non_rank0_total += len(msgspec.msgpack.encode(meta))

    # The rank-0 payload carries the continuation; the others do not duplicate it.
    assert len(rank0["pd_continuation"]) > non_rank0_total


def test_continuation_aware_kv_receiver():
    inner = _FakeReceiver()
    admitted: list[PendingHandoff] = []
    ctrl = PDHandoffController(admit_callback=lambda p: admitted.append(p))
    receiver = ContinuationAwareKVReceiver(
        inner=inner,
        controller=ctrl,
        source_tp_size=1,
        target_tp_size=1,
        is_local=True,
    )

    cont = _sample_continuation()
    producer = PrefillContinuationProducer(tp_size=1)
    meta = producer.prepare_rank_metadata(cont, 0)
    layout = KVPoolLayout(
        layout_id="l1",
        page_size=16,
        buffers=(KVBufferSpec(name="k", bytes_per_page=8192),),
    )
    prepare = KVTransferPrepareMessage(
        request_id=cont.request_id,
        transfer_id=cont.transfer_id,
        from_stage="prefill",
        to_stage="decode",
        source_pool_id="src",
        target_pool_id="dst",
        source_page_indices=(0, 1),
        source_layout=layout,
        metadata=meta,
    )

    destination = receiver.reserve(prepare)
    assert len(inner.reserves) == 1
    assert ctrl.get_pending(cont.request_id).continuation is not None

    receiver.commit(prepare, destination)
    assert len(inner.commits) == 1
    assert len(admitted) == 1
    assert admitted[0].continuation.request_id == cont.request_id


def test_continuation_aware_kv_receiver_invalid_continuation():
    inner = _FakeReceiver()
    ctrl = PDHandoffController()
    receiver = ContinuationAwareKVReceiver(inner=inner, controller=ctrl)

    prepare = KVTransferPrepareMessage(
        request_id="r",
        transfer_id="t",
        from_stage="prefill",
        to_stage="decode",
        source_pool_id="src",
        target_pool_id="dst",
        source_page_indices=(0,),
        source_layout=KVPoolLayout(
            layout_id="l1",
            page_size=16,
            buffers=(KVBufferSpec(name="k", bytes_per_page=8192),),
        ),
        metadata={"pd_continuation": b"not-msgpack"},
    )

    with pytest.raises(ContinuationSchemaError):
        receiver.reserve(prepare)


def test_continuation_aware_kv_receiver_abort():
    inner = _FakeReceiver()
    cleanups: list[tuple[PendingHandoff, str]] = []
    ctrl = PDHandoffController(cleanup_callback=lambda p, r: cleanups.append((p, r)))
    receiver = ContinuationAwareKVReceiver(inner=inner, controller=ctrl)

    cont = _sample_continuation()
    producer = PrefillContinuationProducer(tp_size=1)
    meta = producer.prepare_rank_metadata(cont, 0)
    prepare = KVTransferPrepareMessage(
        request_id=cont.request_id,
        transfer_id=cont.transfer_id,
        from_stage="prefill",
        to_stage="decode",
        source_pool_id="src",
        target_pool_id="dst",
        source_page_indices=(0,),
        source_layout=KVPoolLayout(
            layout_id="l1",
            page_size=16,
            buffers=(KVBufferSpec(name="k", bytes_per_page=8192),),
        ),
        metadata=meta,
    )

    receiver.reserve(prepare)
    err = RuntimeError("transfer failed")
    receiver.abort(prepare, None, err)

    assert len(inner.aborts) == 1
    assert ctrl.get_pending(cont.request_id).aborted
    assert len(cleanups) == 1
