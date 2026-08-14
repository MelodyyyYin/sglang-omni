# SPDX-License-Identifier: Apache-2.0
"""Tests for the generic Prefill-Decode continuation handoff."""

from __future__ import annotations

import asyncio
import dataclasses
import time

import msgspec
import pytest

from sglang_omni.comm.kv_transfer import KVPageDestination, KVTransferPrepareMessage
from sglang_omni.proto import KVBufferSpec, KVPoolLayout
from sglang_omni.scheduling.pd_continuation import (
    ContinuationAwareKVReceiver,
    ContinuationSchemaError,
    DecodeContinuation,
    PDCapabilityError,
    PDHandoffController,
    PendingHandoff,
    PrefillContinuationProducer,
    decode_continuation,
    encode_continuation,
    validate_continuation,
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
    def __init__(self) -> None:
        self.reserves: list[KVTransferPrepareMessage] = []
        self.commits: list[tuple[KVTransferPrepareMessage, KVPageDestination]] = []
        self.aborts: list[
            tuple[KVTransferPrepareMessage, KVPageDestination | None, BaseException]
        ] = []

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
    assert decode_continuation(encode_continuation(cont)) == cont


def test_schema_version_rejection():
    cont = _sample_continuation()
    d = cont.to_dict()
    d["version"] = "pd-continuation-v0"
    with pytest.raises(ContinuationSchemaError):
        decode_continuation(msgspec.msgpack.encode(d))


def test_unknown_key_rejection():
    cont = _sample_continuation()
    d = cont.to_dict()
    d["extra_field"] = 123
    with pytest.raises(ContinuationSchemaError):
        decode_continuation(msgspec.msgpack.encode(d))


def test_exactly_once_rank_ready():
    rank_ready: list[PendingHandoff] = []
    ctrl = PDHandoffController(rank_ready_callback=rank_ready.append)
    cont = _sample_continuation()

    ctrl.start_handoff(cont.request_id, cont.transfer_id)
    ctrl.set_continuation(cont.request_id, cont)
    assert not ctrl.is_rank_ready(cont.request_id)

    ctrl.set_kv_committed(cont.request_id)
    assert ctrl.is_rank_ready(cont.request_id)
    assert len(rank_ready) == 1
    assert rank_ready[0].continuation is cont

    ctrl.set_kv_committed(cont.request_id)
    ctrl.set_continuation(cont.request_id, cont)
    assert len(rank_ready) == 1


def test_kv_first_then_continuation_still_becomes_rank_ready():
    rank_ready: list[PendingHandoff] = []
    ctrl = PDHandoffController(rank_ready_callback=rank_ready.append)
    cont = _sample_continuation()

    ctrl.start_handoff(cont.request_id, cont.transfer_id)
    ctrl.set_kv_committed(cont.request_id)
    assert not ctrl.is_rank_ready(cont.request_id)
    ctrl.set_continuation(cont.request_id, cont)
    assert ctrl.is_rank_ready(cont.request_id)
    assert len(rank_ready) == 1


def test_ack_not_required_for_rank_ready():
    """Rank-ready depends on continuation and KV commit, not ACK."""
    rank_ready: list[PendingHandoff] = []
    ctrl = PDHandoffController(rank_ready_callback=rank_ready.append)
    cont = _sample_continuation()

    ctrl.start_handoff(cont.request_id, cont.transfer_id)
    ctrl.set_continuation(cont.request_id, cont)
    ctrl.set_kv_committed(cont.request_id)
    assert rank_ready


def test_duplicate_and_stale_messages():
    rank_ready: list[PendingHandoff] = []
    ctrl = PDHandoffController(rank_ready_callback=rank_ready.append)
    cont = _sample_continuation()

    ctrl.start_handoff(cont.request_id, cont.transfer_id)
    ctrl.set_continuation(cont.request_id, cont)
    ctrl.set_kv_committed(cont.request_id)

    with pytest.raises(KeyError):
        ctrl.set_continuation("unknown-req", cont)

    ctrl.set_kv_committed(cont.request_id)
    assert len(rank_ready) == 1


def test_abort_cleanup_before_rank_ready():
    cleanups: list[tuple[PendingHandoff, str]] = []
    ctrl = PDHandoffController(cleanup_callback=lambda p, r: cleanups.append((p, r)))
    cont = _sample_continuation()

    ctrl.start_handoff(cont.request_id, cont.transfer_id)
    ctrl.set_continuation(cont.request_id, cont)
    ctrl.abort(cont.request_id, reason="user-cancel")

    assert ctrl.get_pending(cont.request_id).aborted
    assert len(cleanups) == 1
    assert cleanups[0][1] == "user-cancel"

    ctrl.set_kv_committed(cont.request_id)
    ctrl.abort(cont.request_id, reason="again")
    assert len(cleanups) == 1


def test_abort_after_commit_before_rank_ready():
    cleanups: list[tuple[PendingHandoff, str]] = []
    ctrl = PDHandoffController(cleanup_callback=lambda p, r: cleanups.append((p, r)))
    cont = _sample_continuation()

    ctrl.start_handoff(cont.request_id, cont.transfer_id)
    ctrl.set_kv_committed(cont.request_id)
    ctrl.abort(cont.request_id, reason="dropped")
    ctrl.set_continuation(cont.request_id, cont)

    assert ctrl.get_pending(cont.request_id).aborted
    assert not ctrl.is_rank_ready(cont.request_id)
    assert len(cleanups) == 1


def test_abort_after_rank_ready_is_ignored():
    rank_ready: list[PendingHandoff] = []
    cleanups: list[tuple[PendingHandoff, str]] = []
    ctrl = PDHandoffController(
        rank_ready_callback=rank_ready.append,
        cleanup_callback=lambda p, r: cleanups.append((p, r)),
    )
    cont = _sample_continuation()

    ctrl.start_handoff(cont.request_id, cont.transfer_id)
    ctrl.set_continuation(cont.request_id, cont)
    ctrl.set_kv_committed(cont.request_id)
    assert rank_ready

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
    assert len(cleanups) == 1
    assert cleanups[0][1] == "timeout"


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


def _prepare(cont: DecodeContinuation, metadata: dict) -> KVTransferPrepareMessage:
    return KVTransferPrepareMessage(
        request_id=cont.request_id,
        transfer_id=cont.transfer_id,
        from_stage="prefill",
        to_stage="decode",
        source_pool_id="src",
        target_pool_id="dst",
        source_page_indices=(0, 1),
        source_layout=KVPoolLayout(
            layout_id="l1",
            page_size=16,
            buffers=(KVBufferSpec(name="k", bytes_per_page=8192),),
        ),
        metadata=metadata,
    )


def test_continuation_aware_kv_receiver_rank0():
    inner = _FakeReceiver()
    rank_ready: list[PendingHandoff] = []
    ctrl = PDHandoffController(rank_ready_callback=rank_ready.append)
    receiver = ContinuationAwareKVReceiver(inner=inner, controller=ctrl)

    cont = _sample_continuation()
    meta = PrefillContinuationProducer(tp_size=1).prepare_rank_metadata(cont, 0)
    prepare = _prepare(cont, meta)

    destination = receiver.reserve(prepare)
    assert len(inner.reserves) == 1
    assert ctrl.get_pending(cont.request_id).continuation is not None

    receiver.commit(prepare, destination)
    assert len(inner.commits) == 1
    assert len(rank_ready) == 1
    assert rank_ready[0].continuation.request_id == cont.request_id


def test_continuation_aware_kv_receiver_non_rank0():
    """Non-rank-0 TP shards receive a presence flag, not the full continuation."""
    inner = _FakeReceiver()
    rank_ready: list[PendingHandoff] = []
    ctrl = PDHandoffController(rank_ready_callback=rank_ready.append)
    receiver = ContinuationAwareKVReceiver(inner=inner, controller=ctrl)

    cont = _sample_continuation()
    meta = PrefillContinuationProducer(tp_size=2).prepare_rank_metadata(cont, 1)
    assert meta == {"pd_continuation_present": False}
    prepare = _prepare(cont, meta)

    destination = receiver.reserve(prepare)
    assert len(inner.reserves) == 1
    assert ctrl.get_pending(cont.request_id).continuation is None
    assert ctrl.get_pending(cont.request_id).continuation_expected is False

    receiver.commit(prepare, destination)
    assert len(inner.commits) == 1
    assert len(rank_ready) == 1
    assert rank_ready[0].continuation is None


def test_continuation_aware_kv_receiver_invalid_continuation():
    inner = _FakeReceiver()
    receiver = ContinuationAwareKVReceiver(
        inner=inner,
        controller=PDHandoffController(),
    )
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


def test_continuation_aware_kv_receiver_missing_continuation():
    inner = _FakeReceiver()
    receiver = ContinuationAwareKVReceiver(
        inner=inner,
        controller=PDHandoffController(),
    )
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
        metadata={"pd_continuation_present": True},
    )

    with pytest.raises(ContinuationSchemaError, match="pd_continuation required"):
        receiver.reserve(prepare)


def test_continuation_aware_kv_receiver_abort():
    inner = _FakeReceiver()
    cleanups: list[tuple[PendingHandoff, str]] = []
    ctrl = PDHandoffController(cleanup_callback=lambda p, r: cleanups.append((p, r)))
    receiver = ContinuationAwareKVReceiver(inner=inner, controller=ctrl)

    cont = _sample_continuation()
    meta = PrefillContinuationProducer(tp_size=1).prepare_rank_metadata(cont, 0)
    prepare = _prepare(cont, meta)

    receiver.reserve(prepare)
    err = RuntimeError("transfer failed")
    receiver.abort(prepare, None, err)

    assert len(inner.aborts) == 1
    assert ctrl.get_pending(cont.request_id).aborted
    assert len(cleanups) == 1


def test_tp_metadata_duplication():
    cont = _sample_continuation(origin_input_ids=list(range(1000)))
    producer = PrefillContinuationProducer(tp_size=4)

    rank0 = producer.prepare_rank_metadata(cont, 0)
    assert rank0["pd_continuation_present"] is True
    assert isinstance(rank0["pd_continuation"], bytes)

    for rank in range(1, 4):
        meta = producer.prepare_rank_metadata(cont, rank)
        assert meta == {"pd_continuation_present": False}


def test_unsupported_mode_rejections():
    base = _sample_continuation()

    with pytest.raises(PDCapabilityError, match="projected input embeddings"):
        validate_continuation(
            dataclasses.replace(base, input_embeds_are_projected=True),
            source_tp_size=1,
            target_tp_size=1,
        )

    with pytest.raises(PDCapabilityError, match="speculative decoding"):
        validate_continuation(
            dataclasses.replace(base, speculative=True),
            source_tp_size=1,
            target_tp_size=1,
        )

    with pytest.raises(PDCapabilityError, match="multimodal resume"):
        validate_continuation(
            dataclasses.replace(
                base,
                multimodal_resume={"schema": "qwen3-omni-v1", "data": {"foo": 1}},
            ),
            source_tp_size=1,
            target_tp_size=1,
        )

    with pytest.raises(PDCapabilityError, match="grammar"):
        validate_continuation(
            dataclasses.replace(
                base,
                sampling_params={"temperature": 0.0, "grammar": "..."},
            ),
            source_tp_size=1,
            target_tp_size=1,
        )

    with pytest.raises(PDCapabilityError, match="custom logit processor"):
        validate_continuation(
            dataclasses.replace(base, custom_logit_processor="my_processor"),
            source_tp_size=1,
            target_tp_size=1,
        )

    with pytest.raises(PDCapabilityError, match="TP size mismatch"):
        validate_continuation(base, source_tp_size=2, target_tp_size=4)

    with pytest.raises(PDCapabilityError, match="cross-node"):
        validate_continuation(base, source_tp_size=1, target_tp_size=1, is_local=False)
