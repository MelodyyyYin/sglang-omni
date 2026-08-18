# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
from array import array
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling import omni_scheduler as omni_scheduler_module
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.pd_continuation import (
    decode_continuation,
    encode_continuation,
)
from sglang_omni.scheduling.pd_kv_adapter import ReservedAllocation
from sglang_omni.scheduling.pd_runtime import (
    PDDecodeAdmission,
    continuation_from_req,
    req_from_continuation,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


class _ReqPool:
    def __init__(self):
        self.req_to_token = torch.zeros((4, 32), dtype=torch.int64)

    def alloc(self, reqs):
        for index, req in enumerate(reqs):
            req.req_pool_idx = index
        return torch.arange(len(reqs), dtype=torch.int64)

    def write(self, key, value):
        self.req_to_token[key] = value

    def free(self, req):
        req.req_pool_idx = None


class _FailingReqPool(_ReqPool):
    def __init__(self):
        super().__init__()
        self.freed = False

    def write(self, key, value):
        raise RuntimeError("mapping write failed")

    def free(self, req):
        self.freed = True
        super().free(req)


def _prefill_req() -> Req:
    sampling = SamplingParams(
        max_new_tokens=16,
        temperature=0.7,
        top_p=0.9,
        stop_token_ids={2},
        sampling_seed=17,
    )
    req = Req(
        rid="request-1",
        origin_input_text="",
        origin_input_ids=array("q", [10, 11, 12]),
        sampling_params=sampling,
        vocab_size=128,
        eos_token_ids={2},
    )
    req.output_ids.append(42)
    req.cached_tokens = 1
    req.cached_tokens_device = 1
    payload = StagePayload(
        request_id=req.rid,
        request=OmniRequest(inputs="hello", params={"stream": True}),
        data={"state": "generic"},
    )
    data = SGLangARRequestData(
        req=req,
        output_ids=req.output_ids,
        stage_payload=payload,
        input_embeds_are_projected=False,
    )
    req._omni_data = data
    return req


def test_continuation_reconstructs_req_and_transferred_mapping() -> None:
    source = _prefill_req()
    continuation = continuation_from_req(source, "transfer-1")
    continuation = decode_continuation(encode_continuation(continuation))
    pool = _ReqPool()
    allocation = ReservedAllocation(
        slots=torch.tensor([7, 8, 9], dtype=torch.int64),
        page_indices=(7, 8, 9),
        seq_len=3,
    )

    rebuilt = req_from_continuation(
        continuation,
        allocation,
        req_to_token_pool=pool,
    )

    assert rebuilt.rid == source.rid
    assert list(rebuilt.origin_input_ids) == [10, 11, 12]
    assert list(rebuilt.output_ids) == [42]
    assert rebuilt.sampling_params.sampling_seed == 17
    assert rebuilt.sampling_params.stop_token_ids == {2}
    assert rebuilt.prefix_indices.tolist() == [7, 8, 9]
    assert pool.req_to_token[0, :3].tolist() == [7, 8, 9]
    assert rebuilt._omni_data.stage_payload == source._omni_data.stage_payload


def test_reconstruction_failure_releases_request_pool_slot() -> None:
    continuation = continuation_from_req(_prefill_req(), "transfer-1")
    pool = _FailingReqPool()
    allocation = ReservedAllocation(
        slots=torch.tensor([7, 8, 9], dtype=torch.int64),
        page_indices=(7, 8, 9),
        seq_len=3,
    )

    with pytest.raises(RuntimeError, match="mapping write failed"):
        req_from_continuation(
            continuation,
            allocation,
            req_to_token_pool=pool,
        )

    assert pool.freed


def test_committed_decode_admission_enters_waiting_queue() -> None:
    source = _prefill_req()
    continuation = continuation_from_req(source, "transfer-1")
    pool = _ReqPool()
    allocation = ReservedAllocation(
        slots=torch.tensor([7, 8, 9], dtype=torch.int64),
        page_indices=(7, 8, 9),
        seq_len=3,
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler._pd_ready_queue = queue.SimpleQueue()
    scheduler._pd_ready_queue.put(PDDecodeAdmission(continuation, allocation))
    scheduler._aborted_request_ids = set()
    scheduler.req_to_token_pool = pool
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(free=lambda _: None)
    scheduler.waiting_queue = []
    scheduler.outbox = queue.Queue()

    scheduler._drain_pd_admissions()

    assert [req.rid for req in scheduler.waiting_queue] == ["request-1"]
    admitted = scheduler.outbox.get_nowait()
    assert admitted.type == "pd_admitted"
    assert admitted.request_id == "request-1"


def test_aborted_decode_admission_releases_transferred_slots() -> None:
    source = _prefill_req()
    continuation = continuation_from_req(source, "transfer-1")
    allocation = ReservedAllocation(
        slots=torch.tensor([7, 8, 9], dtype=torch.int64),
        page_indices=(7, 8, 9),
        seq_len=3,
    )
    freed = []
    scheduler = object.__new__(OmniScheduler)
    scheduler._pd_ready_queue = queue.SimpleQueue()
    scheduler._pd_ready_queue.put(PDDecodeAdmission(continuation, allocation))
    scheduler._aborted_request_ids = {"request-1"}
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(free=freed.append)
    scheduler.waiting_queue = []
    scheduler.outbox = queue.Queue()

    scheduler._drain_pd_admissions()

    assert scheduler.waiting_queue == []
    assert freed == [allocation.slots]
    assert scheduler.outbox.empty()


def test_prefill_handoff_reuses_request_mapping_and_detaches_batch() -> None:
    req = _prefill_req()
    pool = _ReqPool()
    pool.alloc([req])
    pool.write(
        (req.req_pool_idx, slice(0, 3)),
        torch.tensor([7, 8, 9], dtype=torch.int64),
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler.req_to_token_pool = pool
    scheduler.page_size = 1
    scheduler._pd_pool_id = "prefill:kv"
    scheduler._pd_partner = "decode"
    scheduler.tree_cache = object()
    scheduler.outbox = queue.Queue()
    batch = SimpleNamespace(reqs=[req])

    scheduler._queue_pd_prefill_handoffs(batch)

    assert batch.reqs == []
    handoff = scheduler.outbox.get_nowait()
    assert handoff.type == "pd_handoff"
    assert handoff.data.source_page_indices == (7, 8, 9)
    assert handoff.data.continuation.output_ids == [42]


def test_decode_scheduler_delegates_prebuilt_admission(monkeypatch) -> None:
    running = object()
    batch = object()
    calls = []

    def get_next(_scheduler, running_batch):
        calls.append(running_batch)
        return SimpleNamespace(running_batch=running, batch_to_run=batch)

    monkeypatch.setattr(
        omni_scheduler_module._Upstream,
        "get_next_disagg_decode_batch_to_run",
        get_next,
    )
    scheduler = object.__new__(OmniScheduler)
    scheduler._pd_role = "decode"
    scheduler._pd_ready_queue = queue.SimpleQueue()
    scheduler.running_batch = object()

    assert scheduler.get_next_batch_to_run() is batch
    assert calls
    assert scheduler.running_batch is running
