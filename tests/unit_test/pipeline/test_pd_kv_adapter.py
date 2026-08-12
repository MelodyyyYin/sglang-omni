# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.comm import KVPageDestination, KVPool
from sglang_omni.proto import KVTransferPrepareMessage
from sglang_omni.scheduling.pd_kv_adapter import (
    AllocatorKVReceiver,
    build_kv_pool,
    resolve_page_indices,
)


class _FakeKVPool:
    """Minimal stand-in for an SGLang token_to_kv_pool."""

    def __init__(self, *, page_size: int, item_bytes: int, pages: int, buffers: int):
        self.page_size = page_size
        self._tensors = [
            torch.zeros((pages, item_bytes), dtype=torch.uint8)
            for _ in range(buffers)
        ]
        self._item_bytes = item_bytes

    def _pd_registerable_tensors(self):
        return tuple(self._tensors)

    def get_contiguous_buf_infos(self):
        ptrs = [t.data_ptr() for t in self._tensors]
        lens = [t.numel() for t in self._tensors]
        item_lens = [self._item_bytes for _ in self._tensors]
        return ptrs, lens, item_lens


class _FakeReqToTokenPool:
    def __init__(self, req_to_token: torch.Tensor):
        self.req_to_token = req_to_token


def test_build_kv_pool_wraps_registerable_tensors() -> None:
    fake = _FakeKVPool(page_size=1, item_bytes=8, pages=16, buffers=4)
    pool = build_kv_pool(fake, pool_id="prefill_kv", layout_id="qwen3-omni-v1")

    assert isinstance(pool, KVPool)
    assert pool.pool_id == "prefill_kv"
    assert pool.layout_id == "qwen3-omni-v1"
    assert pool.page_size == 1
    assert len(pool.buffers) == 4
    assert all(b.bytes_per_page == 8 for b in pool.buffers)
    assert all(b.page_count == 16 for b in pool.buffers)


def test_build_kv_pool_layout_matches_across_stages() -> None:
    src = build_kv_pool(
        _FakeKVPool(page_size=1, item_bytes=8, pages=16, buffers=4),
        pool_id="prefill_kv",
        layout_id="qwen3-omni-v1",
    )
    dst = build_kv_pool(
        _FakeKVPool(page_size=1, item_bytes=8, pages=32, buffers=4),
        pool_id="decode_kv",
        layout_id="qwen3-omni-v1",
    )
    # Layout equality is what the comm layer checks before a transfer.
    assert src.layout == dst.layout


def test_resolve_page_indices_page_size_one() -> None:
    req_to_token = torch.zeros((3, 10), dtype=torch.int32)
    req_to_token[1, :5] = torch.tensor([7, 8, 9, 3, 4], dtype=torch.int32)
    pool = _FakeReqToTokenPool(req_to_token)

    pages = resolve_page_indices(pool, req_pool_idx=1, seq_len=5, page_size=1)
    assert pages == (7, 8, 9, 3, 4)


def test_resolve_page_indices_paged_dedup_preserves_order() -> None:
    req_to_token = torch.zeros((2, 10), dtype=torch.int32)
    # page_size=4: slots 0..3 -> page 0, 4..7 -> page 1, 8 -> page 2
    req_to_token[0, :6] = torch.tensor([0, 1, 2, 4, 5, 8], dtype=torch.int32)
    pool = _FakeReqToTokenPool(req_to_token)

    pages = resolve_page_indices(pool, req_pool_idx=0, seq_len=6, page_size=4)
    assert pages == (0, 1, 2)


def test_resolve_page_indices_rejects_bad_args() -> None:
    pool = _FakeReqToTokenPool(torch.zeros((1, 4), dtype=torch.int32))
    with pytest.raises(ValueError):
        resolve_page_indices(pool, req_pool_idx=0, seq_len=0, page_size=1)
    with pytest.raises(ValueError):
        resolve_page_indices(pool, req_pool_idx=0, seq_len=2, page_size=0)


class _FakeAllocator:
    """Contiguous bump allocator mimicking SGLang's alloc/free contract."""

    def __init__(self, capacity: int):
        self._free = list(range(capacity))

    def available_size(self) -> int:
        return len(self._free)

    def alloc(self, need_size: int):
        if len(self._free) < need_size:
            return None
        taken = self._free[:need_size]
        self._free = self._free[need_size:]
        return torch.tensor(taken, dtype=torch.int64)

    def free(self, free_index: torch.Tensor) -> None:
        self._free.extend(int(i) for i in free_index.tolist())


def _prepare(request_id: str, pages: int, *, pool_id: str = "decode_kv"):
    return KVTransferPrepareMessage(
        request_id=request_id,
        transfer_id=f"{request_id}:t",
        from_stage="thinker_prefill",
        to_stage="thinker_decode",
        source_pool_id="prefill_kv",
        target_pool_id=pool_id,
        source_page_indices=tuple(range(pages)),
        source_layout=None,
    )


def test_receiver_reserve_commit_hands_off_allocation() -> None:
    alloc = _FakeAllocator(16)
    committed: list[tuple[str, tuple[int, ...], int]] = []
    receiver = AllocatorKVReceiver(
        pool_id="decode_kv",
        token_to_kv_pool_allocator=alloc,
        page_size=1,
        on_commit=lambda rid, slots, pages, seq_len: committed.append(
            (rid, pages, seq_len)
        ),
    )

    prep = _prepare("req-1", 5)
    dest = receiver.reserve(prep)
    assert isinstance(dest, KVPageDestination)
    assert len(dest.page_indices) == 5
    assert alloc.available_size() == 11  # 16 - 5

    receiver.commit(prep, dest)
    assert committed == [("req-1", dest.page_indices, 5)]
    # Commit hands ownership to the request lifecycle: slots NOT returned here.
    assert alloc.available_size() == 11


def test_receiver_abort_frees_allocation() -> None:
    alloc = _FakeAllocator(8)
    receiver = AllocatorKVReceiver(
        pool_id="decode_kv", token_to_kv_pool_allocator=alloc, page_size=1
    )
    prep = _prepare("req-2", 3)
    dest = receiver.reserve(prep)
    assert alloc.available_size() == 5

    receiver.abort(prep, dest, RuntimeError("boom"))
    assert alloc.available_size() == 8  # freed back


def test_receiver_rejects_exhausted_pool() -> None:
    alloc = _FakeAllocator(2)
    receiver = AllocatorKVReceiver(
        pool_id="decode_kv", token_to_kv_pool_allocator=alloc, page_size=1
    )
    with pytest.raises(RuntimeError, match="exhausted"):
        receiver.reserve(_prepare("req-3", 5))


def test_receiver_rejects_wrong_pool_and_duplicate() -> None:
    alloc = _FakeAllocator(16)
    receiver = AllocatorKVReceiver(
        pool_id="decode_kv", token_to_kv_pool_allocator=alloc, page_size=1
    )
    with pytest.raises(ValueError):
        receiver.reserve(_prepare("req-4", 2, pool_id="other_pool"))

    prep = _prepare("req-5", 2)
    receiver.reserve(prep)
    with pytest.raises(RuntimeError, match="duplicate"):
        receiver.reserve(prep)


def test_receiver_page_size_gt_one_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        AllocatorKVReceiver(
            pool_id="decode_kv",
            token_to_kv_pool_allocator=_FakeAllocator(4),
            page_size=4,
        )
