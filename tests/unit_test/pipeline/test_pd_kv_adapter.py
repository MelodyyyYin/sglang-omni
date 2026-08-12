# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.comm import KVPool
from sglang_omni.scheduling.pd_kv_adapter import build_kv_pool, resolve_page_indices


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
