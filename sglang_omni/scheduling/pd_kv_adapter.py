# SPDX-License-Identifier: Apache-2.0
"""Model-neutral adapters bridging SGLang KV caches to the comm KV-transfer API.

These helpers turn an SGLang ``token_to_kv_pool`` into a
:class:`sglang_omni.comm.KVPool` (so the communication layer can export/import
its pages over CUDA IPC) and resolve the page indices a finished request
occupies. They are intentionally model-agnostic: qwen3_omni and ming_omni PD
wiring both build on them.

The contracts they satisfy are defined by PR #1315 (paged KV transfer
infrastructure) and PR #1382 (rank-to-rank TP KV transfers):

* the source (prefill) stage registers a ``KVPool`` and calls
  ``send_kv_pages`` with the page indices a request occupies;
* the destination (decode) stage registers a ``KVPool`` plus a
  :class:`sglang_omni.comm.KVReceiver` that reserves/commits/aborts pages
  through its own allocator.
"""

from __future__ import annotations

import logging
from typing import Any

from sglang_omni.comm import KVBufferRegion, KVPool

logger = logging.getLogger(__name__)


def build_kv_pool(
    token_to_kv_pool: Any,
    *,
    pool_id: str,
    layout_id: str,
) -> KVPool:
    """Wrap an SGLang ``token_to_kv_pool`` as a comm :class:`KVPool`.

    Uses SGLang's own PD-disaggregation descriptor hook
    ``get_contiguous_buf_infos()`` -> ``(ptrs, lens, item_lens)`` together with
    ``_pd_registerable_tensors()`` so the exported byte views alias exactly the
    tensors SGLang's ``transfer_kv_per_layer_mla`` kernel copies.

    ``layout_id`` must be identical on the prefill and decode stages; the comm
    layer rejects transfers whose source/destination layouts differ.
    """

    registerable = _pd_registerable_tensors(token_to_kv_pool)
    _, _, item_lens = token_to_kv_pool.get_contiguous_buf_infos()
    if len(registerable) != len(item_lens):
        raise ValueError(
            "SGLang KV pool exposed "
            f"{len(registerable)} registerable tensors but {len(item_lens)} "
            "item-length descriptors; cannot build a comm KVPool"
        )

    page_size = int(token_to_kv_pool.page_size)
    buffers = tuple(
        KVBufferRegion(
            name=f"kv_buffer.{index}",
            tensor=tensor,
            bytes_per_page=int(item_len),
        )
        for index, (tensor, item_len) in enumerate(zip(registerable, item_lens))
    )
    if not buffers:
        raise ValueError(f"SGLang KV pool {pool_id!r} exposed no buffers")

    pool = KVPool(
        pool_id=pool_id,
        layout_id=layout_id,
        page_size=page_size,
        buffers=buffers,
    )
    logger.info(
        "pd_kv build_kv_pool pool_id=%s layout_id=%s page_size=%d buffers=%d "
        "bytes_per_page=%s device=%s",
        pool_id,
        layout_id,
        page_size,
        len(buffers),
        tuple(b.bytes_per_page for b in buffers[:1]) if buffers else (),
        pool.device,
    )
    return pool


def _pd_registerable_tensors(token_to_kv_pool: Any) -> tuple[Any, ...]:
    """Return the per-buffer tensors SGLang exposes for PD transfer.

    Newer SGLang builds expose ``_pd_registerable_tensors()`` directly. Fall
    back to per-layer key/value buffers for builds that only expose the
    getter API.
    """

    getter = getattr(token_to_kv_pool, "_pd_registerable_tensors", None)
    if callable(getter):
        return tuple(getter())

    layer_num = int(getattr(token_to_kv_pool, "layer_num"))
    tensors: list[Any] = []
    for layer_id in range(layer_num):
        tensors.append(token_to_kv_pool.get_key_buffer(layer_id))
        tensors.append(token_to_kv_pool.get_value_buffer(layer_id))
    return tuple(tensors)


def resolve_page_indices(
    req_to_token_pool: Any,
    *,
    req_pool_idx: int,
    seq_len: int,
    page_size: int,
) -> tuple[int, ...]:
    """Resolve the KV page indices a request currently occupies.

    ``req_to_token[req_pool_idx, :seq_len]`` holds the per-token KV slot
    indices. For ``page_size == 1`` those slots are the page indices; for
    paged layouts we take the distinct page each slot falls in, preserving
    first-seen order.
    """

    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    slots = req_to_token_pool.req_to_token[req_pool_idx, :seq_len]
    slot_indices = slots.tolist()

    if page_size == 1:
        return tuple(int(slot) for slot in slot_indices)

    pages: list[int] = []
    seen: set[int] = set()
    for slot in slot_indices:
        page = int(slot) // page_size
        if page not in seen:
            seen.add(page)
            pages.append(page)
    return tuple(pages)
