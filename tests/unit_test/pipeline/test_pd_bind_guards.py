# SPDX-License-Identifier: Apache-2.0
"""`bind_pd_runtime` refuses the configurations the PD runtime does not support.

The three guards are the boundary of the current runtime. Nothing pins them, so
removing one on purpose produces no signal from the suite, and removing one by
accident produces none either.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def _scheduler(
    *,
    tp_size: int = 1,
    page_size: int = 1,
    disable_radix_cache: bool = True,
) -> SimpleNamespace:
    """A stand-in carrying only the attributes the guards read."""
    return SimpleNamespace(
        tp_size=tp_size,
        page_size=page_size,
        server_args=SimpleNamespace(disable_radix_cache=disable_radix_cache),
    )


def _bind(scheduler: SimpleNamespace) -> None:
    OmniScheduler.bind_pd_runtime(
        scheduler,
        stage_name="thinker_prefill",
        role="prefill",
        partner="thinker_decode",
    )


def test_tensor_parallel_is_refused() -> None:
    with pytest.raises(NotImplementedError, match="tp_size == 1 only"):
        _bind(_scheduler(tp_size=2))


def test_paged_kv_is_refused() -> None:
    with pytest.raises(NotImplementedError, match="page_size == 1 only"):
        _bind(_scheduler(page_size=16))


def test_radix_cache_enabled_is_refused() -> None:
    with pytest.raises(NotImplementedError, match="RadixCache disabled"):
        _bind(_scheduler(disable_radix_cache=False))


def test_a_supported_configuration_passes_every_guard() -> None:
    """The supported case clears all three, then fails on a later attribute.

    Without this the three tests above would still pass if a guard rejected
    everything, so this is what makes them a boundary rather than a wall.
    """
    with pytest.raises(AttributeError, match="token_to_kv_pool_allocator"):
        _bind(_scheduler())


def test_cuda_graph_is_not_among_the_guards() -> None:
    """PD does not require CUDA graph off, and turning it off is expensive.

    `_run_batch_prebuilt` returns an empty `GenerationBatchResult` and runs no
    forward whenever `batch.inner_idle_batch` is None, which it always is here:
    the only assignment is in `dp_attn.py`, reachable only through
    `require_mlp_sync`, and sglang-omni rejects DP attention at config time.
    So admission has no shape for a captured graph to miss and every step after
    it is `ForwardMode.DECODE`, which the graph covers.

    Measured on two H200s, decode CUDA graph off against on, everything else
    held: the pair's ceiling was 5.9-6.2 rps against 13.2-13.6 at
    `max_running_requests=64`, and 0.50 rps against at least 2.88 at a cap of
    4. Decode-card utilization sits at 10.9-16.5% with graphs off at every
    offered rate and cannot be driven higher, which is the signature of a
    launch-bound decode loop. The smaller the concurrency cap, the more the
    flag costs, because a smaller batch leaves less GPU work to hide the launch
    overhead behind.

    This test exists so that a future change cannot quietly add the flag to
    what PD enforces.
    """
    scheduler = _scheduler()
    scheduler.server_args.disable_cuda_graph = False

    # Clears every guard and fails later on an attribute the stand-in lacks,
    # which is what "not guarded" looks like from here. A new guard on the flag
    # would raise NotImplementedError instead and fail this test.
    with pytest.raises(AttributeError, match="token_to_kv_pool_allocator"):
        _bind(scheduler)
