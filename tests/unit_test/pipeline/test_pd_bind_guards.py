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
