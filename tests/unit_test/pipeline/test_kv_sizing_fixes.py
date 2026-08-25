# SPDX-License-Identifier: Apache-2.0
"""KV budget accounting: attribution failure, and clamping by free memory."""

from __future__ import annotations

import pytest

from sglang_omni.utils.gpu_memory import calculate_stage_budget_available_bytes

_GIB = 1024**3


def _budget(**kwargs) -> int:
    base = {
        "total_memory_bytes": 140 * _GIB,
        "accounted_memory_bytes": 57 * _GIB,
        "memory_fraction": 0.87,
    }
    base.update(kwargs)
    return calculate_stage_budget_available_bytes(**base)


def test_without_a_free_reading_the_budget_is_unchanged() -> None:
    """A caller that cannot read free memory keeps the old behaviour."""
    assert _budget() == int(140 * _GIB * 0.87) - 57 * _GIB


def test_a_co_tenant_no_longer_gets_over_committed() -> None:
    """The stage's own usage is not the whole card's usage.

    Measured: on a card holding 34 GiB of another process, a stage at fraction
    0.87 budgeted 122.49 GiB and ran out of memory at launch.
    """
    unclamped = _budget()
    clamped = _budget(free_memory_bytes=20 * _GIB)

    assert clamped == 20 * _GIB
    assert clamped < unclamped


def test_a_reserve_is_held_back_from_the_clamp() -> None:
    assert _budget(free_memory_bytes=20 * _GIB, reserve_bytes=4 * _GIB) == 16 * _GIB


def test_ample_free_memory_does_not_shrink_the_budget() -> None:
    assert _budget(free_memory_bytes=130 * _GIB) == _budget()


def test_a_budget_leaving_nothing_still_raises() -> None:
    with pytest.raises(RuntimeError, match="no KV-cache headroom"):
        _budget(memory_fraction=0.4)
