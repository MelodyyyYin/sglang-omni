# SPDX-License-Identifier: Apache-2.0
"""One lock across the two threads a PD Decode half allocates from.

The allocator's alloc reads the free list, slices it, and writes the
remainder back, with no lock of its own. On a Decode half the scheduler
thread and the comm event loop both call it, and interleaving there hands
the same slots to both.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from sglang_omni.scheduling.pd_alloc_lock import LockedKVAllocator


class _RacyAllocator:
    """An allocator with the same read-modify-write shape as upstream's."""

    def __init__(self, size: int) -> None:
        self.free_pages = list(range(size))
        self.page_size = 1
        self.handed_out: list[int] = []
        self.free_group: list[list[int]] = []
        self.is_not_in_free_group = True

    def alloc(self, need_size: int):
        taken = self.free_pages[:need_size]
        # A thread switch here is what produces the overlap in production.
        threading.current_thread()
        self.free_pages = self.free_pages[need_size:]
        self.handed_out.extend(taken)
        return taken

    def free(self, index) -> None:
        if self.is_not_in_free_group:
            self.free_pages.extend(index)
        else:
            self.free_group.append(index)

    def free_group_begin(self) -> None:
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self) -> None:
        self.is_not_in_free_group = True
        for indices in self.free_group:
            self.free(indices)
        self.free_group = []

    def available_size(self) -> int:
        return len(self.free_pages)


class _DeterministicAllocFreeRace(_RacyAllocator):
    """Expose the stale write that loses a concurrent cache free."""

    def __init__(self) -> None:
        super().__init__(2)
        self.alloc_read = threading.Event()
        self.finish_alloc = threading.Event()

    def alloc(self, need_size: int):
        taken = self.free_pages[:need_size]
        remaining = self.free_pages[need_size:]
        self.alloc_read.set()
        assert self.finish_alloc.wait(timeout=1)
        self.free_pages = remaining
        self.handed_out.extend(taken)
        return taken


def test_two_threads_never_receive_the_same_slot() -> None:
    inner = _RacyAllocator(4096)
    allocator = LockedKVAllocator(inner)
    seen: list[int] = []
    lock = threading.Lock()

    def take() -> None:
        for _ in range(64):
            got = allocator.alloc(4)
            with lock:
                seen.extend(got)

    threads = [threading.Thread(target=take) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seen) == len(set(seen))


def test_everything_else_is_delegated() -> None:
    inner = _RacyAllocator(16)
    allocator = LockedKVAllocator(inner)

    assert allocator.available_size() == 16
    assert allocator.page_size == 1


def test_a_write_reaches_the_wrapped_allocator() -> None:
    """Upstream code sets attributes on the allocator it was handed."""
    inner = _RacyAllocator(16)
    allocator = LockedKVAllocator(inner)

    allocator.page_size = 4

    assert inner.page_size == 4


def test_freeing_returns_the_slots() -> None:
    inner = _RacyAllocator(8)
    allocator = LockedKVAllocator(inner)

    taken = allocator.alloc(8)
    assert allocator.available_size() == 0

    allocator.free(taken)
    assert allocator.available_size() == 8


def test_comm_alloc_and_cached_alias_free_share_one_domain() -> None:
    """The free must not interleave between alloc's free-list read and write."""

    inner = _DeterministicAllocFreeRace()
    allocator = LockedKVAllocator(inner)
    comm_receiver = SimpleNamespace(allocator=allocator)
    tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
    allocated = []

    alloc_thread = threading.Thread(
        target=lambda: allocated.extend(comm_receiver.allocator.alloc(1))
    )
    free_thread = threading.Thread(
        target=lambda: tree_cache.token_to_kv_pool_allocator.free([2])
    )
    alloc_thread.start()
    assert inner.alloc_read.wait(timeout=1)
    free_thread.start()
    assert free_thread.is_alive()
    inner.finish_alloc.set()
    alloc_thread.join(timeout=1)
    free_thread.join(timeout=1)

    assert allocated == [0]
    assert sorted(inner.free_pages) == [1, 2]
    assert len(inner.free_pages) == len(set(inner.free_pages))


def test_free_group_holds_the_domain_until_group_end() -> None:
    inner = _RacyAllocator(2)
    allocator = LockedKVAllocator(inner)
    allocated: list[int] = []

    allocator.free_group_begin()
    allocator.free([2])
    thread = threading.Thread(target=lambda: allocated.extend(allocator.alloc(1)))
    thread.start()
    assert thread.is_alive()

    allocator.free_group_end()
    thread.join(timeout=1)

    assert allocated == [0]
    assert sorted(inner.free_pages) == [1, 2]
    assert len(inner.free_pages) == len(set(inner.free_pages))
