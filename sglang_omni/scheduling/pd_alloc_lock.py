# SPDX-License-Identifier: Apache-2.0
"""Serialize allocator access shared by PD scheduler and comm threads."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_pd_allocator_sync: ContextVar[bool] = ContextVar(
    "sglang_omni_pd_allocator_sync", default=False
)


@contextmanager
def pd_allocator_sync_scope(enabled: bool) -> Iterator[None]:
    """Select allocator synchronization while a stage factory is constructed."""

    token = _pd_allocator_sync.set(enabled)
    try:
        yield
    finally:
        _pd_allocator_sync.reset(token)


def synchronize_pd_allocator(allocator: Any) -> Any:
    """Wrap *allocator* only for a PD stage being constructed."""

    if not _pd_allocator_sync.get():
        return allocator
    if isinstance(allocator, LockedKVAllocator):
        return allocator
    return LockedKVAllocator(allocator)


class LockedKVAllocator:
    """Delegate the SGLang allocator interface through one reentrant lock."""

    def __init__(self, inner: Any) -> None:
        # Note (Audrey Zheng): set through __dict__ because __setattr__ below
        # forwards to the wrapped allocator.
        self.__dict__["_inner"] = inner
        self.__dict__["_alloc_lock"] = threading.RLock()

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        with self.__dict__["_alloc_lock"]:
            return getattr(self.__dict__["_inner"], name)(*args, **kwargs)

    def alloc(self, need_size: int) -> Any:
        return self._call("alloc", need_size)

    def free(self, free_index: Any) -> Any:
        return self._call("free", free_index)

    def alloc_extend(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("alloc_extend", *args, **kwargs)

    def alloc_decode(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("alloc_decode", *args, **kwargs)

    def available_size(self) -> Any:
        return self._call("available_size")

    def backup_state(self) -> Any:
        return self._call("backup_state")

    def restore_state(self, state: Any) -> Any:
        return self._call("restore_state", state)

    def merge_and_sort_free(self) -> Any:
        return self._call("merge_and_sort_free")

    def clear(self) -> Any:
        return self._call("clear")

    def resize(self, config: Any) -> Any:
        return self._call("resize", config)

    def free_group_begin(self) -> Any:
        lock = self.__dict__["_alloc_lock"]
        lock.acquire()
        try:
            return self.__dict__["_inner"].free_group_begin()
        except BaseException:
            lock.release()
            raise

    def free_group_end(self) -> Any:
        try:
            return self.__dict__["_inner"].free_group_end()
        finally:
            self.__dict__["_alloc_lock"].release()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_inner"], name)

    def __setattr__(self, name: str, value: Any) -> None:
        with self.__dict__["_alloc_lock"]:
            setattr(self.__dict__["_inner"], name, value)

    def __repr__(self) -> str:
        return f"LockedKVAllocator({self.__dict__['_inner']!r})"
