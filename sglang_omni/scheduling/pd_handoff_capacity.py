# SPDX-License-Identifier: Apache-2.0
"""Thread-safe ownership budget for Prefill KV handoffs."""

from __future__ import annotations

import threading


class HandoffCapacity:
    """Bound both the number of leases and the prompt tokens they protect."""

    def __init__(self, *, max_requests: int, max_tokens: int) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_requests = int(max_requests)
        self.max_tokens = int(max_tokens)
        self._requests = 0
        self._tokens = 0
        self._lock = threading.Lock()

    def can_ever_fit(self, tokens: int) -> bool:
        return 0 < int(tokens) <= self.max_tokens

    def try_acquire(self, tokens: int) -> HandoffCapacityLease | None:
        tokens = int(tokens)
        if not self.can_ever_fit(tokens):
            return None
        with self._lock:
            if self._requests >= self.max_requests:
                return None
            if self._tokens + tokens > self.max_tokens:
                return None
            self._requests += 1
            self._tokens += tokens
        return HandoffCapacityLease(self, tokens)

    def _release(self, tokens: int) -> None:
        with self._lock:
            if self._requests <= 0 or self._tokens < tokens:
                raise RuntimeError("handoff capacity accounting underflow")
            self._requests -= 1
            self._tokens -= tokens

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            return self._requests, self._tokens


class HandoffCapacityLease:
    """An exactly-once reservation from :class:`HandoffCapacity`."""

    def __init__(self, owner: HandoffCapacity, tokens: int) -> None:
        self._owner: HandoffCapacity | None = owner
        self.tokens = int(tokens)
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            owner = self._owner
            if owner is None:
                return
            self._owner = None
        owner._release(self.tokens)
