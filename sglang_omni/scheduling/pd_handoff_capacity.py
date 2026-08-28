# SPDX-License-Identifier: Apache-2.0
"""Page-rounded source-KV budget for Prefill handoffs."""

from __future__ import annotations

import threading
from typing import Any

from sglang.srt.utils import ceil_align


class HandoffKVBudget:
    """Own waiting and reserved Prefill requests by request ID."""

    def __init__(self, *, max_tokens: int, page_size: int) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.max_tokens = int(max_tokens)
        self.page_size = int(page_size)
        self._used_tokens = 0
        self._waiting: dict[str, tuple[Any, int]] = {}
        self._held: dict[str, tuple[Any, int, bool]] = {}
        self._lock = threading.Lock()

    def weight(self, req: Any) -> int:
        return ceil_align(len(req.origin_input_ids), self.page_size)

    def submit(self, req: Any) -> bool:
        """Reserve source KV or queue *req* without giving it KV ownership."""

        request_id = req.rid
        tokens = self.weight(req)
        if tokens <= 0 or tokens > self.max_tokens:
            raise ValueError(
                f"request source KV requires {tokens} page-rounded tokens, "
                f"exceeding max_inflight_handoff_tokens={self.max_tokens}"
            )
        with self._lock:
            if request_id in self._waiting or request_id in self._held:
                raise RuntimeError(f"duplicate handoff budget owner {request_id!r}")
            if self._used_tokens + tokens <= self.max_tokens:
                self._held[request_id] = (req, tokens, False)
                self._used_tokens += tokens
                return True
            self._waiting[request_id] = (req, tokens)
            return False

    def admit_waiters(self) -> list[Any]:
        admitted = []
        with self._lock:
            for request_id, (req, tokens) in list(self._waiting.items()):
                if self._used_tokens + tokens > self.max_tokens:
                    break
                del self._waiting[request_id]
                self._held[request_id] = (req, tokens, False)
                self._used_tokens += tokens
                admitted.append(req)
        return admitted

    def mark_transferred(self, request_id: str) -> None:
        with self._lock:
            entry = self._held.get(request_id)
            if entry is None:
                raise RuntimeError(f"handoff has no KV budget for {request_id!r}")
            req, tokens, transferred = entry
            if transferred:
                raise RuntimeError(f"handoff already transferred for {request_id!r}")
            self._held[request_id] = (req, tokens, True)

    def cancel_before_transfer(self, request_id: str) -> Any | None:
        """Cancel a waiter/reservation; transferred ownership stays with its lease."""

        with self._lock:
            waiting = self._waiting.pop(request_id, None)
            if waiting is not None:
                return waiting[0]
            held = self._held.get(request_id)
            if held is None or held[2]:
                return None
            del self._held[request_id]
            self._used_tokens -= held[1]
            return held[0]

    def release_transferred(self, request_id: str) -> bool:
        with self._lock:
            held = self._held.get(request_id)
            if held is None:
                return False
            if not held[2]:
                raise RuntimeError(
                    f"source KV budget released before handoff for {request_id!r}"
                )
            del self._held[request_id]
            self._used_tokens -= held[1]
            return True

    def shutdown_pretransfer(self) -> list[Any]:
        """Claim every request whose KV has not moved to the comm lease."""

        with self._lock:
            requests = [req for req, _tokens in self._waiting.values()]
            self._waiting.clear()
            for request_id, (req, tokens, transferred) in list(self._held.items()):
                if transferred:
                    continue
                requests.append(req)
                self._used_tokens -= tokens
                del self._held[request_id]
            return requests

    @property
    def waiting_count(self) -> int:
        with self._lock:
            return len(self._waiting)

    def snapshot(self) -> tuple[int, int, int]:
        with self._lock:
            return len(self._held), self._used_tokens, len(self._waiting)
