# SPDX-License-Identifier: Apache-2.0
"""Generic Prefill-Decode request handoff contract for PR 2.

This module defines the typed continuation object, its opaque serialization, and
the callback-based readiness/join state machine.  It is intentionally
model-agnostic: it never imports SGLang `Req`, `ScheduleBatch`, or model-specific
resume builders.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

import msgspec

from sglang_omni.comm.kv_transfer import (
    KVPageDestination,
    KVReceiver,
    KVTransferPrepareMessage,
)
from sglang_omni.scheduling.pd_capabilities import (
    PDCapabilityError,
    PDCapabilityPolicy,
)

logger = logging.getLogger(__name__)

CONTINUATION_VERSION = "pd-continuation-v1"


class ContinuationSchemaError(ValueError):
    """Raised when a continuation payload has an unknown or incompatible schema."""


class PDAdmissionCallback(Protocol):
    """Called exactly once when a request is ready to be admitted to decode."""

    def __call__(self, pending: "PendingHandoff") -> None: ...


class PDCleanupCallback(Protocol):
    """Called when a handoff is aborted or times out before admission."""

    def __call__(self, pending: "PendingHandoff", reason: str) -> None: ...


@dataclasses.dataclass(frozen=True)
class DecodeContinuation:
    """Minimal per-request state needed by a decode scheduler to resume generation.

    This is **not** a reconstructed `Req`.  PR 3 will build `Req` objects from
    this payload.  All fields are JSON/msgpack-serializable.
    """

    request_id: str
    transfer_id: str
    origin_input_ids: list[int]
    output_ids: list[int]
    vocab_size: int
    sampling_params: dict[str, Any]
    cached_tokens: int

    version: str = CONTINUATION_VERSION
    origin_input_ids_unpadded: list[int] | None = None
    eos_token_ids: list[int] | None = None
    cached_tokens_device: int = 0
    cached_tokens_host: int = 0
    cached_tokens_storage: int = 0
    mm_image_tokens: int = 0
    mm_audio_tokens: int = 0
    mm_video_tokens: int = 0
    return_logprob: bool = False
    top_logprobs_num: int = 0
    token_ids_logprob: list[int] | None = None
    logprob_start_len: int = -1
    return_hidden_states: bool = False
    return_sampling_mask: bool = False
    return_routed_experts: bool = False
    return_indexer_topk: bool = False
    custom_logit_processor: str | None = None
    input_embeds_are_projected: bool = False
    prefill_input_embeds_shape: tuple[int, ...] | None = None
    speculative: bool = False
    multimodal_resume: dict[str, Any] | None = None
    stage_payload: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.version != CONTINUATION_VERSION:
            raise ContinuationSchemaError(
                f"unsupported continuation version {self.version!r}; "
                f"expected {CONTINUATION_VERSION!r}"
            )
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.transfer_id:
            raise ValueError("transfer_id must be non-empty")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.cached_tokens < 0:
            raise ValueError(f"cached_tokens must be non-negative, got {self.cached_tokens}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecodeContinuation":
        if not isinstance(data, dict):
            raise ContinuationSchemaError("continuation must be a dictionary")
        expected = {f.name for f in dataclasses.fields(cls)}
        extra = set(data) - expected
        if extra:
            raise ContinuationSchemaError(f"unknown continuation keys: {extra}")

        kwargs: dict[str, Any] = {}
        for field in dataclasses.fields(cls):
            if field.name in data:
                kwargs[field.name] = data[field.name]
            elif field.default is not dataclasses.MISSING:
                kwargs[field.name] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                kwargs[field.name] = field.default_factory()
            else:
                raise ContinuationSchemaError(
                    f"missing required continuation key: {field.name!r}"
                )
        return cls(**kwargs)


class ContinuationSerializer:
    """Versioned msgpack encoder/decoder for `DecodeContinuation`."""

    SUPPORTED_VERSION = CONTINUATION_VERSION

    @classmethod
    def encode(cls, continuation: DecodeContinuation) -> bytes:
        return msgspec.msgpack.encode(continuation.to_dict())

    @classmethod
    def decode(cls, data: bytes) -> DecodeContinuation:
        try:
            decoded = msgspec.msgpack.decode(data)
        except Exception as exc:
            raise ContinuationSchemaError("continuation msgpack decode failed") from exc
        if not isinstance(decoded, dict):
            raise ContinuationSchemaError("decoded continuation is not a dict")
        version = decoded.get("version")
        if version != cls.SUPPORTED_VERSION:
            raise ContinuationSchemaError(
                f"unsupported continuation version {version!r}"
            )
        return DecodeContinuation.from_dict(decoded)


@dataclasses.dataclass
class PendingHandoff:
    """Per-request readiness/join record."""

    request_id: str
    transfer_id: str
    continuation: DecodeContinuation | None = None
    kv_committed: bool = False
    aborted: bool = False
    admitted: bool = False
    cleanup_invoked: bool = False
    deadline: float | None = None
    timeout_handle: Any = None
    context: dict[str, Any] = dataclasses.field(default_factory=dict)


class PDHandoffController:
    """Callback-driven join state machine: continuation + KV commit -> admission.

    This class is thread-safe.  It must not add new polling loops: it is driven
    by `set_continuation`, `set_kv_committed`, and `abort` calls (usually from
    `CommEngine`/`KVReceiver` callbacks).
    """

    def __init__(
        self,
        *,
        policy: PDCapabilityPolicy | None = None,
        default_timeout_s: float = 30.0,
        admit_callback: PDAdmissionCallback | None = None,
        cleanup_callback: PDCleanupCallback | None = None,
    ) -> None:
        self._policy = policy or PDCapabilityPolicy()
        self._default_timeout_s = default_timeout_s
        self._admit_callback = admit_callback
        self._cleanup_callback = cleanup_callback
        self._lock = threading.RLock()
        self._pending: dict[str, PendingHandoff] = {}

    def start_handoff(
        self,
        request_id: str,
        transfer_id: str,
        timeout_s: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> PendingHandoff:
        """Create a pending handoff record.

        This should be called when the `KVTransferPrepareMessage` is first seen.
        """
        with self._lock:
            if request_id in self._pending:
                return self._pending[request_id]
            deadline = None
            handle = None
            effective_timeout = timeout_s if timeout_s is not None else self._default_timeout_s
            if effective_timeout is not None:
                deadline = time.monotonic() + effective_timeout
                try:
                    loop = asyncio.get_running_loop()
                    handle = loop.call_later(
                        effective_timeout, self._on_timeout, request_id
                    )
                except RuntimeError:
                    # No running event loop; the caller or tests must call
                    # `check_timeouts()` or trigger `abort()` manually.
                    pass
            pending = PendingHandoff(
                request_id=request_id,
                transfer_id=transfer_id,
                deadline=deadline,
                timeout_handle=handle,
                context=context or {},
            )
            self._pending[request_id] = pending
            logger.debug("started handoff request=%s transfer=%s", request_id, transfer_id)
            return pending

    def set_continuation(
        self,
        request_id: str,
        continuation: DecodeContinuation,
        *,
        source_tp_size: int = 1,
        target_tp_size: int = 1,
        is_local: bool = True,
    ) -> None:
        """Accept the decode continuation for a request.

        The continuation is validated against the configured capability policy.
        """
        with self._lock:
            pending = self._pending.get(request_id)
            if pending is None:
                raise KeyError(f"continuation for unknown request {request_id!r}")

            if pending.admitted or pending.aborted:
                logger.warning(
                    "ignoring continuation for %s: already %s",
                    request_id,
                    "admitted" if pending.admitted else "aborted",
                )
                return

            if pending.continuation is not None:
                logger.warning(
                    "duplicate continuation for request %s ignored", request_id
                )
                return

            self._policy.validate_continuation(
                continuation,
                source_tp_size=source_tp_size,
                target_tp_size=target_tp_size,
                is_local=is_local,
            )

            pending.continuation = continuation
            logger.debug("continuation set for request=%s", request_id)
            self._try_admit(pending)

    def set_continuation_bytes(
        self,
        request_id: str,
        data: bytes,
        *,
        source_tp_size: int = 1,
        target_tp_size: int = 1,
        is_local: bool = True,
    ) -> None:
        """Decode and accept a continuation from raw msgpack bytes."""
        continuation = ContinuationSerializer.decode(data)
        self.set_continuation(
            request_id,
            continuation,
            source_tp_size=source_tp_size,
            target_tp_size=target_tp_size,
            is_local=is_local,
        )

    def set_kv_committed(self, request_id: str) -> None:
        """Notify the controller that the KV receiver has committed the transfer."""
        with self._lock:
            pending = self._pending.get(request_id)
            if pending is None:
                logger.warning("kv_committed for unknown request %s", request_id)
                return
            if pending.admitted or pending.aborted:
                logger.debug(
                    "ignoring kv_committed for %s: already %s",
                    request_id,
                    "admitted" if pending.admitted else "aborted",
                )
                return
            if pending.kv_committed:
                logger.warning("duplicate kv_committed for request %s", request_id)
                return
            pending.kv_committed = True
            logger.debug("kv committed for request=%s", request_id)
            self._try_admit(pending)

    def abort(self, request_id: str, reason: str = "abort") -> None:
        """Abort a pending handoff before admission."""
        with self._lock:
            pending = self._pending.get(request_id)
            if pending is None:
                return
            if pending.admitted:
                logger.debug("abort after admission ignored for %s", request_id)
                return
            if pending.aborted:
                return
            pending.aborted = True
            self._cancel_timeout(pending)
            if not pending.cleanup_invoked:
                pending.cleanup_invoked = True
                if self._cleanup_callback is not None:
                    try:
                        self._cleanup_callback(pending, reason)
                    except Exception:
                        logger.exception("cleanup callback failed for %s", request_id)
            logger.debug("handoff aborted request=%s reason=%s", request_id, reason)

    def check_timeouts(self) -> None:
        """Synchronous timeout helper for callers without a running event loop."""
        now = time.monotonic()
        with self._lock:
            for request_id, pending in list(self._pending.items()):
                if (
                    not pending.admitted
                    and not pending.aborted
                    and pending.deadline is not None
                    and now >= pending.deadline
                ):
                    logger.warning("handoff timeout for request=%s", request_id)
                    self._do_abort_locked(pending, "timeout")

    def get_pending(self, request_id: str) -> PendingHandoff | None:
        with self._lock:
            return self._pending.get(request_id)

    def is_admitted(self, request_id: str) -> bool:
        with self._lock:
            pending = self._pending.get(request_id)
            return pending.admitted if pending else False

    def _try_admit(self, pending: PendingHandoff) -> None:
        if (
            pending.continuation is not None
            and pending.kv_committed
            and not pending.aborted
            and not pending.admitted
        ):
            pending.admitted = True
            self._cancel_timeout(pending)
            logger.debug("admitting request=%s transfer=%s", pending.request_id, pending.transfer_id)
            if self._admit_callback is not None:
                try:
                    self._admit_callback(pending)
                except Exception:
                    logger.exception("admit callback failed for %s", pending.request_id)

    def _on_timeout(self, request_id: str) -> None:
        with self._lock:
            pending = self._pending.get(request_id)
            if pending is None or pending.admitted or pending.aborted:
                return
            self._do_abort_locked(pending, "timeout")

    def _do_abort_locked(self, pending: PendingHandoff, reason: str) -> None:
        pending.aborted = True
        self._cancel_timeout(pending)
        if not pending.cleanup_invoked:
            pending.cleanup_invoked = True
            if self._cleanup_callback is not None:
                try:
                    self._cleanup_callback(pending, reason)
                except Exception:
                    logger.exception("cleanup callback failed for %s", pending.request_id)
        logger.debug("handoff aborted request=%s reason=%s", pending.request_id, reason)

    def _cancel_timeout(self, pending: PendingHandoff) -> None:
        handle = pending.timeout_handle
        if handle is not None:
            try:
                handle.cancel()
            except Exception:
                pass
            pending.timeout_handle = None
        pending.deadline = None


class ContinuationAwareKVReceiver:
    """Wraps a `KVReceiver` and feeds the `PDHandoffController`.

    This is the generic PR 2 adapter that lets the existing `CommEngine`
    hand off a rank-0 continuation without modifying the PR 1 `KVReceiver`
    protocol.
    """

    def __init__(
        self,
        inner: KVReceiver,
        controller: PDHandoffController,
        *,
        source_tp_size: int = 1,
        target_tp_size: int = 1,
        is_local: bool = True,
    ) -> None:
        self._inner = inner
        self._controller = controller
        self._source_tp_size = source_tp_size
        self._target_tp_size = target_tp_size
        self._is_local = is_local

    def reserve(self, request: KVTransferPrepareMessage) -> KVPageDestination:
        self._controller.start_handoff(request.request_id, request.transfer_id)
        self._maybe_ingest_continuation(request)
        return self._inner.reserve(request)

    def commit(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination,
    ) -> None:
        self._inner.commit(request, destination)
        self._controller.set_kv_committed(request.request_id)

    def abort(
        self,
        request: KVTransferPrepareMessage,
        destination: KVPageDestination | None,
        error: BaseException,
    ) -> None:
        self._inner.abort(request, destination, error)
        self._controller.abort(request.request_id, reason=str(error) or type(error).__name__)

    def _maybe_ingest_continuation(self, request: KVTransferPrepareMessage) -> None:
        raw = request.metadata.get("pd_continuation")
        if raw is None:
            return
        if not isinstance(raw, (bytes, bytearray)):
            raise ContinuationSchemaError("pd_continuation metadata must be bytes")
        self._controller.set_continuation_bytes(
            request.request_id,
            bytes(raw),
            source_tp_size=self._source_tp_size,
            target_tp_size=self._target_tp_size,
            is_local=self._is_local,
        )


class PrefillContinuationProducer:
    """Build per-rank metadata for `CommEngine.send_kv_pages`.

    The full continuation is embedded only on rank 0.  All other ranks carry a
    tiny handle so the `ContinuationAwareKVReceiver` can still correlate the
    per-rank KV transfer.
    """

    def __init__(self, tp_size: int = 1) -> None:
        self._tp_size = tp_size

    def prepare_rank_metadata(
        self,
        continuation: DecodeContinuation,
        tp_rank: int,
    ) -> dict[str, Any]:
        if not 0 <= tp_rank < self._tp_size:
            raise ValueError(f"tp_rank {tp_rank} out of range for tp_size {self._tp_size}")
        if tp_rank == 0:
            return {
                "pd_continuation": ContinuationSerializer.encode(continuation),
                "pd_continuation_present": True,
                "tp_rank": tp_rank,
                "tp_size": self._tp_size,
            }
        return {
            "pd_continuation_present": False,
            "pd_continuation_handle": continuation.transfer_id,
            "request_id": continuation.request_id,
            "tp_rank": tp_rank,
            "tp_size": self._tp_size,
        }


class DecodeContinuationConsumer(Protocol):
    """Interface a decode scheduler implements to consume a joined handoff."""

    def on_admit(self, pending: PendingHandoff) -> None: ...
    def on_cleanup(self, pending: PendingHandoff, reason: str) -> None: ...
