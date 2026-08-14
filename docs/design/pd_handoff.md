# SGLang Prefill-Decode Generic Handoff (PR 2)

PR 2 defines the model-agnostic contract for handing a finished prefill request
off to a decode stage.  It adds the types and adapters PR 3 will call but does
not perform decode scheduling, request reconstruction, or model-specific
resume.

## Continuation payload

`DecodeContinuation` is a frozen dataclass that carries the per-request state a
decode scheduler needs to resume generation.  It is msgpack-encoded by
`encode_continuation` and decoded with `decode_continuation`, which rejects
unknown versions and unknown keys.

## Transport envelope

`KVTransferPrepareMessage.metadata` is `dict[str, Any]` and opaque to
`CommEngine`.

- Rank 0 metadata contains the full continuation:
  `{"pd_continuation": <bytes>, "pd_continuation_present": True}`
- Non-rank-0 TP shards carry only the marker:
  `{"pd_continuation_present": False}`

This reuses the existing rank-to-rank `CommEngine.send_kv_pages` path and keeps
request-semantic state on rank 0.

## Readiness join

`PDHandoffController` is a callback-driven, thread-safe state machine with no
polling.  For each request it waits for two rank-local events:

1. A continuation (rank 0) or `set_continuation_not_required` (non-rank-0).
2. `set_kv_committed`, called by the `KVReceiver` adapter after the KV copy is
done.

When both are satisfied and the request is not aborted, it fires the
`rank_ready_callback` exactly once.  The callback is rank-local; PR 3 must use
a cross-rank barrier on top of it before constructing a decode batch.

## ACK semantics

The decode `KVReceiver.commit()` is called before the `DataAckMessage` is sent.
Therefore `rank_ready` does not depend on the ACK.  The ACK is only a
source-side release signal; a missing or delayed ACK stalls prefill cleanup, not
decode scheduling.

## Cleanup and abort

- Before `rank_ready`, `abort()` or a timeout calls the `cleanup_callback`
  exactly once and frees the reserved/committed pages.
- After `rank_ready`, `abort()` is ignored because the normal request lifecycle
  owns cleanup.
- Duplicate continuation, KV commit, and abort calls are idempotent.

## Capability validation

`validate_continuation` rejects anything that cannot be handled generically in
PR 2: unequal prefill/decode TP sizes, cross-node handoffs, projected input
embeddings, speculative decoding, multimodal resume payloads, grammar/structured
output sampling, and custom logit processors.  The adapter performs this check
when it ingests the rank-0 continuation.

## PR 2 / PR 3 boundary

- **PR 2:** per-rank readiness join, opaque continuation transport, capability
  validation, and the `KVReceiver` adapter (`ContinuationAwareKVReceiver`) that
  feeds the controller.
- **PR 3:** logical decode admission across TP ranks, `Req` reconstruction from
  `DecodeContinuation`, `prepare_for_prebuilt()` scheduling, and model-specific
  resume builders.
