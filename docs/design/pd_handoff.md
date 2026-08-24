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
`rank_ready_callback` exactly once. The callback is rank-local. The current
production runtime admits TP=1 requests; cross-rank admission for TP>1 is not
implemented.

## ACK semantics

The decode `KVReceiver.commit()` is called before the `DataAckMessage` is sent.
Therefore `rank_ready` does not depend on the ACK.  The ACK is only a
source-side release signal; a missing or delayed ACK stalls prefill cleanup, not
decode scheduling.

## Cleanup and abort

- Before `rank_ready`, `abort()` or a timeout removes the active handoff and
  calls the `cleanup_callback` exactly once.
- A successful readiness callback removes the handoff before the normal request
  lifecycle takes ownership.
- Late duplicate continuation, KV commit, and abort calls are harmless, and a
  request ID may be reused by a new transfer.

## Capability validation

`validate_continuation` rejects unsupported contracts, including unequal
prefill/decode TP sizes, cross-node handoffs, projected input embeddings,
speculative decoding, multimodal resume payloads, grammar/structured output
sampling, and custom logit processors. Speculative decoding is rejected here;
it is not an objective of this stack. The adapter validates the rank-0
continuation before associating it with a transfer.

## PR 2 / PR 3 boundary

- **PR 2:** per-rank readiness join, opaque continuation transport, capability
  validation, and the `KVReceiver` adapter (`ContinuationAwareKVReceiver`) that
  feeds the controller.
- **PR 3:** TP=1 Decode request reconstruction, committed-KV ownership, and
  scheduler-thread admission. The current runtime also requires `page_size=1`,
  same-node local transfer, and disabled RadixCache.

Model-specific state projection and restoration are supplied by sibling model
integration PRs through the generic PR3 hooks.

## Placement

The two halves may land on different GPUs or on the same one. What PD needs is
the process split, which happens either way: the prefill step leaves the decode
scheduler thread whichever card it runs on. Sharing a card also makes PD
runnable on a one-GPU box and in CI.

Two halves on one card are two process groups sharing a GPU, so the existing
colocation policy applies unchanged: each must declare a share of the card, and
the shares on one GPU may not exceed
`placement.max_total_gpu_memory_fraction_per_gpu`. Declare them per half:

```yaml
pd_disaggregation:
  prefill: {gpu: 0, memory_fraction: 0.25}
  decode:  {gpu: 0, memory_fraction: 0.65}
```

Use that share rather than `mem_fraction_static` when the halves share a card.
`total_gpu_memory_fraction` is a fraction of total physical memory, so it does
not depend on which half loads first. `mem_fraction_static` is computed against
memory free at load time, and the halves race for one startup lock: measured on
one H200, whichever half won the lock sized itself to 780,987 KV tokens and the
other failed to start.

Budget for two copies of the stage's weights on that card. Budget also for the
CUDA-IPC relay pool if the stage carries multimodal payloads: `mm_aggregate` to
the prefill half crosses a process boundary even on one device, and the relay
allocates 1024 MB there. It allocates on the first payload that crosses, not at
startup, so a deployment can pass startup and still be a gigabyte short when the
first image arrives. Measured on one H200 with both halves at 32768 KV tokens:
129,387 MiB of 143,771 used with the relay included.

Prefer the share of the card over an absolute `max_total_tokens` on a shared
GPU. `max_total_tokens` is applied as a minimum against the profiled capacity,
so a cap larger than what the later-loading half can profile stops binding on
that half and the pair reverts to order-dependent sizing without an error.
Measured: at a cap of 131072 the half that won the lock took 131072 while the
other logged `max_total_tokens=131072 is larger than the profiled value 50756`
and took 50756.
