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

<<<<<<< HEAD
## Configuration surface (PR 1 capability)

PR 1 can compile a stage into prefill and decode halves, but nothing exposes
that capability: `pd_disaggregation` is a `StageConfig` field, `stage_overrides`
accepts only `runtime` keys, and no CLI flag sets it. A deployment therefore
cannot turn PD on.

    --pd-stage STAGE=PREFILL_GPUS:DECODE_GPUS

    --pd-stage thinker=0:1        # prefill on GPU 0, decode on GPU 1
    --pd-stage thinker=0,1:2,3    # two GPUs per half; see prerequisites below

The flag addresses a stage by name and carries no model-specific knowledge,
matching `--stage-process STAGE=PROCESS`. Placement is a CLI concern in this
repo — `stage_overrides` rejects `gpu` — so this follows that boundary rather
than widening it. `STAGE` also accepts a role alias through
`isolation_role_to_stage()`, as `--stage-process` does.

`apply_pd_stage_overrides` writes `PDConfig` onto the named stage and then
re-runs `PipelineConfig._validate_pd`: `model_copy` does not re-enter
`model_post_init`, so without that call the placement would reach expansion
unvalidated.

### Runtime prerequisites this flag does not set

`bind_pd_runtime` requires `disable_radix_cache`, `page_size=1`, and
`tp_size=1`. Those are SGLang server args, reachable today only through a
stage's `factory_args.server_args_overrides`, which the CLI does not expose.
`--pd-stage` therefore makes PD compile and launch only when those args are
supplied some other way; on its own it still fails at bind time. A two-GPU half
parses and places, but `bind_pd_runtime` rejects `tp_size=2`, so that form does
not run today.

### Memory budget on the prefill half

Splitting a multimodal stage moves the `mm_aggregate` to prefill edge across a
process boundary. The CUDA-IPC relay then allocates a pool on the prefill GPU
that a colocated deployment never allocates. The default size is 1024 MB:
`relay/cuda_ipc.py` takes `slot_size_mb=512` and `credits=2`. The relay
allocates it on the first payload that crosses the boundary, not at startup, so
a `mem_fraction_static` copied from a colocated deployment fails on the first
multimodal request rather than at launch.

`--pd-stage` moves only the stage it names. Stages that feed it, including the
encoders, keep the GPU they already had, so on a two-GPU multimodal deployment
the prefill card carries the encoders as well. Budget for that pool and for the
encoder activations on the prefill half, or pass `pool_size_mb` to match the
payload. Measured on one H200: the prefill half at `mem_fraction_static=0.87`
plus the encoder process reached 139.5 GiB of 139.80 GiB and failed a 750 MiB
allocation on the first image request. 0.80 left room.

Closing that gap is a PR 3 decision and is deliberately outside this surface.
Either the PD path forces the required args on the generated halves and rejects
a contradicting user value, as `models/ming_tts/engine_builder.py` does for
`disable_radix_cache`, or compilation fails with a message naming what to set.
=======
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
>>>>>>> pd/free-placement-solo
