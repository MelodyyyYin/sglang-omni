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

## Configuration surface (PR 1 capability)

PR 1 can compile a stage into prefill and decode halves, but nothing exposes
that capability: `pd_disaggregation` is a `StageConfig` field, `stage_overrides`
accepts only `runtime` keys, and no CLI flag sets it. A deployment therefore
cannot turn PD on.

    --pd-stage STAGE=PREFILL_GPUS:DECODE_GPUS

    --pd-stage thinker=0:1        # prefill on GPU 0, decode on GPU 1
    --pd-stage thinker=0,1:2,3    # TP=2 on each half

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
supplied some other way; on its own it still fails at bind time.

Closing that gap is a PR 3 decision and is deliberately outside this surface.
Either the PD path forces the required args on the generated halves and rejects
a contradicting user value, as `models/ming_tts/engine_builder.py` does for
`disable_radix_cache`, or compilation fails with a message naming what to set.

---

# Serving shape: what the workload asks of PD

This part records measurements of the Qwen3-Omni thinker and what they imply for the delivery plan. It is written to be falsified: every claim names how it was produced.

## Measured on one H200

`Qwen/Qwen3-Omni-30B-A3B-Instruct`, text-only pipeline, TP=1, `max_running_requests=64`, `enable_mixed_chunk=True`, RadixCache on, decode CUDA graph on, prefill CUDA graph off. Open-loop Poisson arrivals, streaming, `max_tokens=128`. Attribution comes from the existing profiler events in `sglang_omni/profiler/views.py`, grouping every inter-token gap by whether it overlaps another request's prefill window.

A prefill step costs about **59 ms fixed plus 0.010 ms per prompt token**. A 41-token prompt costs 59 ms and a 6944-token prompt costs 129 ms, so a 169x change in prompt length moves the step by 2.2x. The server's own metrics show why: with unique prompts the step reports `#new-token: 27.6`, and with a warm prefix tree it reports `#new-token: 3.2`, and both cost about 59 ms. Doing three tokens of work costs what doing twenty-eight costs.

A decode step that overlaps another request's prefill takes **about 65 ms instead of about 6 ms**. The stall equals one prefill step, and it grows with prompt length exactly as the prefill step does: 61.75 ms at 41 tokens, 134.24 ms at 6944. Reconstructing marginal inter-token latency as `clean * (1 - f) + stalled * f`, where `f` is the fraction of gaps that overlap a prefill, matches the measured mean within 2% at low load and within 10% across the sweep. Prefill collisions account for the latency, and no second mechanism is needed to explain it.

Saturation is **17.5 to 18.1 req/s**, against `1 / 0.059 = 16.9` predicted by the fixed cost alone. Past saturation the admission queue, not inter-token latency, dominates: at 24 req/s offered the median wait from queue entry to prefill start is 12 to 20 s, while the whole 128-token stream costs about 3.5 s of inter-token time.

## What follows for the target workload

**PD should be qualified on long prompts and multimodal input, not on short text.**

Two mechanisms point the same way. First, the fixed 59 ms does not shrink when PD moves the prefill step to another GPU; it only changes which engine pays it. On short text that fixed cost is the whole cost, so a 1P:1D pair inherits the same ceiling that one colocated engine already reached while also doing all the decode work. Second, `enable_mixed_chunk` already folds running decodes into a chunk-prefill step, which is where the +14% in #760 came from, so on short text the overlap PD would create inside one engine largely exists.

Neither holds for multimodal input. `scheduling/sglang_backend/prefill.py::_needs_full_prefill` disables chunking when a request carries projected multimodal embeddings, and mixed-chunk engages only when `chunked_prefill_size > 0`. A multimodal prefill therefore cannot fold decodes in, and it blocks the running batch for its whole duration. At the same time the per-token term stops being negligible: a 6063-token image prompt, the size #1599 measured with `tests/data/cars.jpg`, sits at the crossover where work and fixed overhead are equal.

PR 4 already resumes that path through mRoPE, so moving the qualification workload costs no new engine work. It changes which arm PR 5 leads with.

**The comparison must hold GPU count equal.** Your validated topology uses three H200s. Three colocated replicas is the deployment a reader would otherwise choose, and `docs/basic_usage/mps_dp.md` reports same-GPU DP reaching 1.4 to 2.1x a tuned single replica, so the bar is higher still. A 2-GPU or 3-GPU PD arm compared against a 1-GPU colocated arm would report a speedup that does not exist.

## Why multi-P/D follows from the measurements, not from ambition

The deferred list treats multi-P/multi-D routing as a later feature. The workload argues it is the point rather than an extra.

The prefill and decode shares of a request move with the input. `Awesome-ML-SYS-Tutorial/sglang/sglang-omni/moss-td-asr.md` reports AR decode at 94% of stage time for one 60 s clip at concurrency 1, and encoder plus prefill at 68% for 5 s clips at concurrency 16. Same model, opposite balance. A fixed 1:1 pairing serves one end of that range and wastes a GPU at the other.

This does not argue for building routing now. It argues for one decision now that keeps the cost of adding it later small: **resolve the decode target by logical name at admission rather than by the physical string `PDExecution.partner` holds.** With a single decode worker the resolved value is constant and nothing changes at runtime. Stage replicas already carry the same mechanism, where the coordinator binds at submit and the binding travels on the message envelope, so PD and replicas would share one resolution path instead of growing two.

## Elasticity: what is needed and what is not

Kubernetes-style autoscaling is not needed. `docs/design/refactor_rfc.md` states the target shape as a small fixed number of stages on a small fixed number of GPUs with no autoscaling, and nothing measured here contradicts that.

What the numbers do ask for is a **static N:M ratio chosen at launch**, with the decode worker selected at admission. That is the smallest change that covers the range in the paragraph above, and it reuses the replica binding rather than adding a second mechanism. Runtime replica changes and prefill-to-decode role switching stay out: they touch the KV pool lifecycle and the placement plan, and no measurement here shows they pay.

## Two deferred items look cheaper than the list assumes

**Decode CUDA graph costs nothing.** `disaggregation/decode.py::_run_batch_prebuilt` returns an empty `GenerationBatchResult` and runs no forward, except for the inner idle batch that DP attention uses for its MLP sync, and `ForwardMode.IDLE` is graph-eligible. Every decode step after admission is `ForwardMode.DECODE`. There is no shape bucket for PREBUILT to miss, so the decode half does not need CUDA graph disabled.

**Decode RadixCache has an upstream implementation to port.** `disaggregation/prefill.py` pairs `maybe_cache_unfinished_req` with `release_kv_cache`, holding a lock reference for the duration of the transfer rather than claiming exclusive ownership of the pages. `disaggregation/decode.py` gates the decode side on `disaggregation_decode_enable_radix_cache`, matches the prefix, and calls `inc_lock_ref`. The wire protocol already carries `decode_prefix_len`, with a `nokv` marker for a full prefix hit. That reduces "Decode RadixCache integration and delta KV transfer" from a project to a port, and it lowers transfer volume on exactly the prefixes omni repeats: system prompts, speaker references, and fixed ASR prompts.

## What PR 5 should report

Define goodput as the request rate that meets every SLO component at 90% attainment per GPU, which is the DistServe definition with omni's SLO vector substituted: time to first token, time to first audio, and inter-chunk deadline misses for streaming audio.

Follow the measurement contract already in #1018: an A/A noise band before any comparison, three paired repeats, at least 32 requests, and a sweep that passes the admission ceiling rather than stopping where the system is still healthy. Our A/A run shows why the last point matters: p95 time to first token varied 2.5x between two identical arms, while the conditioned quantity, the median stalled gap, varied 0.9%. Report conditioned quantities, not marginal ones.

Prefer open-loop arrivals. `benchmarks/eval/bench_sweep.py` records the reason: past saturation a closed-loop client absorbs the backlog into its own think time and hides the capacity knee.

Report the prefill step as **steps per second with its fixed and per-token parts separated**, not as prompt tokens per second. The same engine reports 712 and 54,000 prompt tokens per second at 42 and 6944 prompt tokens, a 76x spread with no change in the underlying cost structure.

## Upstream capabilities this stack can adopt

Two constraints the current runtime carries have an implementation in
`sglang.srt.disaggregation` already. Recording the entry points here so the
work is a port rather than a rediscovery.

**Decode CUDA graph.** `PREBUILT` is absent from `ForwardMode.is_cuda_graph()`,
but that does not constrain the decode half. `disaggregation/decode.py`
`_run_batch_prebuilt` returns an empty `GenerationBatchResult` and runs no
forward, apart from the inner idle batch DP attention uses for its MLP sync,
and `IDLE` is graph-eligible. Every step after admission is
`ForwardMode.DECODE`. There is no shape bucket for `PREBUILT` to miss.

**Decode RadixCache.** Upstream keeps prefix reuse on both sides of a
disaggregated pair, using lock references rather than exclusive page ownership:

| Side | Entry points |
| --- | --- |
| prefill | `disaggregation/prefill.py`: `maybe_cache_unfinished_req`, then `release_kv_cache` to unlock the tree |
| decode | `disaggregation/decode.py`: gated on `disaggregation_decode_enable_radix_cache`, matches the prefix, then `inc_lock_ref` on the matched node |
| wire | `decode_prefix_len` in the transfer metadata, with a `nokv` marker when the decode-side prefix already covers the request |

The decode side therefore receives only the pages it does not already hold.
That matters for the prefixes this project repeats: system prompts, speaker
references, and fixed ASR prompts.
