# SGLang Prefill-Decode Generic Handoff (PR 2) — Corrected Design

> **Scope:** PR 2 defines only the generic, model-agnostic request handoff contract:
> typed `DecodeContinuation`, opaque transport piggybacking, per-request readiness
> state, and capability restrictions.  It does **not** wire a real scheduler,
> reconstruct `Req` objects, implement Qwen/Ming resume builders, add GPU E2E
> tests, or change PR 1 code.

---

## 1. ACK / admission ordering (corrected)

### 1.1 What the code actually does

In `sglang_omni/comm/engine.py` the prefill-side `send_kv_pages()` path is:

1. send `KVTransferPrepareMessage` to each decode rank (`send_to_endpoint`),
2. wait for `KVTransferReadyMessage`,
3. dispatch KV page copy (`_send_rank_kv_pages`),
4. send `DataReadyMessage` to tell the decode side to read,
5. call `_arm_pending` and `await asyncio.shield(pending_task)`.

`_watch_pending` (sender side) does:

```python
await asyncio.wait_for(ack, timeout=self.transfer_timeout)
for op in pending.ops:
    op.mark_receiver_done()
    await op.wait_for_completion()
```

The `ack` future is set by `ack_transfer()`, which only reacts to a
`DataAckMessage` **from** the decode side **after** the decode side has already
committed the pages.  Therefore the sender blocks on the ACK, but the decode
side does **not** wait for an ACK before it can run `prepare_for_prebuilt`.

On the decode side `CommEngine._run_rank_control()` receives a
`KVTransferPrepareMessage` and calls `prepare_kv_receive()`.  That calls the
registered `KVReceiver.reserve()` callback, returns a `KVTransferReadyMessage`,
and waits for `DataReadyMessage`.  When `DataReadyMessage` arrives,
`_read_kv_pages()` performs the copy and then:

```python
state.receiver.commit(state.prepare_message, state.destination)
state.recv_done.set_result(None)
await self._send_message(... DataAckMessage(...))
```

`commit` is called **before** the ACK is sent.  `commit` transfers page
ownership to the scheduler/receiver.  Decode admission can therefore happen at
`commit` time; it is not gated on ACK delivery.

### 1.2 Correct semantic statement

- **Admission does NOT require ACK.**  Decode may schedule the request as soon
  as `KVReceiver.commit()` has been called for every TP rank.
- **ACK only controls source-page release.**  The prefill sender uses the ACK to
  drop its `KVPageLease` and notify the transfer operation that the receiver is
  done.  A delayed/missing ACK stalls only the prefill-side cleanup; it must not
  stall decode.
- **Abort cleanup ownership is explicit:**
  - If a request is aborted **before** `reserve()` returns, `CommEngine` sends
    a failed `KVTransferReadyMessage` and the prefill side does not copy.
  - If aborted **during** the copy, `_read_kv_pages()` raises, calls
    `receiver.abort()`, and sends `DataAckMessage(success=False)`.  The receiver
    callback must free the reserved pages.
  - If aborted **after** `commit()` but **before** scheduler admission, the
    readiness/join state (see §5) detects `aborted=True` and calls the cleanup
    callback to free the committed pages.  The ACK is still sent (the source may
    release), but the destination discards the data.
  - If aborted **after** admission, normal request lifecycle cleanup frees the
    KV pages.

### 1.3 Evidence map

| Claim | File | Lines / function |
|-------|------|------------------|
| Sender waits on ACK before releasing lease | `sglang_omni/comm/engine.py` | `_watch_pending`, `ack_transfer` |
| Receiver commits before sending ACK | `sglang_omni/comm/engine.py` | `_read_kv_pages` (commit then `DataAckMessage`) |
| `SchedulerKVReceiver.commit` is scheduler-owned, no source release | `sglang_omni/scheduling/pd_kv_adapter.py` | `SchedulerKVReceiver.commit` |
| Abort during copy calls `abort()` and sends `success=False` ACK | `sglang_omni/comm/engine.py` | `_read_kv_pages` exception path, `cleanup` |

---

## 2. Latency definitions and token-1 ownership

### 2.1 Definitions

We measure three disjoint intervals:

1. **TTFT (time-to-first-token):** wall time from the request entering the
   prefill scheduler until the first generated token is available to be
   streamed/sent to the user.  For streaming this is when token 1 leaves the
   prefill stage; for non-streaming it is the earliest moment the first token
   is known, even if the final response is not emitted until later decode steps.
2. **Handoff gap:** wall time from the end of the prefill forward that produced
   token 1 until the decode `ScheduleBatch` for that request is ready to run its
   first decode forward (token 2).  It covers continuation serialization,
   control-plane latency, KV page copy, `KVReceiver.commit()` on all TP ranks,
   and readiness join.
3. **Token-1-to-token-2 emission gap:** for streaming, wall time between the
   prefill stage emitting token 1 and the decode stage emitting token 2.
   This equals the handoff gap plus the first decode forward latency.

### 2.2 Who emits token 1

Token 1 is produced by the **prefill forward** (`process_batch_result_prefill`
appends the first sampled token to `req.output_ids`).  The prefill stage is
therefore the natural owner of token-1 emission.  Re-executing token 1 on the
decode side (`process_prebuilt` + a PREBUILT forward) is only needed so that the
decode model state matches and the next forward can generate token 2; it is
**not** a re-emission to the client.

This is confirmed by the upstream `prepare_for_prebuilt` path:

```python
def prepare_for_prebuilt(self):
    self.forward_mode = ForwardMode.PREBUILT
    input_ids = [r.get_fill_ids()[len(r.prefix_indices):] for r in reqs]
```

`get_fill_ids()` returns `origin_input_ids + output_ids`, and the slice removes
`prefix_indices` (already-transferred KV).  With `output_ids == [token_1]`, the
PREBUILT forward consumes exactly token 1 and computes token 2.  The decode
emitter then emits token 2.

### 2.3 Failure semantics

| Scenario | Behaviour |
|----------|-----------|
| Prefill fails before token 1 | Return error to caller; no handoff. |
| Prefill emits token 1, then KV transfer fails | For streaming: an error is sent after the already-emitted token 1; downstream should stop.  For non-streaming: the request fails and no response is returned. |
| Decode fails on token 2 | Error is propagated; token 1 has already been streamed for streaming, otherwise the whole request fails. |

Token 1 must be emitted **exactly once**.  The prefill stage is the only stage
that streams token 1; the decode stage streams token 2 and onward.

---

## 3. Minimal `DecodeContinuation` field table

`DecodeContinuation` is the typed, versioned payload that lets a decode
scheduler reconstruct enough per-request state to run `prepare_for_prebuilt()`
and then normal decode.  It is **not** a reconstructed `Req`; PR 3 will build
`Req` objects from it.

### 3.1 Field classification

| Field | Why it is needed | Consumer in decode | Classification |
|-------|------------------|--------------------|----------------|
| `request_id`, `transfer_id` | correlation, dedup, lifecycle | handoff controller | **must carry** |
| `origin_input_ids` | build `Req`, radix prefix match, `get_fill_ids()` | `Req.__init__`, `init_next_round_input`, `prepare_for_prebuilt` input_ids | **must carry** |
| `origin_input_ids_unpadded` | detokenization, streaming text | `Req`, output stream | **must carry** |
| `output_ids` | token 1 sampled by prefill; PREBUILT consumes token 1, process_prebuilt stashes it | `prepare_for_prebuilt` (`get_fill_ids()[prefix_len:]`), `process_prebuilt` (`output_ids[-1]`) | **must carry** |
| `sampling_params` | temperature/top_p/top_k/min_p/repetition/seed/stop/eos/logit_bias/… for every decode step | `SamplingBatchInfo.from_schedule_batch` (`r.sampling_params.*`) | **must carry** |
| `vocab_size` | validate tokens, logit bias shape, stop-token clamp | `Req.__init__`, `SamplingBatchInfo` | **must carry** |
| `eos_token_ids` | per-request / model eos set | `Req.__init__`, `update_finish_state` | **must carry** |
| `cached_tokens` (and device/host/storage breakdown) | decode radix prefix accounting | `_commit_transfer_to_req`, `prepare_for_prebuilt` (`already_computed` seed) | **must carry** |
| `mm_image/audio/video_tokens` | multimodal token budget counters | `_commit_transfer_to_req` sets them on `Req` | **must carry** |
| `return_logprob`, `top_logprobs_num`, `token_ids_logprob`, `logprob_start_len` | return prefill logprobs with continuation | `process_batch_result_prebuilt` / `process_batch_result_decode` | **must carry** |
| `return_hidden_states`, `return_sampling_mask`, `return_routed_experts`, `return_indexer_topk` | feature flags for decode post-processing | `Req` fields, batch result processor | **must carry** |
| `custom_logit_processor` | string key for `CustomLogitProcessor` replay | `SamplingBatchInfo.from_schedule_batch` | **must carry (if present)** |
| `multimodal_resume` | opaque, schema-versioned blob for models that need M-RoPE / pad values / media grid | `prepare_for_prebuilt` (`self.multimodal_inputs`), `ForwardBatch._compute_mrope_positions` | **must carry for M-RoPE models; opaque to generic layer** |
| `stage_payload` | downstream stage needs the original `StagePayload` to produce result/complete message | `result_adapter`, `stream_output` builders | **must carry** |
| `input_embeds_are_projected` | disable chunked prefill; cannot currently transfer projected embeds | `PrefillManager._needs_full_prefill` | **must carry; reject if True** |
| `speculative` flag | speculative decode requires `spec_info`/output_topk/hidden states replay | `prepare_for_prebuilt` spec path, `process_prebuilt` | **must carry; reject if True** |
| `grammar` (inside `sampling_params`) | grammar object reconstruction and token acceptance | `process_prebuilt` (`req.grammar.accept_token`) | **must carry string; reject if non-None in initial PR** |

Fields that are **NOT** carried (derived on decode side):

- `prefix_indices` — obtained from decode-side radix cache match in `pop_preallocated`.
- `extend_range` — computed by `init_next_round_input` on decode.
- `seq_lens`, `req_pool_idx` — allocated locally.
- `positional_embed_overrides` / `input_embeds` — projected embeds are unsupported in PR 2; if a model needs raw input embed transfer, it belongs in PR 3.

### 3.2 Qwen / M-RoPE special case

For Qwen3-Omni the builder computes `mrope_positions` and
`mrope_position_delta` from the prompt and media grids
(`_compute_mrope_positions` in `sglang_omni/models/qwen3_omni/request_builders.py`).

`ForwardBatch._compute_mrope_positions` uses either:

- `mm_input.mrope_positions[:, extend_prefix_len : extend_prefix_len + extend_seq_len]`
  for the PREBUILT extend, or
- `mm_input.mrope_position_delta` as a fallback to expand positions for decode
  steps.

`mrope_positions` has shape `[3, prompt_len]`, which is too large to ship for
long prompts.  The decode side can re-derive positions from `mrope_position_delta`
plus the current sequence length (`_expand_mrope_from_input`).  Therefore the
minimal multimodal resume payload for M-RoPE is:

- `mrope_position_delta` (small vector),
- `pad_values` (small dict), and
- the media grid keys (`image_grid_thw`, `video_grid_thw`, `audio_feature_lengths`,
  `second_per_grid_ts`, `use_audio_in_video`) that are already part of the
  model-specific `model_inputs`.

This is stored in the **opaque** `multimodal_resume` blob, versioned by a
schema string.  The generic PR 2 layer validates the schema is in an allow-list
but does not interpret the contents.

---

## 4. Transport choice

### 4.1 Options considered

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A | Full `DecodeContinuation` in every rank's `KVTransferPrepareMessage.metadata` | simplest, one message type | duplicates continuation `tp_size` times; `origin_input_ids` can be long; semantically per-request data on a per-rank channel |
| B | Full continuation **only on rank 0**; other ranks get a minimal handle | reuses existing per-rank endpoints; rank-0 owns request semantics; no duplication | non-rank-0 decode schedulers cannot see continuation (but they do not need it) |
| C | Small handle in KV metadata + one separate typed P→D control message for full continuation | clean separation, no semantic bloat in KV messages | needs a new control socket and ordering join between two transports; more latency/control-plane code |
| D | Continuation in a side data plane (e.g. RDMA or shared memory) keyed by transfer_id | highest bandwidth for huge payloads | overkill; continuation is tiny; adds backend-specific code |

### 4.2 Chosen design: option B

We piggyback the continuation on the **rank-0** `KVTransferPrepareMessage`
metadata.  All other TP ranks carry only a tiny handle/flag:

```python
if tp_rank == 0:
    metadata["pd_continuation"] = <serialized DecodeContinuation bytes>
else:
    metadata["pd_continuation_present"] = False
```

Reasons:

1. **Lowest latency.**  The `KVTransferPrepareMessage` is already the first
   synchronization point between prefill and decode.  Adding a second control
   message (option C) would introduce an extra network round-trip and ordering
   logic.
2. **No new sockets / polling.**  The existing `CommEngine._run_rank_control`
   loop already receives `KVTransferPrepareMessage`.  PR 2 does not introduce any
   additional polling loops.
3. **No request semantics in CommEngine.**  `KVTransferPrepareMessage.metadata`
   is `dict[str, Any]`.  `CommEngine` treats it as opaque bytes; it never parses
   the continuation.
4. **Rank-0 ownership.**  Request-semantic state logically belongs to the
   scheduler on rank 0.  Non-zero ranks only need transfer IDs and page indices
   to participate in the KV copy.
5. **No duplication.**  The full continuation is sent exactly once per request.

The decode side uses a `ContinuationAwareKVReceiver` wrapper (PR 2 generic
adapter) around the real `KVReceiver`.  The wrapper extracts the continuation
from rank-0 metadata and forwards it to the readiness/join state machine, while
delegating `reserve`/`commit`/`abort` to the underlying receiver.

---

## 5. Polling vs callbacks

The existing `CommEngine` is already **callback/event-driven**:

- `_run_rank_control()` calls `await self._recv_socket.recv()` and dispatches by
  message type.
- `KVReceiver.reserve`/`commit`/`abort` are synchronously-invoked callbacks
  owned by the scheduler/adapter.
- The upstream `DecodeRequest.kv_receiver.poll()` pattern is specific to the
  upstream scheduler and is **not** used by the `CommEngine`/`SchedulerKVReceiver`
  path.

PR 2 therefore:

- does **not** add any new polling loops,
- does **not** call `KVReceiver.poll()`,
- builds a callback-based readiness/join state machine (`PDHandoffController`)
  that is driven by `ContinuationAwareKVReceiver` (`reserve`/`commit`/`abort`)
  and by `set_continuation()` when the rank-0 continuation bytes arrive.

---

## 6. Per-request readiness state and transitions

### 6.1 State container: `PendingHandoff`

```text
PendingHandoff
├── request_id
├── transfer_id
├── continuation: DecodeContinuation | None     # from rank-0 metadata
├── kv_committed: bool                           # set by commit callback
├── aborted: bool
├── admitted: bool                               # exactly-once guard
├── deadline: float | None
└── admit_callback / cleanup_callback
```

### 6.2 Transitions

```
start(request_id, transfer_id, timeout)
    -> create PendingHandoff, schedule timeout

continuation_arrived(bytes)
    -> decode & validate
    -> if pending.aborted: drop
    -> set continuation, _try_admit()

kv_committed()
    -> if pending.aborted or admitted: ignore
    -> set kv_committed=True, _try_admit()

_try_admit()
    -> if continuation and kv_committed and not aborted and not admitted:
           admitted=True; cancel timeout; admit_callback(pending)

abort(reason)
    -> if admitted: ignore (too late, normal lifecycle owns cleanup)
    -> if already aborted: idempotent
    -> aborted=True; cancel timeout; cleanup_callback(pending, reason)

timeout()
    -> abort("timeout")
```

### 6.3 Duplicate / stale handling

- **Duplicate continuation:** same `transfer_id` / `request_id` while not yet
  admitted is ignored after the first valid decode.
- **Duplicate commit:** `KVReceiver.commit` is called once per transfer by
  `CommEngine`.  A second commit for the same `request_id` is logged and ignored.
- **Stale messages:** a continuation/commit for an unknown `request_id` is
  rejected; the receiver wrapper should abort the underlying reservation if
  one exists.
- **Exactly-once admission:** `admitted` is set inside `_try_admit` under a
  lock; after that the pending record is retained only for duplicate suppression
  until `ack` is irrelevant.

---

## 7. PR 2 typed interfaces

- `DecodeContinuation` — frozen dataclass, versioned, msgpack-serializable,
  validates required fields.
- `ContinuationSerializer` — `encode()` / `decode()` with strict version and
  schema checking.
- `MultimodalResumePayload` — opaque `schema` + `data` dict; generic layer only
  checks the schema string.
- `PDCapabilityPolicy` / `PDCapabilityError` — rejects unsupported modes.
- `PendingHandoff` — lightweight per-request readiness record.
- `PDHandoffController` — callback-driven join state machine (continuation + KV
  commit -> exactly-once admission; timeout/abort cleanup).
- `ContinuationAwareKVReceiver` — `KVReceiver` wrapper that peels the rank-0
  continuation out of `KVTransferPrepareMessage.metadata` and feeds the
  controller.
- `PrefillContinuationProducer` — helper to build per-rank metadata for
  `CommEngine.send_kv_pages` (full continuation on rank 0, handle on others).
- `DecodeContinuationConsumer` — callback interface the decode scheduler
  implements to receive the joined `PendingHandoff`.

No `Req`, `ScheduleBatch`, model-specific builders, or real GPU transfer code is
introduced.

---

## 8. Initial unsupported-mode policy (PR 2)

`PDCapabilityPolicy` rejects a handoff unless all of the following hold:

1. Prefill and decode TP sizes are equal.
2. `input_embeds_are_projected` is `False` (projected input embeddings are
   not supported).
3. `speculative` is `False`.
4. If `multimodal_resume` is present, its `schema` is in the allow-list.  In
   PR 2 the allow-list is empty, so any multimodal resume payload is rejected
   (text-only handoff is supported).
5. If `sampling_params` contains `grammar`, `json_schema`, `regex`, or
   `structural_tag`, reject unless `allow_grammar=True` (default `False`).
6. Cross-node transfer is rejected unless `allow_cross_node=True` (default
   `False`).
7. `custom_logit_processor` is rejected unless
   `allow_custom_logit_processor=True` (default `False`).

These are generic checks; PR 3 will opt-in specific models by expanding the
allow-lists and adding resume builders.

---

## 9. Evidence checklist

| Design decision | Supporting code |
|-----------------|-----------------|
| Admission before ACK | `sglang_omni/comm/engine.py` `_read_kv_pages` (commit then DataAckMessage) |
| ACK only for source release | `sglang_omni/comm/engine.py` `_watch_pending`, `ack_transfer` |
| `KVReceiver` callback contract | `sglang_omni/comm/kv_transfer.py` `KVReceiver` Protocol |
| `prepare_for_prebuilt` input_ids from `get_fill_ids` | `sglang/srt/disaggregation/decode_schedule_batch_mixin.py` `prepare_for_prebuilt` |
| `process_prebuilt` uses `output_ids[-1]` | `sglang/srt/disaggregation/decode_schedule_batch_mixin.py` `process_prebuilt` |
| Sampling info built from `sampling_params` | `sglang/srt/sampling/sampling_batch_info.py` `from_schedule_batch` |
| `update_finish_state` uses `sampling_params`, `eos_token_ids` | `sglang/srt/managers/schedule_batch.py` `Req.update_finish_state` |
| M-RoPE fallback uses `mrope_position_delta` | `sglang/srt/model_executor/forward_batch_info.py` `_expand_mrope_from_input` |
| Qwen builder stores `mrope_positions` + `mrope_position_delta` | `sglang_omni/models/qwen3_omni/request_builders.py` `_compute_mrope_positions`, `build_sglang_thinker_request` |
| `KVTransferPrepareMessage` has opaque `metadata` dict | `sglang_omni/proto/kv_transfer.py` |
| No new polling loops needed | `sglang_omni/comm/engine.py` `_run_rank_control` async recv loop |
