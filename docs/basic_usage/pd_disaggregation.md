# Prefill/Decode Disaggregation

> TL;DR: PD disaggregation splits one stage into a prefill process and a decode process so a prefill no longer stalls decoding. It removes that interference almost completely. It does not raise throughput at equal GPU count on Qwen3-Omni thinker: in the measurements below two colocated replicas beat one prefill/decode pair on both text and image. Turn it on when inter-token latency stability is what you need, not to serve more requests per GPU.

A colocated replica runs prefill and decode on one card and one scheduler thread. A prefill therefore blocks the decode steps of every request already running. On Qwen3-Omni that cost is large, and it is worst on images, where a 6127-token prompt cannot be chunked.

PD disaggregation moves prefill into its own process. The prompt KV and the request's continuation state transfer once, over CUDA IPC, and decoding continues on the other half.

## Choosing a configuration

PD is one decision among several, and the others are not about the queue. Work through these in order; each one can rule PD out before you tune anything.

### 1. Is prefill a large enough share of your work?

PD gives prefill its own hardware. That pays only if prefill is a large enough share of the GPU work to justify the hardware it gets. The share is governed by prompt length divided by output length.

| Prompt / output tokens | Prefill share |
| --- | --- |
| 42 / 128 | about 2.6% |
| 6944 / 128 | about 9% |
| 42 / 8 | about 42% |

A 1:1 pair splits the hardware evenly. Against a 2.6% share that leaves the prefill card about 90% idle, which is what the equal-GPU-count measurement found. Long prompts with short outputs are the shape that justifies the split; short prompts with long outputs are not.

If your workload sits at the low end, the interference PD removes may still be worth it, but expect to pay throughput for it. Read the next question before deciding.

### 2. Is your problem throughput, or is it inter-token jitter?

These want opposite answers, and the measurements separate them cleanly.

If the complaint is **throughput per GPU**, PD at 1:1 is not the fix. Two colocated replicas beat one pair on both text and image at equal GPU count.

If the complaint is **an audio or token stream that stutters when another request starts**, PD addresses exactly that. A colocated replica pays 5.9x to 8.5x on the inter-token gap that overlaps a prefill on text, and 8.4x to 20.5x on images. PD brings that to 1.0x. It is worst on images because a 6127-token image prompt cannot be chunked: `_needs_full_prefill` forces the whole prompt through in one step, and no decode work batches with it.

For a real-time stream with a repeating deadline, that jitter is the SLO, and a throughput number does not describe it.

### 3. Does your workload depend on prefix reuse?

PD forces `disable_radix_cache` and `page_size=1` on both halves. A workload built on a long shared prefix loses that reuse: a fixed system prompt, a few-shot preamble, or a reference audio prefix. Measure your cache hit rate before splitting. This cost does not appear in a throughput comparison run with unique prompts.

### 4. Which overload behaviour do you want?

Decide this deliberately, because the default is an unbounded queue and it shows up as latency rather than as an error. See [What to expect](#what-to-expect) for the three settings and which question each answers.

### 5. Does the prefill card have room for what moves onto it?

`--pd-stage` moves only the stage it names. The encoders stay put, and the CUDA-IPC relay pool appears on the prefill card. Budget for both. This is the most common way a first PD launch fails.

## Turn it on

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_pd.yaml \
  --pd-stage thinker=0:1 \
  --port 8000
```

Pick a config whose `config_cls` is not one of the colocated classes. `Qwen3OmniSpeechColocatedPipelineConfig` requires the image encoder, audio encoder, thinker, talker and code2wav to share one GPU, which a split thinker contradicts, and it reports that as `Qwen colocated speech requires exactly one GPU id for ['thinker']`.

Do not combine `--thinker-mem-fraction-static` with a config that already sets `total_gpu_memory_fraction` for the thinker. The two are separate memory contracts and the stage rejects the pair.

`--pd-stage STAGE=PREFILL_GPUS:DECODE_GPUS` splits one stage. Repeat the flag to split more than one. `STAGE` accepts a stage name or a role alias, as `--stage-process` does.

The flag sets the two server args PD requires, `disable_radix_cache` and `page_size=1`, on both halves. Setting a contradicting value in the stage's `server_args_overrides` is rejected at configuration time and names the argument.

`--pd-stage` moves only the stage it names. Stages that feed it, including the image and audio encoders, keep the GPU they already had. On a two-GPU multimodal deployment the prefill card therefore carries the encoders as well.

## Both halves on one GPU

The halves may share a card. What PD needs is the process split, and that happens either way: the prefill step leaves the decode scheduler thread whichever card it runs on. Sharing a card also makes PD runnable on a one-GPU box.

Declare each half's share of the card, in the pipeline config:

```yaml
pd_disaggregation:
  prefill: {gpu: 0, memory_fraction: 0.25}
  decode:  {gpu: 0, memory_fraction: 0.65}
```

Use that share rather than `mem_fraction_static` when the halves share a card. `total_gpu_memory_fraction` is a fraction of total physical memory, so it does not depend on which half loads first. `mem_fraction_static` is computed against memory free at load time, and the two halves race for one startup lock: measured on one H200, whichever half won the lock sized itself to 780,987 KV tokens and the other failed to start. With the share form both halves sized to 21,502 KV tokens under either lock order.

Prefer the share over an absolute `max_total_tokens` here. `max_total_tokens` is applied as a minimum against the profiled capacity, so a cap larger than what the later-loading half can profile stops binding on that half and the pair silently reverts to order-dependent sizing.

## Budget for the relay pool

Splitting a multimodal stage moves the `mm_aggregate` to prefill edge across a process boundary. The CUDA-IPC relay then allocates a pool on the prefill GPU that a colocated deployment never allocates. The default is 1024 MB.

The relay allocates it on the first payload that crosses the boundary, not at startup. A `mem_fraction_static` copied from a colocated deployment therefore passes launch and fails on the first image request. Measured on one H200: the prefill half at `mem_fraction_static=0.87` plus the encoder process reached 139.5 GiB of 139.80 GiB and failed a 750 MiB allocation on the first image request. 0.80 left room.

## What to expect

Measured on two H200s, Qwen3-Omni thinker, CUDA graph on, against two colocated replicas on the same two cards. Full method and data in [PD versus colocated at equal GPU count](../developer_reference/pd_vs_colocated.md).

**Interference goes away.** The ratio of the median inter-token gap that overlaps a prefill to the median gap that does not:

| | Colocated | PD |
| --- | --- | --- |
| Text | 5.9x to 8.5x | 1.00x to 1.54x |
| Image | 8.4x to 20.5x | 0.97x to 1.02x |

On the image arm a prefill in flight on the other card is statistically indistinguishable from no prefill at all.

**Throughput does not.** On text, PD saturated by an offered 16 requests per second and held about 10.6 achieved. Colocated still tracked the offered rate at 44 with full admission, so its ceiling was not established and the 3.5x gap at that rate is a lower bound.

The reason is that prefill is a small share of this workload's GPU work. At 42 prompt tokens and 128 output tokens, a 58 ms first forward sits against a 2.2 s decode tail, about 2.6%. A 1:1 pair splits the hardware evenly against that split, so the prefill card measured 10.1% mean utilization against 67.9% on a colocated card. Prefill share is governed by prompt length divided by output length, so it rises with longer prompts and shorter outputs.

**Overload appears as latency, not rejection.** With no bound the decode half accumulates. At an offered 16 it held about 437 requests against `max_running_requests=64`, and a request took 40.96 s against 2.29 s colocated, while the admission rate read 100% throughout. An operator watching the admission rate sees a healthy system delivering 41-second latency.

Three settings bound it, and they answer different questions. Pick by which question you can answer for your deployment.

| Setting | Question it answers | What it does |
| --- | --- | --- |
| `SGLANG_REQ_WAITING_TIMEOUT` | how long may one request wait | Drops a request that has waited longer, with HTTP 503 and `Request waiting timeout reached.` |
| `--max-queued-requests` | how many may wait | Rejects arrivals beyond the bound with `The request queue is full.` |
| `--enable-priority-scheduling` | which requests survive | Evicts the least-preferred queued request instead of the arrival, and only when the arrival ranks strictly higher |

Prefer the timeout as the primary bound for interactive serving. Its unit is a duration you already know, namely your client's own timeout, whereas a queue length converts to a wait only through the current service rate: 16 queued requests is two seconds at one load and forty at another. The timeout also tracks load on its own, admitting fewer as service slows, with no retuning.

Keep `--max-queued-requests` as the memory bound. A duration does not cap how many requests are resident, because a burst can leave many of them still inside their deadline.

Leave both unset for offline batch work, where every request should eventually complete and nobody is waiting. `models/qwen3_tts` sets `max_queued_requests` to 16 for its generation stage.

Set priorities when one deployment serves both interactive and batch traffic. Then a single bound does the right thing for both, because the batch requests are the ones evicted under pressure.

## Limits today

- `tp_size` must be 1. A two-GPU half parses and places, and is then rejected when the runtime binds.
- `page_size` is forced to 1 and the radix cache is disabled, so prefix reuse is unavailable on a PD stage.
- One prefill half pairs with one decode half. Ratios other than 1:1 are not configurable.
- Both halves must be on the same node.
