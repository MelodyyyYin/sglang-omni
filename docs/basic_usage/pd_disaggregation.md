# Prefill/Decode Disaggregation

> TL;DR: PD disaggregation splits one stage into a prefill process and a decode process, so a prefill no longer stalls the decoding already in flight. It removes that interference almost completely, on any placement, including both halves on one GPU. How much hardware you then give each half is a separate, continuous choice, and it is the choice that decides what the split costs and what it can return. One prefill half against one decode half on two cards is the configuration measured here: it has parity with two colocated replicas as its throughput ceiling, and on Qwen3-Omni thinker it currently lands below that.

A colocated replica runs prefill and decode on one card and one scheduler thread. A prefill therefore blocks the decode steps of every request already running. On Qwen3-Omni that cost is large, and it is worst on images, where a 6127-token prompt cannot be chunked.

PD disaggregation moves prefill into its own process. The prompt KV and the request's continuation state transfer once, over CUDA IPC, and decoding continues on the other half. Where that other half runs, and how much of a card it gets, is what the rest of this page is about.

## What the split actually gives you

PD here is not one topology. It is a resource split you set continuously, at three layers, and each layer answers a different question.

| Layer | How you set it | What it decides |
| --- | --- | --- |
| Which devices each half gets | `thinker=0:1`, `thinker=0:0`, `thinker=0,1:2,3` | whether the halves are separated in execution only, or in hardware as well |
| Each half's share of a device | `thinker=0@0.25:0@0.65` | how one card divides between the two halves, as a continuous fraction |
| Each half's runtime settings | `pd_disaggregation.prefill.server_args` | lets a half be tuned for its own work rather than for the average of both |

The first layer is the one people usually read as the whole feature, and it is the one with a degenerate case worth knowing: **both halves on one GPU still separates them.** What PD needs is the process split, and a prefill step leaves the decode scheduler thread whichever card it runs on. Giving prefill a second card is an additional, separable decision about hardware, not what makes PD work.

That is why the split is a parameter rather than a topology. `thinker=0:0` and `thinker=0:1` differ in how much hardware you spend, not in whether the halves are separated.

## Choosing a configuration

Three questions decide whether and how to split. A fourth decides what the split is optimizing, and it selects a configuration rather than ruling PD in or out.

### 1. Is prefill a large enough share of your work?

Prefill share is governed by prompt length divided by output length.

| Prompt / output tokens | Prefill share |
| --- | --- |
| 42 / 128 | about 2.6% |
| 6944 / 128 | about 9% |
| 42 / 8 | about 42% |

This sets what a *second card* for prefill can be worth. Splitting across two cards divides the hardware evenly; against a 2.6% share that leaves the prefill card about 90% idle, which is what the equal-GPU-count run measured. Long prompts with short outputs — document QA, long-audio transcription, classification, scoring — are the shape that earns a second card. Short prompts with long outputs are not.

A low share does not rule out splitting. It argues for spending less hardware on it: the same-GPU form separates the halves without a second card at all.

### 2. Which overload behaviour do you want?

Decide it, because the default is an unbounded queue and it surfaces as latency rather than as an error. See [What to expect](#what-to-expect) for the three settings and which question each answers.

### 3. Does the prefill card have room for what moves onto it?

`--pd-stage` moves only the stage it names. The encoders stay where they were, and the CUDA-IPC relay pool appears on the prefill card. Budget for both. This is the most common way a first PD launch fails.

### 4. Are you optimizing throughput, or inter-token stability?

Both are real goals and they select different configurations.

**Inter-token stability.** A colocated replica pays 5.9x to 8.5x on the inter-token gap that overlaps a prefill on text, and 8.4x to 20.5x on images. PD brings that to 1.0x: on the image arm a prefill in flight on the other card is statistically indistinguishable from no prefill at all. Images are worst because a 6127-token image prompt cannot be chunked — `_needs_full_prefill` forces the whole prompt through one step and no decode work batches with it. For a stream with a repeating deadline, such as speech output or a full-duplex voice turn, this jitter *is* the SLO and a throughput number does not describe it. Any split addresses it, including the same-GPU form.

**Throughput.** One prefill half against one decode half, on two cards, has parity as its ceiling. That is arithmetic, not a defect: a card doing both jobs has the harmonic mean of its prefill-only and decode-only capacities, a 1:1 pair has the minimum of the two, and the minimum never exceeds the harmonic mean, with equality only when the halves are exactly balanced. So 1:1 across two cards is the wrong configuration to reach for when throughput is the goal — not because splitting is wrong, but because that particular ratio cannot exceed parity even in theory.

The configurations whose arithmetic differs are ratios other than 1:1, and both halves on one card. Neither has been measured here yet. Ratios other than 1:1 are not configurable today.

The measured gap is also larger than the arithmetic predicts: PD's decode card carries about a 2.9x handicap while its prefill card carries none. Four candidates can account for it and the data cannot yet separate them — the forced `page_size=1`, the forced `disable_radix_cache`, the absence of mixed-chunk batching on the decode side, and the cost of receiving KV over CUDA IPC. Three of those four are current constraints of this implementation rather than properties of disaggregation, so expect this number to move.

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
- `page_size` is forced to 1 and the radix cache is disabled on both halves, so prefix reuse is unavailable on a PD stage. A workload built on a long shared prefix — a fixed system prompt, a few-shot preamble, a reference audio prefix — loses that reuse, and the loss does not appear in a throughput comparison run with unique prompts. This is a constraint of the current implementation, not of disaggregation.
- One prefill half pairs with one decode half. Ratios other than 1:1 are not configurable.
- Both halves must be on the same node.
