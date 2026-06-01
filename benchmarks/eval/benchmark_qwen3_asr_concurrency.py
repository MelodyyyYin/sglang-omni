# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR concurrency-scaling benchmark on SeedTTS EN (issue #646).

Sweeps ASR transcription fan-out (concurrency) against a *running* Qwen3-ASR
SGLang Omni router and reports, for each concurrency level, the metrics tracked
in issue #646: corpus/per-sample WER, wall-clock, throughput, latency
percentiles, RTF, and per-worker routing balance. This produces the repeatable
concurrency-scaling data the issue's acceptance criteria ask for, and lets us
decide the right ASR fan-out for the small correctness gate vs. the full
SeedTTS EN transcription / WER workloads.

This script transcribes the SeedTTS *reference* clips directly (no TTS
generation step), so it isolates ASR behavior from TTS.

Usage:

    # Download the test set once:
    python -m benchmarks.dataset.prepare --dataset seedtts

    # Launch Qwen3-ASR (DP=2 to match TTS CI):
    python -m sglang_omni.cli serve \
        --model-path Qwen/Qwen3-ASR-1.7B \
        --dp-size 2 \
        --port 8000

    # Sweep the issue's matrix (3 repeats each) over the full SeedTTS EN set:
    python -m benchmarks.eval.benchmark_qwen3_asr_concurrency \
        --port 8000 \
        --concurrencies 1,2,4,8,16,32,64 \
        --repeats 3

    # Quick check on the 20-sample correctness subset:
    python -m benchmarks.eval.benchmark_qwen3_asr_concurrency \
        --port 8000 --max-samples 20 --concurrencies 2,32 --repeats 3
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import time
from dataclasses import dataclass, field

import requests
from jiwer import process_words

from benchmarks.benchmarker.utils import get_wav_duration
from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.tasks.tts import (
    QWEN3_ASR_MODEL_PATH,
    load_router_asr,
    normalize_text,
    transcribe,
)

DEFAULT_CONCURRENCIES = "1,2,4,8,16,32,64"


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _fetch_worker_snapshot(host: str, port: int) -> dict | None:
    """Best-effort read of the router /workers snapshot (None if unavailable)."""
    try:
        response = requests.get(
            f"http://{host}:{port}/workers",
            timeout=10,
            proxies={"http": None, "https": None},
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _worker_routed_counts(snapshot: dict | None) -> list[int]:
    if not snapshot:
        return []
    return [int(w.get("routed_requests", 0)) for w in snapshot.get("workers", [])]


def _worker_delta(before: dict | None, after: dict | None) -> dict:
    """Routed/successful/failed deltas and per-worker routed balance."""
    if not before or not after:
        return {}

    def _by_id(snapshot: dict, key: str) -> dict[str, int]:
        return {
            str(w.get("display_id")): int(w.get(key, 0))
            for w in snapshot.get("workers", [])
        }

    out: dict[str, object] = {}
    for key in ("routed_requests", "successful_requests", "failed_requests"):
        before_by_id = _by_id(before, key)
        after_by_id = _by_id(after, key)
        deltas = {
            wid: after_by_id.get(wid, 0) - before_by_id.get(wid, 0)
            for wid in after_by_id
        }
        out[f"total_{key}"] = sum(deltas.values())
        if key == "routed_requests":
            out["per_worker_routed"] = deltas
    return out


@dataclass
class RepeatResult:
    concurrency: int
    repeat: int
    evaluated: int
    total: int
    skipped: int
    corpus_wer: float
    per_sample_wer_mean: float
    per_sample_wer_p95: float
    per_sample_wer_max: float
    wall_clock_s: float
    throughput_samples_per_s: float
    latency_mean_s: float
    latency_p50_s: float
    latency_p95_s: float
    latency_p99_s: float
    rtf_mean: float
    rtf_p95: float
    rtf_p99: float
    worker: dict = field(default_factory=dict)


def _run_one(
    *,
    asr: dict,
    host: str,
    port: int,
    samples: list[SampleInput],
    lang: str,
    concurrency: int,
    repeat: int,
) -> RepeatResult:
    audio_durations: dict[str, float] = {}
    for sample in samples:
        with open(sample.ref_audio, "rb") as handle:
            audio_durations[sample.sample_id] = get_wav_duration(handle.read())

    def _transcribe(sample: SampleInput) -> tuple[str, str, float]:
        start = time.perf_counter()
        text = transcribe(asr, sample.ref_audio, lang, "cuda")
        return sample.sample_id, text, time.perf_counter() - start

    hyp_by_id: dict[str, str] = {}
    latencies_s: list[float] = []
    rtfs: list[float] = []
    errors = 0

    before = _fetch_worker_snapshot(host, port)
    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_transcribe, sample) for sample in samples]
        for future in concurrent.futures.as_completed(futures):
            try:
                sample_id, text, latency_s = future.result()
            except Exception as exc:  # noqa: BLE001 - report and continue
                errors += 1
                print(f"  [conc={concurrency} rep={repeat}] request failed: {exc}")
                continue
            hyp_by_id[sample_id] = text
            latencies_s.append(latency_s)
            duration = audio_durations.get(sample_id, 0.0)
            if duration > 0:
                rtfs.append(latency_s / duration)
    wall_clock_s = time.perf_counter() - wall_start
    after = _fetch_worker_snapshot(host, port)

    ref_norms: list[str] = []
    hyp_norms: list[str] = []
    sample_wers: list[float] = []
    for sample in samples:
        if sample.sample_id not in hyp_by_id:
            continue
        ref_norm = normalize_text(sample.ref_text, lang)
        hyp_norm = normalize_text(hyp_by_id[sample.sample_id], lang)
        ref_norms.append(ref_norm)
        hyp_norms.append(hyp_norm)
        sample_wers.append(process_words(ref_norm, hyp_norm).wer)

    corpus_wer = process_words(ref_norms, hyp_norms).wer if ref_norms else 0.0
    evaluated = len(latencies_s)
    return RepeatResult(
        concurrency=concurrency,
        repeat=repeat,
        evaluated=evaluated,
        total=len(samples),
        skipped=len(samples) - evaluated,
        corpus_wer=corpus_wer,
        per_sample_wer_mean=statistics.mean(sample_wers) if sample_wers else 0.0,
        per_sample_wer_p95=_percentile(sample_wers, 95),
        per_sample_wer_max=max(sample_wers, default=0.0),
        wall_clock_s=wall_clock_s,
        throughput_samples_per_s=evaluated / wall_clock_s if wall_clock_s else 0.0,
        latency_mean_s=statistics.mean(latencies_s) if latencies_s else 0.0,
        latency_p50_s=_percentile(latencies_s, 50),
        latency_p95_s=_percentile(latencies_s, 95),
        latency_p99_s=_percentile(latencies_s, 99),
        rtf_mean=statistics.mean(rtfs) if rtfs else 0.0,
        rtf_p95=_percentile(rtfs, 95),
        rtf_p99=_percentile(rtfs, 99),
        worker=_worker_delta(before, after),
    )


def _aggregate(results: list[RepeatResult]) -> dict:
    """Mean/best/worst across repeats for the headline metrics."""

    def _stat(attr: str) -> dict:
        values = [getattr(r, attr) for r in results]
        return {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "concurrency": results[0].concurrency,
        "repeats": len(results),
        "evaluated": results[0].evaluated,
        "total": results[0].total,
        "skipped": results[0].skipped,
        "corpus_wer": _stat("corpus_wer"),
        "per_sample_wer_max": _stat("per_sample_wer_max"),
        "wall_clock_s": _stat("wall_clock_s"),
        "throughput_samples_per_s": _stat("throughput_samples_per_s"),
        "latency_mean_s": _stat("latency_mean_s"),
        "latency_p95_s": _stat("latency_p95_s"),
        "latency_p99_s": _stat("latency_p99_s"),
        "rtf_mean": _stat("rtf_mean"),
        "rtf_p95": _stat("rtf_p95"),
        "per_repeat": [vars(r) for r in results],
    }


def _print_table(aggregates: list[dict]) -> None:
    header = (
        "| conc | reps | wall(s) mean | thrpt mean | thrpt best | "
        "lat mean(s) | lat p95(s) | rtf mean | rtf p95 | corpus WER | max WER |"
    )
    sep = "|---:" * 11 + "|"
    print("\n" + header)
    print(sep)
    for agg in aggregates:
        print(
            f"| {agg['concurrency']} | {agg['repeats']} "
            f"| {agg['wall_clock_s']['mean']:.3f} "
            f"| {agg['throughput_samples_per_s']['mean']:.3f} "
            f"| {agg['throughput_samples_per_s']['max']:.3f} "
            f"| {agg['latency_mean_s']['mean']:.3f} "
            f"| {agg['latency_p95_s']['mean']:.3f} "
            f"| {agg['rtf_mean']['mean']:.4f} "
            f"| {agg['rtf_p95']['mean']:.4f} "
            f"| {agg['corpus_wer']['max']:.4f} "
            f"| {agg['per_sample_wer_max']['max']:.4f} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port of the running Qwen3-ASR SGLang Omni router.",
    )
    parser.add_argument(
        "--meta",
        default=DATASETS["seedtts"],
        help="SeedTTS source (HF repo id or local meta.lst).",
    )
    parser.add_argument("--lang", default="en", choices=["en", "zh"])
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit samples (0 = full SeedTTS set; 1088 for EN).",
    )
    parser.add_argument(
        "--concurrencies",
        default=DEFAULT_CONCURRENCIES,
        help="Comma-separated ASR concurrency levels to sweep.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--model-path",
        default=QWEN3_ASR_MODEL_PATH,
        help="ASR model id served by the router.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one discarded warmup pass before timing each concurrency.",
    )
    parser.add_argument(
        "--output",
        default="qwen3_asr_concurrency_results.json",
        help="Where to write the full JSON results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    concurrencies = [int(c) for c in args.concurrencies.split(",") if c.strip()]
    max_samples = args.max_samples if args.max_samples > 0 else None

    samples = load_seedtts_samples(args.meta, max_samples=max_samples, split=args.lang)
    print(
        f"Loaded {len(samples)} SeedTTS {args.lang} samples; "
        f"sweeping concurrency={concurrencies} x {args.repeats} repeats "
        f"against {args.host}:{args.port} ({args.model_path})"
    )

    asr = load_router_asr(args.port, model_path=args.model_path)
    aggregates: list[dict] = []
    for concurrency in concurrencies:
        if args.warmup:
            print(f"[conc={concurrency}] warmup pass ...")
            _run_one(
                asr=asr,
                host=args.host,
                port=args.port,
                samples=samples,
                lang=args.lang,
                concurrency=concurrency,
                repeat=0,
            )
        repeats: list[RepeatResult] = []
        for repeat in range(1, args.repeats + 1):
            result = _run_one(
                asr=asr,
                host=args.host,
                port=args.port,
                samples=samples,
                lang=args.lang,
                concurrency=concurrency,
                repeat=repeat,
            )
            repeats.append(result)
            print(
                f"[conc={concurrency} rep={repeat}] "
                f"wall={result.wall_clock_s:.3f}s "
                f"thrpt={result.throughput_samples_per_s:.3f}/s "
                f"lat_mean={result.latency_mean_s:.3f}s "
                f"lat_p95={result.latency_p95_s:.3f}s "
                f"rtf_mean={result.rtf_mean:.4f} "
                f"corpus_wer={result.corpus_wer:.4f} "
                f"skipped={result.skipped}"
            )
            if result.worker.get("per_worker_routed"):
                print(f"    routed per worker: {result.worker['per_worker_routed']}")
        aggregates.append(_aggregate(repeats))

    _print_table(aggregates)

    payload = {
        "config": {
            "host": args.host,
            "port": args.port,
            "meta": args.meta,
            "lang": args.lang,
            "model_path": args.model_path,
            "num_samples": len(samples),
            "concurrencies": concurrencies,
            "repeats": args.repeats,
            "warmup": args.warmup,
        },
        "results": aggregates,
    }
    output_path = os.path.abspath(args.output)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
