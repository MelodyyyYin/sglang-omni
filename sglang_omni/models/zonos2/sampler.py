# SPDX-License-Identifier: Apache-2.0
"""Per-codebook TTS sampler.

Order: repetition penalty, temperature, top-k, softmax, top-p, min-p,
multinomial (argmax when temperature <= 0). Logits arrive soft-capped.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from sglang_omni.models.zonos2.text_frontend import TTSSamplingParams


def _apply_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0 or p >= 1.0:
        return probs
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort = probs_sort.masked_fill(mask, 0.0)
    probs = probs.scatter(-1, probs_idx, probs_sort)
    return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def _apply_min_p(probs: torch.Tensor, min_p: float) -> torch.Tensor:
    if min_p <= 0.0:
        return probs
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    probs = probs.masked_fill(probs < (min_p * top_probs), 0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def _apply_repetition_penalty(
    logits: torch.Tensor,
    rep_token_ids: Optional[torch.Tensor],
    penalties: Optional[torch.Tensor],
) -> torch.Tensor:
    """Penalize per-codebook repeats. logits (B,C,V), rep_token_ids (B,C,W)."""
    if rep_token_ids is None or penalties is None or rep_token_ids.numel() == 0:
        return logits
    B, C, V = logits.shape
    safe = rep_token_ids.clamp(min=0, max=V - 1).long()
    valid = (rep_token_ids >= 0) & (rep_token_ids < V)
    counts = torch.zeros((B, C, V), dtype=torch.int32, device=logits.device)
    counts.scatter_add_(-1, safe, valid.to(torch.int32))
    repeated = counts > 0
    pen = penalties.view(B, 1, 1).clamp(min=1.0)
    adjusted = torch.where(logits > 0, logits / pen, logits * pen)
    return torch.where(repeated, adjusted, logits)


@torch.no_grad()
def sample_tts(
    logits: torch.Tensor,
    params: TTSSamplingParams,
    rep_token_ids: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """logits (B, C, V) soft-capped -> sampled codes (B, C) int64."""
    B, C, V = logits.shape
    device = logits.device
    penalties = (
        torch.full((B,), params.repetition_penalty, device=device)
        if rep_token_ids is not None and params.repetition_penalty != 1.0
        else None
    )
    logits = _apply_repetition_penalty(logits, rep_token_ids, penalties)

    if params.temperature <= 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / max(params.temperature, 1e-8)
    flat = logits.view(B * C, V)
    if 0 < params.top_k < V:
        values, _ = torch.topk(flat, params.top_k, dim=-1)
        kth = values[..., -1].unsqueeze(-1)
        flat = flat.masked_fill(flat < kth, float("-inf"))
    probs = F.softmax(flat, dim=-1)
    if 0.0 < params.top_p < 1.0:
        probs = _apply_top_p(probs, params.top_p)
    if params.min_p > 0.0:
        probs = _apply_min_p(probs, params.min_p)

    invalid = probs.sum(dim=-1) <= 0
    if bool(invalid.any()):
        greedy = flat.argmax(dim=-1)
        fb = torch.zeros_like(probs)
        fb.scatter_(-1, greedy.unsqueeze(-1), 1.0)
        probs = torch.where(invalid.unsqueeze(-1), fb, probs)

    nxt = torch.multinomial(probs, num_samples=1, generator=generator)
    return nxt.view(B, C)
