# SPDX-License-Identifier: Apache-2.0
"""Self-contained PyTorch ZONOS2 model and model.pth loader.

Engine-independent port of the Zyphra reference forward, with plain attention so
it can run full-sequence prefill and a KV-cached AR decode loop.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang_omni.models.zonos2.hf_config import Zonos2Config


def _rms_norm(
    x: torch.Tensor, weight: Optional[torch.Tensor], eps: float
) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    return cap * torch.tanh(x / cap)


class RotaryCache:
    """Interleaved (is_neox=False) RoPE matching the reference flashinfer call."""

    def __init__(self, head_dim: int, base: float, max_pos: int, device, dtype):
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
        t = torch.arange(max_pos, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.cos = freqs.cos()
        self.sin = freqs.sin()
        self.head_dim = head_dim

    def apply(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[positions]
        sin = self.sin[positions]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        out = torch.empty_like(x)
        out[..., 0::2] = rx1
        out[..., 1::2] = rx2
        return out


class Zonos2Attention(nn.Module):
    def __init__(self, cfg: Zonos2Config):
        super().__init__()
        self.h = cfg.dim
        self.nq = cfg.n_heads
        self.nkv = cfg.n_kv_heads
        self.hd = cfg.head_dim
        self.wq = nn.Parameter(torch.empty(self.nq * self.hd, self.h))
        self.wkv = nn.Parameter(torch.empty(2, self.nkv * self.hd, self.h))
        self.wo = nn.Parameter(torch.empty(self.h, self.nq * self.hd))
        self.gater = nn.Parameter(torch.empty(self.nq, self.h))
        self.temp = nn.Parameter(torch.empty(1, self.nq, 1))

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        rope: RotaryCache,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        T = x.shape[0]
        gate = torch.sigmoid(F.linear(x, self.gater))
        q = F.linear(x, self.wq).view(T, self.nq, self.hd)
        k = F.linear(x, self.wkv[0]).view(T, self.nkv, self.hd)
        v = F.linear(x, self.wkv[1]).view(T, self.nkv, self.hd)

        q = _rms_norm(q, None, 1e-6) * self.temp.abs().to(q.dtype)
        k = _rms_norm(k, None, 1e-6)
        q = rope.apply(q, positions)
        k = rope.apply(k, positions)

        if kv_cache is not None:
            if kv_cache.get("k") is not None:
                k = torch.cat([kv_cache["k"], k], dim=0)
                v = torch.cat([kv_cache["v"], v], dim=0)
            kv_cache["k"], kv_cache["v"] = k, v

        rep = self.nq // self.nkv
        kk = k.repeat_interleave(rep, dim=1)
        vv = v.repeat_interleave(rep, dim=1)
        qh = q.transpose(0, 1)
        kh = kk.transpose(0, 1)
        vh = vv.transpose(0, 1)
        # Causal mask only on square prefill; a decode step (q_len=1) attends all cached keys.
        is_causal = qh.shape[-2] == kh.shape[-2]
        o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=is_causal)
        o = o.transpose(0, 1)
        o = o * gate.unsqueeze(-1)
        o = o.reshape(T, self.nq * self.hd)
        return F.linear(o, self.wo)


class Zonos2DenseFFN(nn.Module):
    def __init__(self, cfg: Zonos2Config):
        super().__init__()
        self.inter = cfg.intermediate_size
        self.w_in = nn.Parameter(torch.empty(2, self.inter, cfg.dim))  # [up, gate]
        self.w_out = nn.Parameter(torch.empty(cfg.dim, self.inter))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = F.linear(x, self.w_in[0])
        gate = F.linear(x, self.w_in[1])
        return F.linear(up * F.silu(gate), self.w_out)


class Zonos2Router(nn.Module):
    """Sonic EDA router. Balancing biases shift expert selection (+legacy / -quantile)."""

    def __init__(self, cfg: Zonos2Config, layer_id: int):
        super().__init__()
        rd = cfg.moe_router_dim
        E = cfg.moe_n_experts
        self.top_k = cfg.topk_for_layer(layer_id)
        self.use_eda = cfg.uses_eda(layer_id)
        self.eps = cfg.norm_eps
        strat = (
            (cfg.moe_balancing_strategy or "legacy").strip().lower().replace("-", "_")
        )
        self.bias_sign = -1.0 if strat in ("current", "quantile", "qbalancing") else 1.0
        self.down_proj = nn.Linear(cfg.dim, rd, bias=True)
        self.router_mlp_0 = nn.Linear(rd, rd, bias=True)
        self.router_mlp_2 = nn.Linear(rd, rd, bias=True)
        self.router_mlp_4 = nn.Linear(rd, E, bias=False)
        self.rmsnorm_eda = nn.Parameter(torch.empty(rd))
        if self.use_eda:
            self.router_states_scale = nn.Parameter(torch.empty(rd))
        self.register_buffer("balancing_biases", torch.zeros(E), persistent=True)

    def forward(self, h: torch.Tensor, router_states: Optional[torch.Tensor]):
        h = self.down_proj(h)
        if self.use_eda and router_states is not None:
            h = h + router_states * self.router_states_scale
        rs_next = h.clone()
        h = _rms_norm(h, self.rmsnorm_eda, self.eps)
        x = F.gelu(self.router_mlp_0(h))
        x = F.gelu(self.router_mlp_2(x))
        logits = self.router_mlp_4(x)
        probs = torch.softmax(logits.float(), dim=-1)
        scores = probs + self.bias_sign * self.balancing_biases.float()
        _, ids = torch.topk(scores, self.top_k, dim=-1)
        weights = torch.gather(probs, -1, ids)
        return weights, ids.to(torch.long), rs_next


class Zonos2MoE(nn.Module):
    """Eager grouped MoE (loop over experts). FusedMoE kernel is a Phase-2 opt."""

    def __init__(self, cfg: Zonos2Config, layer_id: int):
        super().__init__()
        self.router = Zonos2Router(cfg, layer_id)
        self.E = cfg.moe_n_experts
        self.inter = cfg.intermediate_size
        # gate_up_weight stacks gate first then up
        self.gate_up_weight = nn.Parameter(torch.empty(self.E, 2 * self.inter, cfg.dim))
        self.down_weight = nn.Parameter(torch.empty(self.E, cfg.dim, self.inter))

    def forward(self, x: torch.Tensor, router_states):
        weights, ids, rs_next = self.router(x, router_states)
        out = torch.zeros_like(x)
        for e in range(self.E):
            sel = ids == e
            tok_mask = sel.any(dim=1)
            if not bool(tok_mask.any()):
                continue
            xe = x[tok_mask]
            gu = F.linear(xe, self.gate_up_weight[e])
            g, u = gu.chunk(2, dim=-1)
            ye = F.linear(F.silu(g) * u, self.down_weight[e])
            w = (weights * sel.to(weights.dtype)).sum(dim=1)[tok_mask].unsqueeze(-1)
            out[tok_mask] += (ye.float() * w.float()).to(out.dtype)
        return out, rs_next


class Zonos2Block(nn.Module):
    def __init__(self, cfg: Zonos2Config, layer_id: int):
        super().__init__()
        self.eps = cfg.norm_eps
        self.attention = Zonos2Attention(cfg)
        self.attention_norm = nn.Parameter(torch.empty(cfg.dim))
        self.ffn_norm = nn.Parameter(torch.empty(cfg.dim))
        self.is_moe = cfg.is_moe_layer(layer_id)
        self.feed_forward = (
            Zonos2MoE(cfg, layer_id) if self.is_moe else Zonos2DenseFFN(cfg)
        )

    def forward(self, x, residual, router_states, positions, rope, kv_cache):
        if residual is None:
            residual = x
            h = _rms_norm(x, self.attention_norm, self.eps)
        else:
            residual = x + residual
            h = _rms_norm(residual, self.attention_norm, self.eps)
        h = self.attention(h, positions, rope, kv_cache)
        residual = h + residual
        h = _rms_norm(residual, self.ffn_norm, self.eps)
        if self.is_moe:
            h, router_states = self.feed_forward(h, router_states)
        else:
            h = self.feed_forward(h)
            router_states = None
        return h, residual, router_states


class Zonos2Model(nn.Module):
    def __init__(self, cfg: Zonos2Config):
        super().__init__()
        self.cfg = cfg
        self.n_codebooks = cfg.n_codebooks
        self.audio_vocab = cfg.audio_vocab
        emb = []
        for _ in range(cfg.n_codebooks):
            emb.append(nn.Embedding(cfg.codebook_size + 2, cfg.dim))
        emb.append(nn.Embedding(cfg.text_vocab + 1, cfg.dim))
        self.embedders = nn.ModuleList(emb)
        self.eps = cfg.norm_eps
        self.speaker_lda_projection = nn.Linear(
            cfg.speaker_embedding_dim, cfg.speaker_lda_dim, bias=True
        )
        self.speaker_projection = nn.Linear(cfg.speaker_lda_dim, cfg.dim, bias=True)
        self.layers = nn.ModuleList([Zonos2Block(cfg, i) for i in range(cfg.n_layers)])
        self.out_norm = nn.Parameter(torch.empty(cfg.dim))
        self.multi_output = nn.Parameter(
            torch.empty(self.audio_vocab * self.n_codebooks, cfg.dim)
        )
        self._rope = None

    def _rope_cache(self, device, dtype):
        if self._rope is None:
            self._rope = RotaryCache(
                self.cfg.head_dim,
                self.cfg.rope_theta,
                self.cfg.max_seqlen,
                device,
                dtype,
            )
        return self._rope

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids [T, n_codebooks+1] -> summed embedding [T, dim]
        out = self.embedders[0](input_ids[:, 0].contiguous())
        for i in range(1, input_ids.shape[1]):
            out = out + self.embedders[i](input_ids[:, i].contiguous())
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        speaker_emb: Optional[torch.Tensor] = None,
        speaker_positions: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
    ) -> torch.Tensor:
        device = self.multi_output.device
        dtype = self.multi_output.dtype
        rope = self._rope_cache(device, dtype)
        x = self.embed(input_ids)
        if (
            speaker_emb is not None
            and speaker_positions is not None
            and speaker_emb.numel() > 0
        ):
            s = self.speaker_lda_projection(
                speaker_emb.to(self.speaker_lda_projection.weight.dtype)
            )
            s = self.speaker_projection(s.to(self.speaker_projection.weight.dtype))
            x = x.index_copy(0, speaker_positions, s.to(x.dtype))
        x = _rms_norm(x, None, self.eps)  # emb_norm is affine-free
        residual = None
        router_states = None
        for i, layer in enumerate(self.layers):
            kvc = kv_caches[i] if kv_caches is not None else None
            x, residual, router_states = layer(
                x, residual, router_states, positions, rope, kvc
            )
        h = _rms_norm(x + residual, self.out_norm, self.eps)
        logits = F.linear(h, self.multi_output)
        logits = logits.view(*logits.shape[:-1], self.n_codebooks, self.audio_vocab)
        if self.cfg.loss_softcap > 0:
            logits = softcap(logits, self.cfg.loss_softcap)
        return logits


@torch.no_grad()
def load_zonos2_model(
    model_path: str, device: str = "cuda", dtype=torch.bfloat16
) -> Zonos2Model:
    """Build the model from params.json and load model.pth into it."""
    import os

    from sglang_omni.models.zonos2.hf_config import load_zonos2_pretrained_config

    cfg = load_zonos2_pretrained_config(model_path)
    model = Zonos2Model(cfg).to(device=device, dtype=dtype)

    pth = (
        model_path
        if model_path.endswith((".pt", ".pth"))
        else os.path.join(model_path, "model.pth")
    )
    sd = torch.load(pth, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    # drop training-only router keys
    sd = {
        k: v
        for k, v in sd.items()
        if ".router.ent_denom" not in k and ".router.normalized_entropy" not in k
    }
    fixed = {}
    for k, v in sd.items():
        if ".parametrizations." in k and ".original" in k:
            k = k.replace(".parametrizations.", ".").replace(".original", "")
        fixed[k] = v
    sd = fixed

    used = set()

    def take(key):
        used.add(key)
        return sd[key].to(device=device, dtype=dtype)

    def take_f32(key):
        used.add(key)
        return sd[key].to(device=device, dtype=torch.float32)

    for i in range(cfg.n_codebooks + 1):
        model.embedders[i].weight.data.copy_(
            take(f"multi_embedder.embedders.{i}.weight")
        )
    model.out_norm.data.copy_(take("out_norm.weight"))
    model.multi_output.data.copy_(take("multi_output.weight"))
    if "speaker_lda_projection.weight" in sd:
        model.speaker_lda_projection.weight.data.copy_(
            take("speaker_lda_projection.weight")
        )
        if "speaker_lda_projection.bias" in sd:
            model.speaker_lda_projection.bias.data.copy_(
                take("speaker_lda_projection.bias")
            )
        else:
            model.speaker_lda_projection.bias.data.zero_()
    model.speaker_projection.weight.data.copy_(take("speaker_projection.weight"))
    model.speaker_projection.bias.data.copy_(take("speaker_projection.bias"))

    for i, layer in enumerate(model.layers):
        p = f"layers.{i}."
        a = layer.attention
        a.wq.data.copy_(take(p + "attention.wq.weight"))
        a.wkv.data.copy_(take(p + "attention.wkv.weight"))
        a.wo.data.copy_(take(p + "attention.wo.weight"))
        a.gater.data.copy_(take(p + "attention.gater.weight"))
        a.temp.data.copy_(take(p + "attention.temp"))
        layer.attention_norm.data.copy_(take(p + "attention_norm.weight"))
        layer.ffn_norm.data.copy_(take(p + "ffn_norm.weight"))
        ff = layer.feed_forward
        if layer.is_moe:
            # w13 interleaves gate/up rows; de-interleave then stack gate-first
            w13 = take(p + "feed_forward.experts.w13")
            gate = w13[:, 0::2, :]
            up = w13[:, 1::2, :]
            ff.gate_up_weight.data.copy_(torch.cat([gate, up], dim=1))
            ff.down_weight.data.copy_(take(p + "feed_forward.experts.w2"))
            r = ff.router
            r.down_proj.weight.data.copy_(
                take(p + "feed_forward.router.down_proj.weight")
            )
            r.down_proj.bias.data.copy_(take(p + "feed_forward.router.down_proj.bias"))
            r.router_mlp_0.weight.data.copy_(
                take(p + "feed_forward.router.router_mlp.0.weight")
            )
            r.router_mlp_0.bias.data.copy_(
                take(p + "feed_forward.router.router_mlp.0.bias")
            )
            r.router_mlp_2.weight.data.copy_(
                take(p + "feed_forward.router.router_mlp.2.weight")
            )
            r.router_mlp_2.bias.data.copy_(
                take(p + "feed_forward.router.router_mlp.2.bias")
            )
            r.router_mlp_4.weight.data.copy_(
                take(p + "feed_forward.router.router_mlp.4.weight")
            )
            r.rmsnorm_eda.data.copy_(take(p + "feed_forward.router.rmsnorm_eda.weight"))
            if r.use_eda:
                r.router_states_scale.data.copy_(
                    take(p + "feed_forward.router.router_states_scale")
                )
            r.balancing_biases.data.copy_(
                take_f32(p + "feed_forward.router.balancing_biases")
            )
        else:
            ff.w_in.data.copy_(take(p + "feed_forward.w_in.weight"))
            ff.w_out.data.copy_(take(p + "feed_forward.w_out.weight"))

    leftover = set(sd.keys()) - used
    if leftover:
        raise RuntimeError(
            f"Unconsumed checkpoint keys ({len(leftover)}): {sorted(leftover)[:20]}"
        )
    model.eval()
    return model
