# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 autoregressive decode engine (batch=1, full-history KV cache).

Prefill over the prompt, sample one 9-codebook frame per step, feed it back.
Output frames are in the DELAYED codebook space; the vocoder strips the delay.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang_omni.models.zonos2.modeling import Zonos2Model, load_zonos2_model
from sglang_omni.models.zonos2.sampler import TTSSamplingParams, sample_tts


class Zonos2Engine:
    def __init__(self, model: Zonos2Model):
        self.model = model
        self.cfg = model.cfg
        self.device = model.multi_output.device

    @classmethod
    def from_pretrained(
        cls, model_path: str, device: str = "cuda", dtype=torch.bfloat16
    ):
        return cls(load_zonos2_model(model_path, device=device, dtype=dtype))

    @torch.no_grad()
    def generate_one(
        self,
        input_ids: torch.Tensor,
        params: Optional[TTSSamplingParams] = None,
        speaker_emb: Optional[torch.Tensor] = None,
        speaker_position: int = 0,
    ) -> dict:
        """input_ids: [P, 10] int prompt rows. Returns {audio_tokens [[9]...], eos_frame}."""
        params = params or TTSSamplingParams()
        model = self.model
        device = self.device
        n = model.n_codebooks
        eoa = self.cfg.eoa_id
        text_pad = self.cfg.text_vocab

        ids = input_ids.to(device=device, dtype=torch.long)
        P = ids.shape[0]
        kv = [dict() for _ in model.layers]
        pos = torch.arange(P, device=device)

        spk = spk_pos = None
        if speaker_emb is not None:
            spk = speaker_emb.to(device=device, dtype=torch.float32).reshape(1, -1)
            spk_pos = torch.tensor(
                [int(speaker_position)], device=device, dtype=torch.long
            )

        logits = model(
            ids, pos, speaker_emb=spk, speaker_positions=spk_pos, kv_caches=kv
        )
        last = logits[-1:]

        gen = None
        if params.seed is not None:
            gen = torch.Generator(device=device).manual_seed(int(params.seed))

        cb_size = self.cfg.codebook_size
        frames: list[list[int]] = []
        # The penalty window spans into the prompt, matching the reference; eoa/pad
        # codes are filtered out by value below.
        rep_hist: list[list[int]] = ids[:, :n].tolist()
        eos_frame: Optional[int] = None
        countdown = 0

        for step in range(params.max_tokens):
            rep_ids = None
            if params.repetition_penalty != 1.0 and rep_hist:
                window = rep_hist[-params.repetition_window :]
                w = torch.tensor(window, device=device, dtype=torch.long).T
                rep = torch.full((n, w.shape[1]), -1, device=device, dtype=torch.long)
                rc = min(params.repetition_codebooks, n)
                # only penalize real codes; eoa/pad (>= codebook_size) never repeat-penalized
                rep[:rc] = torch.where(w[:rc] < cb_size, w[:rc], rep[:rc])
                rep_ids = rep.unsqueeze(0)

            codes = sample_tts(
                last.float(), params, rep_token_ids=rep_ids, generator=gen
            )[0]
            frame = codes.tolist()
            frames.append(frame)
            rep_hist.append(frame)

            if eos_frame is None:
                eos_cols = [frame[i] == eoa for i in range(n)]
                if any(eos_cols):
                    max_eos_cb = max(i for i, e in enumerate(eos_cols) if e)
                    eos_frame = max(0, step - max_eos_cb)
                    countdown = n + 1
            if eos_frame is not None and countdown > 0:
                countdown -= 1
            if not params.ignore_eos and eos_frame is not None and countdown <= 0:
                break

            # feedback row is the 9 sampled codes followed by a text-pad column
            next_row = torch.cat(
                [codes, torch.tensor([text_pad], device=device, dtype=torch.long)]
            ).unsqueeze(0)
            p = torch.tensor([P + step], device=device)
            last = model(next_row, p, kv_caches=kv)

        return {"audio_tokens": frames, "eos_frame": eos_frame}
