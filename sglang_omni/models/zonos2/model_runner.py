# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 model runner for OmniScheduler.

Each step the radix-cached backbone yields one hidden state per request; the
runner then runs the multi-codebook head, samples 9 DAC codes (per-codebook),
advances the any-of-9 EOS state machine, and stages the next frame's summed
input embedding. There is no frame-local loop. Decode feeds ``input_embeds``
directly (the staged feedback), and the appended radix id is a content hash of
the frame so batched requests never alias in the cache.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.zonos2.radix_hash import EOS_SENTINEL, poly_row_hash
from sglang_omni.models.zonos2.sampler import sample_tts


class Zonos2ModelRunner(ModelRunner):
    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self._outbox: Any | None = None

    def set_stream_outbox(self, outbox: Any) -> None:
        self._outbox = outbox

    # ---- input embeddings ----

    def custom_prefill_forward(self, forward_batch, schedule_batch, requests):
        del schedule_batch
        forward_batch.input_embeds = self._build_prefill_embeds(forward_batch, requests)
        return None

    def before_decode(
        self, forward_batch, schedule_batch, requests, *, is_lookahead=False
    ):
        del schedule_batch, is_lookahead
        dim = self.model.config.dim
        rows = []
        for sr in requests:
            q = sr.data.pending_feedback_queue
            rows.append(
                q.popleft()
                if q
                else torch.zeros(dim, device=self.model.device, dtype=self.model.dtype)
            )
        forward_batch.input_embeds = torch.stack(rows, dim=0).to(
            device=self.model.device, dtype=self.model.dtype
        )

    def _build_prefill_embeds(self, forward_batch, requests) -> torch.Tensor:
        model = self.model
        pieces = []
        for sr in requests:
            data = sr.data
            req = data.req
            prefix_len = len(req.prefix_indices)
            req_len = int(req.extend_input_len)
            rows = data.prompt_rows[prefix_len : prefix_len + req_len].to(model.device)
            emb = model.embed_frames(rows)
            if data.speaker_emb is not None:
                pos = int(data.speaker_position) - prefix_len
                if 0 <= pos < req_len:
                    s = model.speaker_lda_projection(
                        data.speaker_emb.to(
                            model.device, model.speaker_lda_projection.weight.dtype
                        )
                    )
                    s = model.speaker_projection(
                        s.to(model.speaker_projection.weight.dtype)
                    )
                    emb[pos] = s.to(emb.dtype)
            pieces.append(emb)
        return torch.cat(pieces, dim=0).to(device=model.device, dtype=model.dtype)

    # ---- frame collection (head + sample + EOS + feedback) ----

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        if bool(getattr(schedule_batch, "is_prefill_only", False)):
            return
        self._collect_frame(
            result, forward_batch, schedule_batch, requests, is_prefill=True
        )

    def post_decode(self, result, forward_batch, schedule_batch, requests):
        self._collect_frame(
            result, forward_batch, schedule_batch, requests, is_prefill=False
        )

    def _last_token_hidden(self, hidden, forward_batch, is_prefill) -> torch.Tensor:
        if not is_prefill:
            return hidden
        lens = forward_batch.extend_seq_lens
        idx = torch.cumsum(lens.to(hidden.device, torch.long), dim=0) - 1
        return hidden[idx]

    def _collect_frame(
        self, result, forward_batch, schedule_batch, requests, *, is_prefill
    ):
        model = self.model
        hidden = result.logits_output.hidden_states
        hidden = self._last_token_hidden(hidden, forward_batch, is_prefill)
        logits = model.compute_logits(hidden)  # [B, 9, 1026], bf16
        n = model.n_codebooks
        eoa = model.config.eoa_id
        text_pad = model.config.text_vocab
        cb_size = model.config.codebook_size

        next_ids = torch.empty(len(requests), dtype=torch.long, device=hidden.device)
        for b, sr in enumerate(requests):
            data = sr.data
            rep_ids = self._rep_window(data, n, cb_size, hidden.device)
            gen = data.generator
            codes = sample_tts(
                logits[b : b + 1].float(),
                data.params,
                rep_token_ids=rep_ids,
                generator=gen,
            )[0]
            frame = codes.tolist()
            data.output_codes.append(codes.detach().to("cpu"))
            data.rep_hist.append(frame)

            step = data.generation_step
            if data.eos_frame is None:
                hits = [frame[i] == eoa for i in range(n)]
                if any(hits):
                    data.eos_frame = max(
                        0, step - max(i for i, h in enumerate(hits) if h)
                    )
                    data.eos_countdown = n + 1
            finished = False
            if data.eos_frame is not None:
                if data.eos_countdown > 0:
                    data.eos_countdown -= 1
                if data.eos_countdown <= 0:
                    finished = True
            data.generation_step += 1

            # feedback embedding for the next step (9 codes + text-pad column)
            row = torch.cat(
                [codes, torch.tensor([text_pad], device=codes.device, dtype=torch.long)]
            ).unsqueeze(0)
            data.pending_feedback_queue.append(model.embed_frames(row)[0].detach())

            key = int(poly_row_hash(row.cpu())[0]) if not finished else EOS_SENTINEL
            next_ids[b] = key

        result.next_token_ids = next_ids
        schedule_batch.output_ids = next_ids

    def _rep_window(self, data, n, cb_size, device):
        if data.params.repetition_penalty == 1.0 or not data.rep_hist:
            return None
        window = data.rep_hist[-data.params.repetition_window :]
        w = torch.tensor(window, device=device, dtype=torch.long).T
        rep = torch.full((n, w.shape[1]), -1, device=device, dtype=torch.long)
        rc = min(data.params.repetition_codebooks, n)
        rep[:rc] = torch.where(w[:rc] < cb_size, w[:rc], rep[:rc])
        return rep.unsqueeze(0)
