# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 cross-stage payload contract (frozen, append-only).

Carried inside ``StagePayload.data`` as a plain dict via ``to_dict``/``from_dict``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ZONOS2_SAMPLE_RATE = 44100
N_CODEBOOKS = 9
# 9 audio codebook columns + 1 text column.
FRAME_WIDTH = 10


@dataclass
class Zonos2State:
    """Per-request state threaded through the ZONOS2 pipeline."""

    # request inputs (preprocessing)
    text: str = ""
    ref_audio: Any | None = None  # path / bytes / data-uri for voice cloning
    ref_text: str | None = None
    language: str | None = None
    speaking_rate: float | None = None
    conditioning: dict[str, Any] = field(default_factory=dict)

    # preprocessing output
    # (T, FRAME_WIDTH) rows: audio cols hold audio_pad_id, last col holds text/conditioning ids.
    input_ids: Any | None = None
    speaker_token_positions: list[int] = field(default_factory=lambda: [0])

    # speaker_encode output
    speaker_emb: Any | None = None  # (2048,) f32 CPU tensor, or None for zero-shot
    speaker_fingerprint: str | None = None  # stable hash for radix extra_key

    # tts_engine output
    audio_codes: Any | None = None  # delayed (T, 9) int tensor, pre-shear
    eos_frame: int | None = None

    # vocoder / bookkeeping
    sample_rate: int = ZONOS2_SAMPLE_RATE
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0

    # (de)serialization across the stage stream queue

    @staticmethod
    def _tensor_to_payload(value: Any) -> Any:
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return value

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "generation_kwargs": dict(self.generation_kwargs),
            "sample_rate": int(self.sample_rate),
            "speaker_token_positions": list(self.speaker_token_positions),
        }
        if self.ref_audio is not None:
            data["ref_audio"] = self.ref_audio
        if self.ref_text is not None:
            data["ref_text"] = self.ref_text
        if self.language is not None:
            data["language"] = self.language
        if self.speaking_rate is not None:
            data["speaking_rate"] = float(self.speaking_rate)
        if self.conditioning:
            data["conditioning"] = dict(self.conditioning)
        if self.input_ids is not None:
            data["input_ids"] = self._tensor_to_payload(self.input_ids)
        if self.speaker_emb is not None:
            data["speaker_emb"] = self._tensor_to_payload(self.speaker_emb)
        if self.speaker_fingerprint is not None:
            data["speaker_fingerprint"] = self.speaker_fingerprint
        if self.audio_codes is not None:
            data["audio_codes"] = self._tensor_to_payload(self.audio_codes)
        if self.eos_frame is not None:
            data["eos_frame"] = int(self.eos_frame)
        if self.prompt_tokens:
            data["prompt_tokens"] = int(self.prompt_tokens)
        if self.completion_tokens:
            data["completion_tokens"] = int(self.completion_tokens)
        if self.engine_time_s:
            data["engine_time_s"] = float(self.engine_time_s)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "Zonos2State":
        if not isinstance(data, dict):
            data = {}
        gen = data.get("generation_kwargs")
        cond = data.get("conditioning")
        positions = data.get("speaker_token_positions")
        return cls(
            text=str(data.get("text", "")),
            ref_audio=data.get("ref_audio"),
            ref_text=data.get("ref_text"),
            language=data.get("language"),
            speaking_rate=(
                float(data["speaking_rate"])
                if data.get("speaking_rate") is not None
                else None
            ),
            conditioning=dict(cond) if isinstance(cond, dict) else {},
            input_ids=data.get("input_ids"),
            speaker_token_positions=(
                list(positions) if isinstance(positions, (list, tuple)) else [0]
            ),
            speaker_emb=data.get("speaker_emb"),
            speaker_fingerprint=data.get("speaker_fingerprint"),
            audio_codes=data.get("audio_codes"),
            eos_frame=(
                int(data["eos_frame"]) if data.get("eos_frame") is not None else None
            ),
            sample_rate=int(
                data.get("sample_rate", ZONOS2_SAMPLE_RATE) or ZONOS2_SAMPLE_RATE
            ),
            generation_kwargs=dict(gen) if isinstance(gen, dict) else {},
            prompt_tokens=int(data.get("prompt_tokens", 0) or 0),
            completion_tokens=int(data.get("completion_tokens", 0) or 0),
            engine_time_s=float(data.get("engine_time_s", 0.0) or 0.0),
        )
