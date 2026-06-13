# SPDX-License-Identifier: Apache-2.0
"""Map an OpenAI /v1/audio/speech request onto a Zonos2State.

Reuses the MOSS input/reference parsers so ``--ref-format references`` payloads
work unchanged.
"""

from __future__ import annotations

import base64
import re
from typing import Any

from sglang_omni.models.moss_tts.request_builders import (
    _resolve_optional_text,
    normalize_moss_tts_inputs,
    resolve_moss_reference,
)
from sglang_omni.models.zonos2.payload_types import Zonos2State
from sglang_omni.proto import StagePayload

_DATA_URI_RE = re.compile(r"^data:[^;,]*;base64,(?P<data>.+)$", re.DOTALL)

_SAMPLING_FIELDS = (
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "repetition_penalty",
    "seed",
)


def ref_audio_to_encoder_input(ref_audio: Any) -> Any:
    """Decode a base64 data-URI reference to raw bytes; pass paths/arrays through."""
    if isinstance(ref_audio, str):
        m = _DATA_URI_RE.match(ref_audio)
        if m is not None:
            return base64.b64decode(m.group("data"))
    return ref_audio


def build_zonos2_state(payload: StagePayload) -> Zonos2State:
    inputs = payload.request.inputs or {}
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params")
    if not isinstance(tts_params, dict):
        tts_params = {}

    text, references = normalize_moss_tts_inputs(inputs)
    ref_audio, ref_text = resolve_moss_reference(references, tts_params)
    language = _resolve_optional_text(
        tts_params.get("language") or params.get("language")
    )

    explicit = tts_params.get("explicit_generation_params")
    explicit = (
        {str(f) for f in explicit}
        if isinstance(explicit, (list, tuple, set))
        else set()
    )

    gen: dict[str, Any] = {}
    raw_max = params.get("max_new_tokens")
    if raw_max is not None and not isinstance(raw_max, bool):
        gen["max_tokens"] = int(raw_max)
    for field in _SAMPLING_FIELDS:
        for source in (tts_params, params):
            val = source.get(field)
            if val is None:
                continue
            # tts_params always wins; params only honored when explicitly whitelisted
            if field in explicit or source is tts_params:
                gen[field] = int(val) if field in ("top_k", "seed") else float(val)
            break

    return Zonos2State(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        language=language,
        generation_kwargs=gen,
    )
