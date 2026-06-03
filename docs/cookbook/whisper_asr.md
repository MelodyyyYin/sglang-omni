# Whisper ASR

Whisper ASR checkpoints can be started through the OpenAI-compatible `/v1/audio/transcriptions` endpoint, but this path is experimental in the current SGLang-Omni tree. Prefer [Qwen3-ASR](qwen3_asr.md) for validated ASR serving.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download a Whisper checkpoint:

```bash
hf download openai/whisper-large-v3
```

## Server Configuration

Whisper ASR runs a single ASR stage on one GPU.

```bash
sgl-omni serve \
  --model-path openai/whisper-large-v3 \
  --port 8000
```

## Transcribe Audio

Use a minimal multipart request to verify that the server accepts audio and returns the transcription response schema:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=openai/whisper-large-v3 \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "openai/whisper-large-v3",
            "response_format": "json",
        },
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
print(resp.json()["text"])
```

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | `english` | Language hint; `en`, `eng`, and `english` normalize to `english` |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `temperature` | float | `0.0` | Sampling temperature |

The request builder also supports `task` (`transcribe` by default) and `max_new_tokens`, but the public transcription endpoint currently exposes only the fields above. The route uses the ASR stage default unless the pipeline is configured another way.

## Known Limitations

- This path is not yet correctness-validated. On the current test environment, `openai/whisper-large-v3` starts on a clean H200 and returns the OpenAI-compatible response schema, but short local smoke-test clips returned an empty `text` field.
- First startup can spend several minutes in torch compile / CUDA graph capture.
- The endpoint accepts one uploaded file per request.
- Audio is resampled to 16 kHz before transcription.
- `prompt` is accepted by the HTTP endpoint for OpenAI compatibility, but Whisper ASR currently does not pass it into decoding.
