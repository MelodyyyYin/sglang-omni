# SPDX-License-Identifier: Apache-2.0
"""OmniScheduler-backed ZONOS2 AR engine stage (radix cache + batched decode).

Builds the SGLang infrastructure for the custom ZONOS2 backbone and drives it
with Zonos2ModelRunner. The checkpoint ships params.json + a flat model.pth
(no config.json / safetensors), so we synthesize an HF config shim and symlink
model.pth as pytorch_model.bin for the loader; our model.load_weights does the
key mapping.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from sglang_omni.models.zonos2.hf_config import (
    Zonos2Config,
    load_zonos2_pretrained_config,
)


def _build_config_shim(model_path: str, cfg: Zonos2Config) -> str:
    shim = tempfile.mkdtemp(prefix="zonos2_sglang_")
    with open(os.path.join(model_path, "params.json")) as f:
        params = json.load(f)
    params.update(
        architectures=["Zonos2SGLangModel"],
        model_type="zonos2",
        hidden_size=cfg.dim,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.audio_vocab,
        max_position_embeddings=cfg.max_seqlen,
        rms_norm_eps=cfg.norm_eps,
        torch_dtype="bfloat16",
        tie_word_embeddings=False,
    )
    with open(os.path.join(shim, "config.json"), "w") as f:
        json.dump(params, f)
    os.symlink(
        os.path.join(model_path, "model.pth"), os.path.join(shim, "pytorch_model.bin")
    )
    return shim


def _register_zonos2_autoconfig() -> None:
    from transformers import AutoConfig

    try:
        AutoConfig.register("zonos2", Zonos2Config)
    except (ValueError, KeyError):
        pass  # already registered


def create_sglang_omni_tts_engine_executor(
    model_path: str,
    *,
    gpu_id: int | None = 0,
    dtype: str = "bfloat16",
    mem_fraction_static: float = 0.5,
    **_: Any,
) -> Any:
    from sglang_omni.models.zonos2.model_runner import Zonos2ModelRunner
    from sglang_omni.models.zonos2.sglang_request_builders import (
        make_zonos2_scheduler_adapters,
    )
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    cfg = load_zonos2_pretrained_config(model_path)
    _register_zonos2_autoconfig()
    shim = _build_config_shim(model_path, cfg)
    gpu = int(gpu_id) if gpu_id is not None else 0

    server_args = build_sglang_server_args(
        shim,
        context_length=cfg.max_seqlen,
        dtype=dtype,
        disable_cuda_graph=True,  # O3 adds graph capture
        disable_overlap_schedule=True,
        enable_torch_compile=False,
        max_running_requests=16,
        mem_fraction_static=mem_fraction_static,
        sampling_backend="pytorch",
        trust_remote_code=True,
    )

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args, gpu, model_arch_override="Zonos2SGLangModel"
    )

    model = model_worker.model_runner.model
    output_proc = SGLangOutputProcessor(
        capture_hidden=False, capture_hidden_layers=None, model=model
    )
    request_builder, result_adapter = make_zonos2_scheduler_adapters(model=model)

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=Zonos2ModelRunner(model_worker, output_proc),
        request_builder=request_builder,
        result_adapter=result_adapter,
    )
