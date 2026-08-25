# SPDX-License-Identifier: Apache-2.0
"""Share one copy of a stage's weights between two processes on one GPU.

A PD-disaggregated stage runs its two halves in separate processes. On one
device that means two copies of the same weights: measured at 57 GiB each for
the Qwen3-Omni thinker, which leaves a 140 GiB card room for about 21,500 KV
tokens per half against 677,613 on a colocated card.

The halves already map each other's GPU memory for the KV plane. Weights are
the easier case: static, read-only, allocated once, never reclaimed until
shutdown, so none of the reserve, commit and abort machinery applies. One half
exports handles to its parameter storage; the other points its own parameters
at that storage and releases what it loaded.

Peak memory is unchanged, because the adopting half still constructs and loads
before it swaps. The KV pools are sized after that, so they see the freed
space.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WeightLayoutMismatch(RuntimeError):
    """The two halves disagree about which parameters exist.

    Raised rather than skipped. A silently unshared parameter costs the memory
    without saying so, and a wrongly shared one is worse.
    """


def export_parameter_handles(model: Any) -> dict[str, Any]:
    """Return one CUDA IPC handle per parameter, keyed by parameter name.

    Handles are small tokens, not copies, so exporting allocates nothing. The
    exporting process must outlive every process that adopts them.
    """
    from torch.multiprocessing.reductions import reduce_tensor

    handles: dict[str, Any] = {}
    for name, param in model.named_parameters():
        if not param.is_cuda:
            continue
        handles[name] = reduce_tensor(param.data)
    logger.info("exported %d parameter handles for weight sharing", len(handles))
    return handles


def adopt_parameter_handles(model: Any, handles: dict[str, Any]) -> int:
    """Point this model's parameters at exported storage; return bytes released.

    Every exported name must exist here and every local name must have been
    exported. A mismatch means the two halves built different models, and
    continuing would either waste the memory silently or read the wrong bytes.

    Calls ``empty_cache`` at the end, which is required rather than tidy.
    Dropping the references returns the blocks to torch's caching allocator,
    where they stay invisible to every other process -- and the other process
    using them is the entire point. Measured on one H200 with 57.17 GiB across
    1757 tensors: after dropping the references the device still reported all
    of it held while ``memory_allocated`` reported zero, so a check on
    ``memory_allocated`` alone would have called that a success.
    """
    import torch

    named = dict(model.named_parameters())
    _check_parameters_match(named, handles)

    released = 0
    for name, handle in handles.items():
        param = named[name]
        rebuild, args = handle
        shared = rebuild(*args)
        released += param.data.numel() * param.data.element_size()
        param.data = shared

    torch.cuda.empty_cache()
    logger.info(
        "adopted %d shared parameters, released %.2f GiB to the device",
        len(handles),
        released / 1024**3,
    )
    return released


def _check_parameters_match(
    named: dict[str, Any],
    handles: dict[str, Any],
) -> None:
    """Fail before mutating anything if the two models disagree."""
    missing = sorted(set(handles) - set(named))
    if missing:
        raise WeightLayoutMismatch(
            f"{len(missing)} exported parameters are absent from this model, "
            f"starting with {missing[:3]}"
        )
    extra = sorted(set(named) - set(handles))
    if extra:
        raise WeightLayoutMismatch(
            f"{len(extra)} of this model's parameters were not exported, "
            f"starting with {extra[:3]}"
        )


@dataclasses.dataclass(frozen=True)
class WeightSharingPlan:
    """What this half does about weights at startup, and with whom.

    Built by :func:`plan_for_pd_halves`, which returns ``None`` unless the two
    halves are on one device: a CUDA IPC handle names memory on a particular
    GPU, so halves on different cards each need their own copy.
    """

    stage_name: str
    peer_stage: str
    rendezvous_dir: Path


def plan_for_pd_halves(
    *,
    stage_name: str,
    peer_stage: str,
    own_gpu: int | list[int] | None,
    peer_gpu: int | list[int] | None,
    rendezvous_dir: Path,
) -> WeightSharingPlan | None:
    """Return the plan for this half, or None when sharing does not apply."""
    if own_gpu is None or peer_gpu != own_gpu:
        return None
    return WeightSharingPlan(
        stage_name=stage_name,
        peer_stage=peer_stage,
        rendezvous_dir=rendezvous_dir,
    )


def apply_weight_sharing(model: Any, plan: WeightSharingPlan) -> int:
    """Adopt the peer's weights if it published, else publish for it.

    Neither half waits. ``_construct_scheduler`` builds a stage inside
    ``gpu_startup_lock(gpu_id)``, so two halves on one device load one at a
    time: the first finds nothing published and publishes, the second finds
    that file and adopts. Assigning the exporting role in advance instead
    would deadlock whenever the assigned adopter won the lock, because it
    would hold the lock while waiting for a half that needs the same lock to
    load at all.

    Call this after the weights are loaded and before the KV pool is sized.
    Peak memory is unchanged either way, because the adopting half still loads
    before it swaps, but the pool is sized after this returns and so sees the
    space the swap released.

    Returns the bytes this half released, which is 0 when it published.
    """
    from sglang_omni.model_runner.weight_rendezvous import (
        publish_parameter_handles,
        read_parameter_handles,
    )

    handles = read_parameter_handles(
        rendezvous_dir=plan.rendezvous_dir, stage_name=plan.peer_stage
    )
    if handles is not None:
        return adopt_parameter_handles(model, handles)

    publish_parameter_handles(
        export_parameter_handles(model),
        rendezvous_dir=plan.rendezvous_dir,
        stage_name=plan.stage_name,
    )
    return 0
