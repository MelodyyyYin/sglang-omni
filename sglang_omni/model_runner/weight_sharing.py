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

    Sharing applies only when the two halves are on one device, and that is
    settled by the published handles rather than by this plan: a CUDA IPC
    handle names memory on a particular GPU, the publisher records which, and
    :func:`apply_weight_sharing` declines handles from another one.
    """

    stage_name: str
    peer_stage: str
    rendezvous_dir: Path
    gpu_id: int
    publishes: bool = True
    adopted: dict[str, Any] | None = None


def apply_weight_sharing(model: Any, plan: WeightSharingPlan) -> int:
    """Publish this half's weights, or adopt the peer's. Returns bytes released.

    Which half publishes is decided from the declared shares, not from load
    order: the publisher keeps the copy it loaded, so its budget must hold the
    weights as well as its KV, and letting a race pick that half makes the same
    placement start one time and fail the next.

    The adopter's wait happens before ``gpu_startup_lock`` is taken, so by the
    time this runs the handles are already in hand.

    Call this after the weights are loaded and before the KV pool is sized.
    Peak memory is unchanged either way, because the adopting half still loads
    before it swaps, but the pool is sized after this returns and so sees the
    space the swap released.
    """
    from sglang_omni.model_runner.weight_rendezvous import publish_parameter_handles

    if plan.publishes:
        handles = export_parameter_handles(model)
        publish_parameter_handles(
            handles,
            rendezvous_dir=plan.rendezvous_dir,
            stage_name=plan.stage_name,
            gpu_id=plan.gpu_id,
            weight_bytes=_parameter_bytes(model),
        )
        return 0

    if plan.adopted is None:
        return 0
    return adopt_parameter_handles(model, plan.adopted)


def _parameter_bytes(model: Any) -> int:
    """Total bytes of this model's CUDA parameters."""
    total = 0
    for _name, param in model.named_parameters():
        if getattr(param, "is_cuda", False):
            total += param.data.numel() * param.data.element_size()
    return total
