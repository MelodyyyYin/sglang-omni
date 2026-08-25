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

import logging
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
