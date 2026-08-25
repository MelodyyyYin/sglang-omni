# SPDX-License-Identifier: Apache-2.0
"""Hand parameter handles from one PD half to the other at startup.

:mod:`sglang_omni.model_runner.weight_sharing` can export and adopt handles,
but the two halves are separate processes with no channel between them at load
time. The KV plane does not supply one: ``prepare_kv_receive`` and
``send_kv_pages`` are per-transfer and run long after both halves are up.

This uses the directory the run already has. ``create_ipc_runtime_dir`` makes
one private directory per pipeline instance before any stage is spawned, every
stage is handed endpoints inside it, and it is removed when the run ends. A
file there is therefore visible to both halves, private to the run, and cleaned
up without new ownership rules.

Publishing is a write to a temporary name followed by ``os.replace``, so a
reader never observes a partial file. Reading returns ``None`` when the peer
has not published rather than waiting for it: ``_construct_scheduler`` builds
each stage inside ``gpu_startup_lock(gpu_id)``, so two halves on one device
load one at a time, and a reader that waited would hold that lock against the
very half it is waiting for. A half that finds nothing publishes its own
handles instead, so whichever loads second is the one that adopts.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SUBDIR = "pd-weights"


class RendezvousUnavailable(RuntimeError):
    """The run's IPC directory could not be derived from an endpoint."""


def rendezvous_dir_from_endpoint(endpoint: str) -> Path:
    """Return the run's directory, given any ``ipc://`` endpoint from this run.

    ``allocate_endpoints`` puts every socket directly in the run directory, so
    the parent of an endpoint path is that directory. Deriving it here keeps
    the halves from needing a new argument threaded through stage startup.
    """
    if not endpoint.startswith("ipc://"):
        raise RendezvousUnavailable(
            f"expected an ipc:// endpoint to locate the run directory, got {endpoint!r}"
        )
    return Path(endpoint[len("ipc://") :]).parent


def publish_parameter_handles(
    handles: dict[str, Any],
    *,
    rendezvous_dir: Path,
    stage_name: str,
) -> Path:
    """Write *handles* where the peer half can read them. Returns the path."""
    directory = Path(rendezvous_dir) / _SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"{stage_name}.pkl"
    staging = directory / f"{stage_name}.pkl.{os.getpid()}"
    staging.write_bytes(pickle.dumps(handles))
    os.replace(staging, final)
    logger.info(
        "published %d parameter handles for %s at %s",
        len(handles),
        stage_name,
        final,
    )
    return final


def read_parameter_handles(
    *,
    rendezvous_dir: Path,
    stage_name: str,
) -> dict[str, Any] | None:
    """Return the handles *stage_name* published, or None if it has not.

    This does not wait. ``_construct_scheduler`` builds a stage inside
    ``gpu_startup_lock(gpu_id)``, so two halves on one device load one at a
    time and the second one to load finds the first one's file already there.
    Waiting here would instead hold that lock against the half being waited
    for, which needs the same lock to load at all.
    """
    path = Path(rendezvous_dir) / _SUBDIR / f"{stage_name}.pkl"
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        logger.info(
            "%s has not published parameter handles; this half publishes its own",
            stage_name,
        )
        return None
    handles = pickle.loads(payload)
    logger.info("adopted %d parameter handles from %s", len(handles), stage_name)
    return handles
