# SPDX-License-Identifier: Apache-2.0
"""The startup channel that carries parameter handles between PD halves.

The halves are separate processes with no channel at load time. These pin the
three properties the exchange depends on: the run directory is derivable from
an endpoint the stage already has, a reader never sees a partial file, and a
peer that never publishes costs memory rather than the startup.
"""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import pytest

from sglang_omni.model_runner.weight_rendezvous import (
    RendezvousUnavailable,
    await_parameter_handles,
    publish_parameter_handles,
    rendezvous_dir_from_endpoint,
)


def test_the_run_directory_comes_from_an_endpoint_the_stage_already_has() -> None:
    """allocate_endpoints puts every socket directly in the run directory."""
    endpoint = "ipc:///tmp/sglang_omni/qwen3-omni-ab12/stage_thinker_prefill.sock"

    assert rendezvous_dir_from_endpoint(endpoint) == Path(
        "/tmp/sglang_omni/qwen3-omni-ab12"
    )


def test_a_non_ipc_endpoint_is_rejected() -> None:
    """A TCP endpoint has no run directory, and guessing one would misplace it."""
    with pytest.raises(RendezvousUnavailable):
        rendezvous_dir_from_endpoint("tcp://127.0.0.1:5555")


def test_handles_survive_the_round_trip(tmp_path: Path) -> None:
    handles = {"model.layers.0.weight": ("rebuild", (1, 2, 3))}

    publish_parameter_handles(handles, rendezvous_dir=tmp_path, stage_name="prefill")

    assert (
        await_parameter_handles(
            rendezvous_dir=tmp_path, stage_name="prefill", timeout_s=1.0
        )
        == handles
    )


def test_a_peer_that_never_publishes_returns_none(tmp_path: Path) -> None:
    """Failing to share costs memory; failing startup would cost the run."""
    started = time.monotonic()

    result = await_parameter_handles(
        rendezvous_dir=tmp_path, stage_name="prefill", timeout_s=0.2
    )

    assert result is None
    assert time.monotonic() - started >= 0.2


def test_the_reader_waits_for_a_late_publisher(tmp_path: Path) -> None:
    """The halves load concurrently, so the reader normally arrives first."""
    handles = {"w": ("rebuild", ())}
    publisher = multiprocessing.Process(
        target=_publish_after,
        args=(str(tmp_path), 0.15, handles),
    )
    publisher.start()
    try:
        result = await_parameter_handles(
            rendezvous_dir=tmp_path, stage_name="prefill", timeout_s=5.0
        )
    finally:
        publisher.join(timeout=10)

    assert result == handles


def test_a_reader_never_observes_a_partial_file(tmp_path: Path) -> None:
    """Publishing stages under another name, so the final path appears whole."""
    directory = tmp_path / "pd-weights"
    directory.mkdir()
    big = {f"layer.{i}.weight": ("rebuild", tuple(range(64))) for i in range(400)}

    publish_parameter_handles(big, rendezvous_dir=tmp_path, stage_name="prefill")

    leftovers = [p.name for p in directory.iterdir() if p.name != "prefill.pkl"]
    assert leftovers == []
    assert (
        await_parameter_handles(
            rendezvous_dir=tmp_path, stage_name="prefill", timeout_s=1.0
        )
        == big
    )


def _publish_after(directory: str, delay_s: float, handles: dict) -> None:
    time.sleep(delay_s)
    publish_parameter_handles(
        handles, rendezvous_dir=Path(directory), stage_name="prefill"
    )
