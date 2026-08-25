# SPDX-License-Identifier: Apache-2.0
"""Sharing one copy of a stage's weights between two PD halves on one GPU."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.model_runner.weight_sharing import (
    WeightLayoutMismatch,
    _check_parameters_match,
)


def _model(*names: str) -> dict:
    return {name: SimpleNamespace(shape=(4, 4)) for name in names}


def test_matching_names_pass() -> None:
    named = _model("layer.0.weight", "layer.1.weight")

    _check_parameters_match(named, dict.fromkeys(named))


def test_an_exported_parameter_this_model_lacks_is_refused() -> None:
    """The two halves built different models; adopting would read the wrong bytes."""
    named = _model("layer.0.weight")
    handles = dict.fromkeys(_model("layer.0.weight", "layer.1.weight"))

    with pytest.raises(WeightLayoutMismatch, match="absent from this model"):
        _check_parameters_match(named, handles)


def test_a_parameter_that_was_not_exported_is_refused() -> None:
    """Skipping it silently would cost the memory without saying so."""
    named = _model("layer.0.weight", "layer.1.weight")
    handles = dict.fromkeys(_model("layer.0.weight"))

    with pytest.raises(WeightLayoutMismatch, match="were not exported"):
        _check_parameters_match(named, handles)


def test_the_message_names_the_offending_parameters() -> None:
    """A count alone does not tell the reader where the models diverged."""
    named = _model("a", "b")
    handles = dict.fromkeys(_model("a", "b", "c", "d", "e", "f"))

    with pytest.raises(WeightLayoutMismatch) as excinfo:
        _check_parameters_match(named, handles)

    assert "'c'" in str(excinfo.value)


def test_nothing_is_mutated_before_the_check_passes() -> None:
    """The check runs first so a mismatch leaves the model as it was."""
    named = _model("layer.0.weight")
    before = dict(named)

    with pytest.raises(WeightLayoutMismatch):
        _check_parameters_match(named, dict.fromkeys(_model("other.weight")))

    assert named == before


def test_two_halves_on_one_gpu_get_a_plan() -> None:
    from sglang_omni.model_runner.weight_sharing import plan_for_pd_halves

    plan = plan_for_pd_halves(
        stage_name="thinker_prefill",
        peer_stage="thinker_decode",
        role="prefill",
        own_gpu=0,
        peer_gpu=0,
        rendezvous_dir=Path("/run/x"),
    )

    assert plan is not None
    assert plan.exports is True


def test_halves_on_different_gpus_get_no_plan() -> None:
    """A CUDA IPC handle names memory on one device; two cards need two copies."""
    from sglang_omni.model_runner.weight_sharing import plan_for_pd_halves

    assert (
        plan_for_pd_halves(
            stage_name="thinker_prefill",
            peer_stage="thinker_decode",
            role="prefill",
            own_gpu=0,
            peer_gpu=1,
            rendezvous_dir=Path("/run/x"),
        )
        is None
    )


def test_the_decode_half_adopts_rather_than_exports() -> None:
    """The exporter has to outlive the adopter, so which one exports is fixed."""
    from sglang_omni.model_runner.weight_sharing import plan_for_pd_halves

    plan = plan_for_pd_halves(
        stage_name="thinker_decode",
        peer_stage="thinker_prefill",
        role="decode",
        own_gpu=0,
        peer_gpu=0,
        rendezvous_dir=Path("/run/x"),
    )

    assert plan.exports is False
    assert plan.peer_stage == "thinker_prefill"


def test_an_adopter_whose_peer_never_publishes_keeps_its_weights(tmp_path) -> None:
    """Giving up costs memory; raising here would cost the startup."""
    from sglang_omni.model_runner.weight_sharing import (
        WeightSharingPlan,
        apply_weight_sharing,
    )

    plan = WeightSharingPlan(
        stage_name="thinker_decode",
        peer_stage="thinker_prefill",
        exports=False,
        rendezvous_dir=tmp_path,
        timeout_s=0.1,
    )

    assert apply_weight_sharing(object(), plan) == 0
