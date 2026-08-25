# SPDX-License-Identifier: Apache-2.0
"""Sharing one copy of a stage's weights between two PD halves on one GPU."""

from __future__ import annotations

import time
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


def test_the_first_half_to_load_publishes(tmp_path) -> None:
    """Nothing is published yet, so this half exports rather than waiting."""
    from sglang_omni.model_runner.weight_sharing import (
        WeightSharingPlan,
        apply_weight_sharing,
    )

    model = SimpleNamespace(named_parameters=lambda: iter(()))
    plan = WeightSharingPlan(
        stage_name="thinker_prefill",
        peer_stage="thinker_decode",
        rendezvous_dir=tmp_path,
        gpu_id=0,
    )

    assert apply_weight_sharing(model, plan) == 0
    assert (tmp_path / "pd-weights" / "thinker_prefill.pkl").exists()


def test_the_second_half_to_load_adopts(tmp_path) -> None:
    """gpu_startup_lock serializes the two, so the second finds the first's file."""
    from sglang_omni.model_runner.weight_sharing import (
        WeightSharingPlan,
        apply_weight_sharing,
    )

    model = SimpleNamespace(named_parameters=lambda: iter(()))
    apply_weight_sharing(
        model,
        WeightSharingPlan(
            stage_name="thinker_prefill",
            peer_stage="thinker_decode",
            rendezvous_dir=tmp_path,
            gpu_id=0,
        ),
    )
    apply_weight_sharing(
        model,
        WeightSharingPlan(
            stage_name="thinker_decode",
            peer_stage="thinker_prefill",
            rendezvous_dir=tmp_path,
            gpu_id=0,
        ),
    )

    assert not (tmp_path / "pd-weights" / "thinker_decode.pkl").exists()


def test_a_half_on_another_card_keeps_its_own_weights(tmp_path) -> None:
    """Cross-GPU halves each need their own copy, and the device says so."""
    from sglang_omni.model_runner.weight_sharing import (
        WeightSharingPlan,
        apply_weight_sharing,
    )

    model = SimpleNamespace(named_parameters=lambda: iter(()))
    apply_weight_sharing(
        model,
        WeightSharingPlan(
            stage_name="thinker_prefill",
            peer_stage="thinker_decode",
            rendezvous_dir=tmp_path,
            gpu_id=0,
        ),
    )

    released = apply_weight_sharing(
        model,
        WeightSharingPlan(
            stage_name="thinker_decode",
            peer_stage="thinker_prefill",
            rendezvous_dir=tmp_path,
            gpu_id=1,
        ),
    )

    assert released == 0
    assert (tmp_path / "pd-weights" / "thinker_decode.pkl").exists()


def test_neither_half_blocks_on_the_other(tmp_path) -> None:
    """A half that waited would hold the GPU startup lock against its peer."""
    from sglang_omni.model_runner.weight_sharing import (
        WeightSharingPlan,
        apply_weight_sharing,
    )

    plan = WeightSharingPlan(
        stage_name="thinker_decode",
        peer_stage="thinker_prefill",
        rendezvous_dir=tmp_path,
        gpu_id=0,
    )
    started = time.monotonic()

    apply_weight_sharing(SimpleNamespace(named_parameters=lambda: iter(())), plan)

    assert time.monotonic() - started < 1.0
