# SPDX-License-Identifier: Apache-2.0
"""Sharing one copy of a stage's weights between two PD halves on one GPU."""

from __future__ import annotations

import time
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


def test_the_publishing_half_publishes(tmp_path) -> None:
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
        publishes=True,
    )

    assert apply_weight_sharing(model, plan) == 0
    assert (tmp_path / "pd-weights" / "thinker_prefill.pkl").exists()


def test_the_adopting_half_publishes_nothing(tmp_path) -> None:
    """Two publishers would leave each half holding its own copy."""
    from sglang_omni.model_runner.weight_sharing import (
        WeightSharingPlan,
        apply_weight_sharing,
    )

    plan = WeightSharingPlan(
        stage_name="thinker_decode",
        peer_stage="thinker_prefill",
        rendezvous_dir=tmp_path,
        gpu_id=0,
        publishes=False,
    )

    apply_weight_sharing(SimpleNamespace(named_parameters=lambda: iter(())), plan)

    assert not (tmp_path / "pd-weights" / "thinker_decode.pkl").exists()


def test_an_adopter_that_got_nothing_keeps_its_weights(tmp_path) -> None:
    """Giving up costs memory; failing the stage would cost the run."""
    from sglang_omni.model_runner.weight_sharing import (
        WeightSharingPlan,
        apply_weight_sharing,
    )

    plan = WeightSharingPlan(
        stage_name="thinker_decode",
        peer_stage="thinker_prefill",
        rendezvous_dir=tmp_path,
        gpu_id=0,
        publishes=False,
        adopted=None,
    )

    assert apply_weight_sharing(SimpleNamespace(), plan) == 0


def test_the_larger_share_publishes() -> None:
    """The publisher keeps its copy, so its budget must hold the weights."""
    from sglang_omni.config import expand_pd_stages
    from sglang_omni.config.schema import PDConfig, PDStagePlacement
    from tests.unit_test.pipeline.helpers import stage

    if "memory_fraction" not in PDStagePlacement.model_fields:
        pytest.skip("per-half shares arrive with the placement surface")

    stages = [
        stage(
            "thinker",
            terminal=True,
            pd_disaggregation=PDConfig(
                prefill=PDStagePlacement(gpu=0, memory_fraction=0.30),
                decode=PDStagePlacement(gpu=0, memory_fraction=0.62),
            ),
        )
    ]

    halves = {s.name: s for s in expand_pd_stages(stages, entry_stage="thinker").stages}

    assert halves["thinker_decode"].pd_execution.publishes_weights is True
    assert halves["thinker_prefill"].pd_execution.publishes_weights is False


def test_exactly_one_half_publishes_on_equal_shares() -> None:
    """A tie must still be decided, or both halves keep their own copy."""
    from sglang_omni.config import expand_pd_stages
    from sglang_omni.config.schema import PDConfig, PDStagePlacement
    from tests.unit_test.pipeline.helpers import stage

    if "memory_fraction" in PDStagePlacement.model_fields:
        pd = PDConfig(
            prefill=PDStagePlacement(gpu=0, memory_fraction=0.47),
            decode=PDStagePlacement(gpu=0, memory_fraction=0.47),
        )
    else:
        pd = PDConfig(
            prefill=PDStagePlacement(gpu=0), decode=PDStagePlacement(gpu=0)
        )
    stages = [stage("thinker", terminal=True, pd_disaggregation=pd)]

    halves = {s.name: s for s in expand_pd_stages(stages, entry_stage="thinker").stages}
    roles = [halves[n].pd_execution.publishes_weights for n in halves]

    assert sum(bool(r) for r in roles) == 1
