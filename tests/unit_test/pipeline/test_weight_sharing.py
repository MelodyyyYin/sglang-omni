# SPDX-License-Identifier: Apache-2.0
"""Adopting shared weights refuses to proceed on a mismatch."""

from __future__ import annotations

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
