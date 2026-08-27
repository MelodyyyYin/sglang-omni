# SPDX-License-Identifier: Apache-2.0
"""Both PD halves may share one GPU, each declaring its share of the card."""

from __future__ import annotations

import pytest

from sglang_omni.config import expand_pd_stages
from sglang_omni.config.schema import PDConfig, PDStagePlacement, PipelineConfig
from tests.unit_test.pipeline.helpers import stage


def _halves(prefill: PDStagePlacement, decode: PDStagePlacement) -> dict:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            stage(
                "thinker",
                terminal=True,
                pd_disaggregation=PDConfig(prefill=prefill, decode=decode),
            )
        ],
    )
    expansion = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )
    return {s.name: s for s in expansion.stages}


def _fraction(stage_config) -> float | None:
    return stage_config.gpu_memory_fraction


def test_one_gpu_holds_both_halves() -> None:
    halves = _halves(
        PDStagePlacement(gpu=0, memory_fraction=0.25),
        PDStagePlacement(gpu=0, memory_fraction=0.65),
    )

    assert halves["thinker_prefill"].gpu == 0
    assert halves["thinker_decode"].gpu == 0


def test_the_halves_stay_in_separate_processes_on_one_gpu() -> None:
    """Sharing a device is not sharing a process; PD needs the second split."""
    halves = _halves(
        PDStagePlacement(gpu=0, memory_fraction=0.25),
        PDStagePlacement(gpu=0, memory_fraction=0.65),
    )

    assert halves["thinker_prefill"].process != halves["thinker_decode"].process


def test_each_half_carries_its_own_share_of_the_card() -> None:
    """The shares reach the existing per-stage budget, which caps the sum."""
    halves = _halves(
        PDStagePlacement(gpu=0, memory_fraction=0.25),
        PDStagePlacement(gpu=0, memory_fraction=0.65),
    )

    assert _fraction(halves["thinker_prefill"]) == 0.25
    assert _fraction(halves["thinker_decode"]) == 0.65


def test_a_half_without_a_share_keeps_the_stage_budget() -> None:
    halves = _halves(PDStagePlacement(gpu=0), PDStagePlacement(gpu=1))

    assert _fraction(halves["thinker_prefill"]) is None
    assert _fraction(halves["thinker_decode"]) is None


def test_separate_gpus_still_expand() -> None:
    halves = _halves(PDStagePlacement(gpu=0), PDStagePlacement(gpu=1))

    assert halves["thinker_prefill"].gpu == 0
    assert halves["thinker_decode"].gpu == 1


def test_a_share_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError):
        PDStagePlacement(gpu=0, memory_fraction=1.5)


def test_a_gpu_list_must_still_match_tp_size() -> None:
    with pytest.raises(ValueError, match="entries but tp_size"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                stage(
                    "thinker",
                    terminal=True,
                    tp_size=2,
                    pd_disaggregation=PDConfig(
                        prefill=PDStagePlacement(gpu=[0]),
                        decode=PDStagePlacement(gpu=[1, 2]),
                    ),
                )
            ],
        )
