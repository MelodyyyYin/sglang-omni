# SPDX-License-Identifier: Apache-2.0
"""Explicit prefill/decode placement overrides for pipeline stages."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sglang_omni.config.schema import (
    PDConfig,
    PDStagePlacement,
    PipelineConfig,
    StageConfig,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PDStageAssignment:
    """One parsed ``--pd-stage`` value."""

    stage_name: str
    prefill_gpu: int | list[int]
    decode_gpu: int | list[int]
    prefill_fraction: float | None
    decode_fraction: float | None


def parse_pd_stage_assignment(value: str) -> _PDStageAssignment:
    """Parse one ``STAGE=PREFILL_GPUS:DECODE_GPUS`` placement assignment.

    Both halves accept the same comma form the per-stage GPU flags use, so a
    TP=2 split is ``thinker=0,1:2,3``.

    A half may carry its share of the card after ``@``, as in
    ``thinker=0@0.25:0@0.65``. That is required when the halves share a GPU,
    because they are then two process groups sizing pools from one device and
    each has to declare what it claims. Without it the flag can express the
    placement but not the budget, and the run fails later in topology
    validation with no way to fix it from the command line.
    """
    stage_name, separator, placement = value.partition("=")
    stage_name = stage_name.strip()
    placement = placement.strip()

    if not separator or not stage_name or not placement:
        raise ValueError(
            f"Invalid PD stage assignment {value!r}; "
            "expected STAGE=PREFILL_GPUS:DECODE_GPUS"
        )

    prefill_spec, colon, decode_spec = placement.partition(":")
    if not colon or not prefill_spec.strip() or not decode_spec.strip():
        raise ValueError(
            f"Invalid PD stage assignment {value!r}; "
            "expected STAGE=PREFILL_GPUS:DECODE_GPUS"
        )

    prefill_gpu, prefill_fraction = _split_share(value, "prefill", prefill_spec)
    decode_gpu, decode_fraction = _split_share(value, "decode", decode_spec)
    return _PDStageAssignment(
        stage_name=stage_name,
        prefill_gpu=prefill_gpu,
        decode_gpu=decode_gpu,
        prefill_fraction=prefill_fraction,
        decode_fraction=decode_fraction,
    )


def _split_share(
    value: str,
    role: str,
    spec: str,
) -> tuple[int | list[int], float | None]:
    """Split ``GPUS`` or ``GPUS@SHARE`` into the two parts."""
    gpu_spec, at_sign, share_spec = spec.partition("@")
    gpu = _parse_gpu_spec(value, role, gpu_spec)
    if not at_sign:
        return gpu, None
    share_spec = share_spec.strip()
    try:
        share = float(share_spec)
    except ValueError:
        raise ValueError(
            f"Invalid PD stage assignment {value!r}; {role} share "
            f"{share_spec!r} is not a number"
        ) from None
    if not 0.0 < share <= 1.0:
        raise ValueError(
            f"Invalid PD stage assignment {value!r}; {role} share {share} "
            "must be in (0, 1]"
        )
    return gpu, share


def apply_pd_stage_overrides(
    pipeline_config: PipelineConfig,
    *,
    pd_stages: list[str] | None = None,
) -> PipelineConfig:
    """Return a config with the requested stages split into prefill/decode halves.

    The compiler already knows how to expand a PD-marked stage; this only gives
    that capability a user-facing address. Without it ``pd_disaggregation`` is
    reachable only by constructing ``StageConfig`` in Python, so a deployment
    cannot turn PD on at all.
    """
    if not pd_stages:
        return pipeline_config

    config = pipeline_config.model_copy(deep=True)
    stages = {stage.name: stage for stage in config.stages}
    role_map = type(config).isolation_role_to_stage()

    requested: dict[str, _PDStageAssignment] = {}
    for assignment in pd_stages:
        parsed = parse_pd_stage_assignment(assignment)
        stage = _resolve_stage(stages, role_map, parsed.stage_name)
        if stage.name in requested:
            raise ValueError(
                f"Stage {stage.name!r} has multiple --pd-stage assignments"
            )
        requested[stage.name] = parsed

    for stage_name, parsed in requested.items():
        prefill_gpu = parsed.prefill_gpu
        decode_gpu = parsed.decode_gpu
        stage = stages[stage_name]
        if stage.pd_disaggregation is not None:
            raise ValueError(
                f"Stage {stage_name!r} already declares pd_disaggregation; "
                "--pd-stage would override the pipeline's own placement"
            )
        stage.pd_disaggregation = PDConfig(
            prefill=PDStagePlacement(
                gpu=prefill_gpu, memory_fraction=parsed.prefill_fraction
            ),
            decode=PDStagePlacement(
                gpu=decode_gpu, memory_fraction=parsed.decode_fraction
            ),
        )
        logger.info(
            "PD placement: stage=%s prefill_gpu=%s decode_gpu=%s",
            stage_name,
            prefill_gpu,
            decode_gpu,
        )

    # Re-run the schema's own PD checks: model_copy does not re-enter
    # model_post_init, so the placement we just wrote is unvalidated until here.
    config._validate_pd()
    return config


def _parse_gpu_spec(assignment: str, role: str, spec: str) -> int | list[int]:
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if not parts:
        raise ValueError(
            f"Invalid PD stage assignment {assignment!r}; {role} GPUs are empty"
        )
    try:
        gpu_ids = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(
            f"Invalid PD stage assignment {assignment!r}; "
            f"{role} GPUs must be integers"
        ) from exc
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(
            f"Invalid PD stage assignment {assignment!r}; "
            f"{role} GPUs must be non-negative"
        )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(
            f"Invalid PD stage assignment {assignment!r}; "
            f"{role} GPUs repeat a device"
        )
    return gpu_ids[0] if len(gpu_ids) == 1 else gpu_ids


def _resolve_stage(
    stages: dict[str, StageConfig],
    role_map: dict[str, str],
    requested_name: str,
) -> StageConfig:
    stage_name = role_map.get(requested_name, requested_name)
    stage = stages.get(stage_name)
    if stage is None:
        known = ", ".join(sorted(stages))
        raise ValueError(
            f"--pd-stage references unknown stage {requested_name!r}; "
            f"known stages: {known}"
        )
    return stage
