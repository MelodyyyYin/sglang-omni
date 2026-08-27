# SPDX-License-Identifier: Apache-2.0
"""Generic PD (prefill/decode) capability declaration and validation.

A factory must explicitly opt in via :func:`pd_disaggregation_capable`; the
compiler validates the marker in the parent process before spawning workers so a
mis-configured pipeline fails fast.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from sglang_omni.config.pd_rewrite import PD_REQUIRED_SERVER_ARGS
from sglang_omni.config.schema import EngineArgs, StageConfig
from sglang_omni.utils.imports import import_string

# Note (Yue Yin): A private dunder avoids collisions with factory-owned metadata.
_PD_CAPABLE_ATTR = "__sglang_pd_disaggregation_capable__"


def pd_disaggregation_capable(factory: Callable[..., Any]) -> Callable[..., Any]:
    """Mark *factory* as able to run as a PD prefill/decode half."""
    setattr(factory, _PD_CAPABLE_ATTR, True)
    return factory


def factory_supports_pd(factory: Callable[..., Any]) -> bool:
    """Return whether *factory* declared PD-disaggregation capability."""
    return bool(getattr(factory, _PD_CAPABLE_ATTR, False))


def validate_pd_capabilities(stages: Iterable[StageConfig]) -> None:
    """Reject PD-enabled stages whose factory is not PD-capable.

    Runs in the parent process before workers are spawned. Only PD-expanded
    stages are checked, so ordinary non-PD pipelines do not import factories
    here.
    """
    for stage in stages:
        if stage.pd_execution is None:
            continue
        factory = import_string(stage.factory_path)
        if not factory_supports_pd(factory):
            raise ValueError(
                f"Stage {stage.name!r} is PD-disaggregated (role="
                f"{stage.pd_execution.role!r}) but its factory "
                f"{stage.factory_path!r} "
                "is not PD-capable; decorate the factory with "
                "@pd_disaggregation_capable to opt in"
            )


def apply_pd_required_server_args(stages: Iterable[StageConfig]) -> list[StageConfig]:
    """Set the engine args PD needs on each generated half.

    ``bind_pd_runtime`` refuses anything else, so supplying them here turns a
    configuration that would fail at bind time into one that runs.

    They go in the stage's ``engine`` block, which is where SGLang ServerArgs
    live. That block only exists on stage types that declare ``engine_stage``,
    so a PD half of another kind is left alone rather than given a block it
    may not carry. ``EngineArgs`` allows free-form keys and reports them from
    ``overrides()``, so neither key needs declaring.

    A value that contradicts one of them is rejected rather than overwritten,
    because the request cannot be honoured and failing at bind time would
    report it as a runtime error instead of a configuration one.
    """
    out: list[StageConfig] = []
    for stage in stages:
        if stage.pd_execution is None or not type(stage).engine_stage:
            out.append(stage)
            continue
        engine = stage.engine or EngineArgs()
        current = engine.overrides()
        updates: dict[str, Any] = {}
        for key, required in PD_REQUIRED_SERVER_ARGS.items():
            existing = current.get(key)
            if existing is not None and existing != required:
                raise ValueError(
                    f"Stage {stage.name!r} is PD-disaggregated, which requires "
                    f"engine.{key}={required!r}, but the configuration sets "
                    f"engine.{key}={existing!r}"
                )
            updates[key] = required
        out.append(
            stage.model_copy(update={"engine": engine.model_copy(update=updates)})
        )
    return out
