# SPDX-License-Identifier: Apache-2.0
"""Generic PD (prefill/decode) capability declaration and validation.

Note: (Yue Yin) A stage may only be PD-disaggregated if its factory declares
that it can run as a prefill/decode half. This capability is expressed by a
marker attribute the factory author sets via :func:`pd_disaggregation_capable`,
never by a model-name or factory-name conditional. The compiler validates the
marker before process launch so a mis-configured pipeline fails fast in the
parent process instead of crashing a spawned worker.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from sglang_omni.config.schema import StageConfig
from sglang_omni.utils.imports import import_string

# Note: (Yue Yin) Marker attribute stamped on a factory callable. Kept as a
# private dunder so it cannot collide with ordinary factory kwargs/attributes.
_PD_CAPABLE_ATTR = "__sglang_pd_disaggregation_capable__"


def pd_disaggregation_capable(factory: Callable[..., Any]) -> Callable[..., Any]:
    """Mark *factory* as able to run as a PD prefill/decode half.

    Note: (Yue Yin) Model authors decorate their stage factory with this to opt
    into PD disaggregation. It is the single generic capability signal; no code
    branches on model or factory names.
    """
    setattr(factory, _PD_CAPABLE_ATTR, True)
    return factory


def factory_supports_pd(factory: Callable[..., Any]) -> bool:
    """Return whether *factory* declared PD-disaggregation capability."""
    return bool(getattr(factory, _PD_CAPABLE_ATTR, False))


def validate_pd_capabilities(stages: Iterable[StageConfig]) -> None:
    """Reject PD-enabled stages whose factory is not PD-capable.

    Note: (Yue Yin) Runs in the parent process before workers are spawned.
    Only factories of stages that actually carry PD execution metadata are
    imported, so ordinary non-PD pipelines never import a factory here.
    """
    for stage in stages:
        if stage.pd_execution is None:
            continue
        factory = import_string(stage.factory)
        if not factory_supports_pd(factory):
            raise ValueError(
                f"Stage {stage.name!r} is PD-disaggregated (role="
                f"{stage.pd_execution.role!r}) but its factory {stage.factory!r} "
                "is not PD-capable; decorate the factory with "
                "@pd_disaggregation_capable to opt in"
            )
