# SPDX-License-Identifier: Apache-2.0
"""Static per-model capability vocabulary (RFC #661, Template 6).

Each model package exports a ``CAPABILITIES`` constant declaring which shared
optimizations it supports. This is a *static* declaration ("this model supports
this optimization under its default config"), not a runtime state ("this
optimization is enabled right now"). Runtime gating lives in the model runner.

The registry logs the full table at startup; a unit test asserts every
in-scope model declares one, so the table cannot silently drift.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, fields

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageCapabilities:
    """Static per-model capability annotation.

    All fields default to False. Override to True only for capabilities that
    are verified and passing accuracy + perf CI.
    """

    supports_cuda_graph: bool = False
    supports_async_decode: bool = False
    supports_torch_compile: bool = False
    supports_streaming_vocoder: bool = False
    supports_reference_audio: bool = False


def collect_capabilities(
    package_name: str = "sglang_omni.models",
) -> dict[str, StageCapabilities]:
    """Map model package short-name -> its declared ``CAPABILITIES``.

    Enumerates the model subpackages and reads ``CAPABILITIES`` from each.
    Packages that do not declare one (or fail to import) are skipped.
    """
    package = importlib.import_module(package_name)
    table: dict[str, StageCapabilities] = {}
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            continue
        short = name.rsplit(".", 1)[-1]
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            logger.debug("Skipping %s while collecting capabilities: %s", name, exc)
            continue
        caps = getattr(module, "CAPABILITIES", None)
        if isinstance(caps, StageCapabilities):
            table[short] = caps
    return table


def format_capability_table(table: dict[str, StageCapabilities]) -> str:
    flds = fields(StageCapabilities)
    widths = [max(len(f.name) - len("supports_"), 3) + 2 for f in flds]
    header = f"{'model':<18}" + "".join(
        f"{f.name.replace('supports_', ''):<{w}}" for f, w in zip(flds, widths)
    )
    rows = [header]
    for model, caps in sorted(table.items()):
        cells = "".join(
            f"{('yes' if getattr(caps, f.name) else '-'):<{w}}"
            for f, w in zip(flds, widths)
        )
        rows.append(f"{model:<18}{cells}")
    return "\n".join(rows)


def log_capability_table() -> None:
    table = collect_capabilities()
    if table:
        logger.info("Stage capability table:\n%s", format_capability_table(table))
