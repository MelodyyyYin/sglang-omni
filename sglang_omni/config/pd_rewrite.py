# SPDX-License-Identifier: Apache-2.0
"""Generic PD (prefill/decode) disaggregation graph rewrite.

Note: (Yue Yin) This expands each logical stage marked ``pd_disaggregation``
into two physical stages, ``<stage>_prefill`` and ``<stage>_decode``. It is
purely structural — no model conditionals, no scheduler/KV/request wiring:

* inbound edges (other stages' ``next``/``stream_to``/``wait_for``/
  ``project_payload`` referencing the logical stage) are rewritten to
  terminate at the prefill half;
* the original downstream edges (``next``/``route_fn``/``terminal``/
  ``stream_to``) originate from the decode half;
* fan-in (``wait_for``/``merge_fn``) stays on the prefill half, since inputs
  arrive there; the decode half admits a prebuilt request from the PD handoff
  and therefore carries no fan-in;
* the prefill -> decode handoff is an explicit PD control/KV relationship
  recorded in a typed ``pd_execution`` field (``role``/``partner``), never an
  ordinary ``next`` payload edge. Treating decode as a normal ``next`` target
  would make the coordinator admit it as a fresh downstream request and
  double-transport the payload alongside the KV plane.
"""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.config.schema import PDExecution, StageConfig

PREFILL_SUFFIX = "_prefill"
DECODE_SUFFIX = "_decode"


@dataclass(frozen=True)
class PDExpansion:
    """Result of the PD graph rewrite.

    Note: (Yue Yin) ``routing_map`` and ``terminal_map`` separate the two
    logical->physical identities a PD stage now has. New requests and dynamic
    routes that name logical stage ``S`` must reach the prefill half
    (``routing_map``), while request completion for a terminal ``S`` is owned by
    the decode half (``terminal_map``). Callers compose these maps with the
    fusion name map so downstream routing and completion accounting use the
    correct physical identity.
    """

    stages: list[StageConfig]
    entry_stage: str
    routing_map: dict[str, str]
    terminal_map: dict[str, str]


def expand_pd_stages(
    stages: list[StageConfig],
    *,
    entry_stage: str,
) -> PDExpansion:
    """Expand PD-disaggregation stages into prefill/decode physical stages.

    Stages without ``pd_disaggregation`` are preserved in order, with inbound
    references to any PD stage rewritten to that stage's prefill half. Calling
    this again on an already-expanded stage list is a safe no-op because the
    generated halves clear ``pd_disaggregation`` (idempotent).
    """
    pd_names = {s.name for s in stages if s.pd_disaggregation is not None}
    if not pd_names:
        return PDExpansion(
            stages=list(stages),
            entry_stage=entry_stage,
            routing_map={},
            terminal_map={},
        )

    # Note: (Yue Yin) External references to a PD stage terminate at its
    # prefill half; downstream continuation is owned by the decode half.
    inbound_rename = {name: f"{name}{PREFILL_SUFFIX}" for name in pd_names}
    # Note: (Yue Yin) Completion for a logical terminal PD stage is owned by
    # the decode half, so terminal identity maps to <name>_decode.
    terminal_rename = {name: f"{name}{DECODE_SUFFIX}" for name in pd_names}

    out: list[StageConfig] = []
    for s in stages:
        if s.pd_disaggregation is None:
            out.append(_rewrite_inbound_refs(s, inbound_rename))
            continue
        prefill, decode = _split_pd_stage(s, inbound_rename)
        out.append(prefill)
        out.append(decode)

    new_entry = inbound_rename.get(entry_stage, entry_stage)
    return PDExpansion(
        stages=out,
        entry_stage=new_entry,
        routing_map=dict(inbound_rename),
        terminal_map=dict(terminal_rename),
    )


def _rename_targets(
    targets: str | list[str] | None,
    rename: dict[str, str],
) -> str | list[str] | None:
    if targets is None:
        return None
    if isinstance(targets, str):
        return rename.get(targets, targets)
    return [rename.get(t, t) for t in targets]


def _rewrite_inbound_refs(
    s: StageConfig,
    rename: dict[str, str],
) -> StageConfig:
    next_ = _rename_targets(s.next, rename)
    stream_to = [rename.get(t, t) for t in s.stream_to]
    wait_for = [rename.get(t, t) for t in s.wait_for] if s.wait_for else s.wait_for
    project_payload = {rename.get(k, k): v for k, v in s.project_payload.items()}

    if (
        next_ == s.next
        and stream_to == list(s.stream_to)
        and wait_for == s.wait_for
        and project_payload == s.project_payload
    ):
        return s

    return s.model_copy(
        deep=True,
        update={
            "next": next_,
            "stream_to": stream_to,
            "wait_for": wait_for,
            "project_payload": project_payload,
        },
    )


def _split_pd_stage(
    s: StageConfig,
    inbound_rename: dict[str, str],
) -> tuple[StageConfig, StageConfig]:
    pd = s.pd_disaggregation
    assert pd is not None
    prefill_name = f"{s.name}{PREFILL_SUFFIX}"
    decode_name = f"{s.name}{DECODE_SUFFIX}"

    # Note: (Yue Yin) Prefill half keeps fan-in inputs; owns no ordinary
    # downstream edge. Its only successor is the PD control/KV relationship
    # (recorded below in the typed pd_execution field).
    prefill = s.model_copy(
        deep=True,
        update={
            "name": prefill_name,
            "gpu": pd.prefill.gpu if pd.prefill.gpu is not None else s.gpu,
            "process": pd.prefill.process or prefill_name,
            "next": None,
            "terminal": False,
            "route_fn": None,
            "stream_to": [],
            "stream_done_to_fn": None,
            "pd_disaggregation": None,
            # Note: (Yue Yin) Typed PD metadata, not factory_args, so it is
            # never forwarded as a factory constructor kwarg.
            "pd_execution": PDExecution(role="prefill", partner=decode_name),
        },
    )

    # Note: (Yue Yin) Decode half owns the logical stage's real downstream
    # edges, streaming, and terminal flag. It admits a prebuilt request from
    # the PD handoff, so it carries no fan-in.
    decode = s.model_copy(
        deep=True,
        update={
            "name": decode_name,
            "gpu": pd.decode.gpu if pd.decode.gpu is not None else s.gpu,
            "process": pd.decode.process or decode_name,
            "next": _rename_targets(s.next, inbound_rename),
            "stream_to": [inbound_rename.get(t, t) for t in s.stream_to],
            "project_payload": {
                inbound_rename.get(k, k): v for k, v in s.project_payload.items()
            },
            "wait_for": None,
            "wait_for_fn": None,
            "merge_fn": None,
            "pd_disaggregation": None,
            # Note: (Yue Yin) Symmetric partner identity with the prefill half.
            "pd_execution": PDExecution(role="decode", partner=prefill_name),
        },
    )

    return prefill, decode
