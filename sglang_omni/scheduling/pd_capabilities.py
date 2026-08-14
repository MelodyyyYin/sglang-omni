# SPDX-License-Identifier: Apache-2.0
"""Capability policy for Prefill-Decode request handoff.

PR 2 keeps the policy generic and conservative.  Later model-specific work
(PR 3) can opt-in features by expanding the allow-lists here.
"""

from __future__ import annotations

import dataclasses
from typing import Any


class PDCapabilityError(ValueError):
    """Raised when a continuation cannot be handled by the current PD policy."""


@dataclasses.dataclass(frozen=True)
class PDCapabilityPolicy:
    """Decides whether a `DecodeContinuation` is supported in this configuration.

    The defaults reject every feature that requires model-specific or
    scheduler-specific integration beyond the generic PR 2 contract.
    """

    allow_projected_input_embeds: bool = False
    allow_speculative_decoding: bool = False
    allow_grammar: bool = False
    allow_custom_logit_processor: bool = False
    allow_cross_node: bool = False
    require_equal_tp_size: bool = True
    allowed_multimodal_schemas: frozenset[str] = frozenset()

    def validate_continuation(
        self,
        continuation: "DecodeContinuation",
        *,
        source_tp_size: int,
        target_tp_size: int,
        is_local: bool = True,
    ) -> None:
        """Raise `PDCapabilityError` if the continuation is not supported."""

        # Classification A: the rank-to-rank transfer currently requires matching
        # TP sizes and does not yet support resharding between prefill/decode.
        if self.require_equal_tp_size and source_tp_size != target_tp_size:
            raise PDCapabilityError(
                f"TP size mismatch: prefill={source_tp_size} decode={target_tp_size}"
            )

        # Classification A: only local cuda_ipc topology is implemented;
        # cross-node transport would need RDMA/SHM backend wiring.
        if not is_local and not self.allow_cross_node:
            raise PDCapabilityError("cross-node PD handoff is not supported")

        # Classification A: the generic continuation only carries token ids.
        # Transferring projected input embeddings needs a different data path.
        if (
            not self.allow_projected_input_embeds
            and getattr(continuation, "input_embeds_are_projected", False)
        ):
            raise PDCapabilityError("projected input embeddings are not supported")

        # Classification A: speculative decode requires spec_info, output_topk,
        # hidden-state replay, etc., which are not in the generic contract.
        if (
            not self.allow_speculative_decoding
            and getattr(continuation, "speculative", False)
        ):
            raise PDCapabilityError("speculative decoding is not supported")

        # Classification B: multimodal resume data is an opaque schema-versioned
        # blob.  PR 2 is text-only, so the allow-list is empty by default.
        # PR 3 can opt-in known schemas.
        mm_resume = getattr(continuation, "multimodal_resume", None) or {}
        if mm_resume:
            schema = mm_resume.get("schema") if isinstance(mm_resume, dict) else None
            if schema not in self.allowed_multimodal_schemas:
                raise PDCapabilityError(
                    f"multimodal resume schema {schema!r} is not supported"
                )

        # Classification B: grammar/structured-output requires reconstructing a
        # model-specific grammar object on the decode side and accepting tokens
        # during process_prebuilt.  PR 2 conservatively rejects it.
        sampling = getattr(continuation, "sampling_params", None) or {}
        if not self.allow_grammar and self._sampling_has_grammar(sampling):
            raise PDCapabilityError("grammar / structured-output sampling is not supported")

        # Classification B: custom logit processors need runtime lookup/registration
        # on the decode side.  PR 2 does not carry the processor object, only a key.
        if not self.allow_custom_logit_processor and getattr(
            continuation, "custom_logit_processor", None
        ):
            raise PDCapabilityError("custom logit processor is not supported")

    @staticmethod
    def _sampling_has_grammar(sampling: dict[str, Any]) -> bool:
        """Detect grammar/json_schema/regex/structural_tag in sampling params."""
        if not isinstance(sampling, dict):
            return False
        keys = ("grammar", "json_schema", "regex", "structural_tag", "ebnf")
        return any(sampling.get(k) for k in keys)
