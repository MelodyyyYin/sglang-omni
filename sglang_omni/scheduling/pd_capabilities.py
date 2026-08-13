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

        if self.require_equal_tp_size and source_tp_size != target_tp_size:
            raise PDCapabilityError(
                f"TP size mismatch: prefill={source_tp_size} decode={target_tp_size}"
            )

        if not is_local and not self.allow_cross_node:
            raise PDCapabilityError("cross-node PD handoff is not supported")

        if getattr(continuation, "input_embeds_are_projected", False):
            raise PDCapabilityError("projected input embeddings are not supported")

        if getattr(continuation, "speculative", False):
            raise PDCapabilityError("speculative decoding is not supported")

        mm_resume = getattr(continuation, "multimodal_resume", None) or {}
        if mm_resume:
            schema = mm_resume.get("schema") if isinstance(mm_resume, dict) else None
            if schema not in self.allowed_multimodal_schemas:
                raise PDCapabilityError(
                    f"multimodal resume schema {schema!r} is not supported"
                )

        sampling = getattr(continuation, "sampling_params", None) or {}
        if not self.allow_grammar and self._sampling_has_grammar(sampling):
            raise PDCapabilityError("grammar / structured-output sampling is not supported")

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
