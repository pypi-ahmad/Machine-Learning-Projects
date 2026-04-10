"""Document Type Classifier Router — routing logic.

Given a classification result, decide which downstream pipeline should
handle the document.  This module is fully independent of the classifier
— it only consumes classification outputs.

In production, each ``route_*`` method would dispatch to a message
queue, REST endpoint, or local processing function.  Here they are
stubs that return a ``RoutingDecision`` dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from classifier import ClassificationResult
from config import ROUTE_TABLE, RouterConfig


@dataclass
class RoutingDecision:
    """The result of routing a classified document."""

    document_type: str       # predicted class
    pipeline: str            # downstream pipeline name
    confidence: float
    routed: bool             # True if confidence >= threshold
    reason: str              # human-readable explanation


# ── Pipeline stub hooks ───────────────────────────────────
# Each function is a placeholder.  Replace with actual logic
# (e.g. submit to a queue, call a microservice, run OCR, …).

def _stub_handler(doc_type: str, pipeline: str) -> str:
    """Default no-op handler — returns a status string."""
    return f"[STUB] Document '{doc_type}' dispatched to '{pipeline}'"


# Registry of concrete handlers (can be extended at runtime)
_PIPELINE_HANDLERS: dict[str, Callable[[str, str], str]] = {}


def register_handler(pipeline: str, fn: Callable[[str, str], str]) -> None:
    """Register a concrete handler for a pipeline name."""
    _PIPELINE_HANDLERS[pipeline] = fn


def get_handler(pipeline: str) -> Callable[[str, str], str]:
    """Return the handler for *pipeline*, falling back to the stub."""
    return _PIPELINE_HANDLERS.get(pipeline, _stub_handler)


# ── Router ────────────────────────────────────────────────

class DocumentRouter:
    """Route classified documents to downstream pipelines."""

    def __init__(self, cfg: RouterConfig | None = None) -> None:
        self.cfg = cfg or RouterConfig()

    def route(self, result: ClassificationResult) -> RoutingDecision:
        """Decide which pipeline should handle *result*."""
        below_threshold = result.confidence < self.cfg.confidence_threshold
        pipeline = self.cfg.route_table.get(
            result.class_name, self.cfg.fallback_pipeline
        )

        if below_threshold:
            pipeline = self.cfg.fallback_pipeline
            reason = (
                f"Confidence {result.confidence:.2%} below threshold "
                f"({self.cfg.confidence_threshold:.2%}) — sent to manual review"
            )
        else:
            reason = (
                f"Document type '{result.display_label}' "
                f"(conf {result.confidence:.2%}) → {pipeline}"
            )

        return RoutingDecision(
            document_type=result.class_name,
            pipeline=pipeline,
            confidence=result.confidence,
            routed=not below_threshold,
            reason=reason,
        )

    def route_batch(
        self, results: list[ClassificationResult]
    ) -> list[RoutingDecision]:
        """Route a batch of classification results."""
        return [self.route(r) for r in results]

    def execute(self, decision: RoutingDecision) -> str:
        """Execute the routing decision through the registered handler."""
        handler = get_handler(decision.pipeline)
        return handler(decision.document_type, decision.pipeline)

    def route_and_execute(
        self, result: ClassificationResult
    ) -> tuple[RoutingDecision, str]:
        """Route then execute in one call."""
        decision = self.route(result)
        status = self.execute(decision)
        return decision, status
