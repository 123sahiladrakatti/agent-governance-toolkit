"""Optional domain-correctness contract for normalized graph traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from domain import GraphOutcome


@dataclass(frozen=True)
class CorrectnessResult:
    """Domain component's record-relative verdict.

    The first four fields are the stable customer-facing contract.  The
    remaining attribution fields let richer implementations explain a failure
    without forcing domain concepts into the framework.
    """

    passed: bool
    expected: Any
    origin_agent: str | None
    reason: str
    actual: Any = None
    category: str | None = None
    origin_node: str | None = None
    related_agents: tuple[str, ...] = ()
    checks: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CorrectnessComponent(Protocol):
    """Customer-supplied correctness component.

    ``trace`` is the normalized agent graph. ``records`` is deliberately
    opaque to the framework and belongs to the customer's domain adapter.
    """

    def check(self, trace: GraphOutcome, records: object) -> CorrectnessResult:
        ...


__all__ = ["CorrectnessComponent", "CorrectnessResult"]
