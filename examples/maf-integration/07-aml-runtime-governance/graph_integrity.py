"""Domain-agnostic provenance and fan-in consistency checks.

The checker deliberately knows nothing about AML records, findings, or
dispositions.  It verifies only facts carried by the normalized graph trace:
which parent produced a finding, which version the consumer read, and whether
the consumer preserved that finding at fan-in.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain import GraphOutcome


@dataclass(frozen=True)
class GraphIntegrityResult:
    """Result of validating provenance encoded in a ``GraphOutcome``."""

    passed: bool
    category: str | None
    origin_agent: str | None
    origin_node: str | None
    related_agents: tuple[str, ...]
    reason: str
    missing_parents: tuple[str, ...] = ()
    altered_parents: tuple[str, ...] = ()
    stale_parents: tuple[str, ...] = ()
    future_version_parents: tuple[str, ...] = ()


def check_graph_integrity(outcome: GraphOutcome) -> GraphIntegrityResult:
    """Check fan-in provenance and versions without domain-specific truth.

    A step is inconsistent when a declared parent is absent, its consumed
    finding differs from the parent's emitted finding, or the consumed version
    is older/newer than the version that parent emitted.  Missing version
    metadata remains backward-compatible: it is interpreted as the parent's
    current version.
    """

    by_agent = {step.agent: step for step in outcome.steps}
    for consumer in outcome.steps:
        if not consumer.received_from:
            continue

        missing = tuple(parent for parent in consumer.received_from if parent not in by_agent)
        # A simple handoff may declare only its parent edge.  Provenance values
        # are optional there; fan-in consumers opt in by recording findings or
        # versions for the state they actually consumed.
        if not missing and not consumer.received_findings_by_parent and not consumer.received_versions_by_parent:
            continue
        present = tuple(parent for parent in consumer.received_from if parent in by_agent)
        altered = tuple(
            parent
            for parent in present
            if consumer.received_findings_by_parent.get(parent) != by_agent[parent].finding
        )
        stale = tuple(
            parent
            for parent in present
            if consumer.received_versions_by_parent.get(parent, by_agent[parent].finding_version)
            < by_agent[parent].finding_version
        )
        future = tuple(
            parent
            for parent in present
            if consumer.received_versions_by_parent.get(parent, by_agent[parent].finding_version)
            > by_agent[parent].finding_version
        )
        related = tuple(dict.fromkeys((*missing, *altered, *stale, *future)))
        if not related:
            continue

        details: list[str] = []
        if missing:
            details.append(f"missing parents: {', '.join(missing)}")
        if altered:
            details.append(f"altered findings: {', '.join(altered)}")
        if stale:
            versions = ", ".join(
                f"{parent} v{consumer.received_versions_by_parent.get(parent, 1)}"
                f" < v{by_agent[parent].finding_version}"
                for parent in stale
            )
            details.append(f"stale case state: {versions}")
        if future:
            versions = ", ".join(
                f"{parent} v{consumer.received_versions_by_parent[parent]}"
                f" > v{by_agent[parent].finding_version}"
                for parent in future
            )
            details.append(f"invalid future versions: {versions}")

        return GraphIntegrityResult(
            passed=False,
            category="unreconciled-conflict",
            origin_agent=consumer.agent,
            origin_node=consumer.node_id,
            related_agents=related,
            reason=(
                f"{consumer.agent} (node {consumer.node_id}) did not reconcile its parent state: "
                f"{'; '.join(details)}."
            ),
            missing_parents=missing,
            altered_parents=altered,
            stale_parents=stale,
            future_version_parents=future,
        )

    return GraphIntegrityResult(
        passed=True,
        category=None,
        origin_agent=None,
        origin_node=None,
        related_agents=(),
        reason="Graph provenance and consumed parent versions are consistent.",
    )


__all__ = ["GraphIntegrityResult", "check_graph_integrity"]
