"""Declared pipeline structure and run-trace ingestion.

Permissions and topology live in ``SystemDescription`` and are never inferred
from observed run data.  The adapter combines that trusted declaration with a
customer trace to produce the normalized ``GraphOutcome`` used by governance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from domain import GraphOutcome, GraphStep


def _identity(value):
    return value


@dataclass(frozen=True)
class AgentDescription:
    agent: str
    node_id: str
    allowed_actions: frozenset[str]


@dataclass(frozen=True)
class SystemDescription:
    name: str
    schema: str
    agents: tuple[AgentDescription, ...]
    topology: dict[str, tuple[str, ...]]
    final_agent: str
    final_action: str = "commit_result"
    resolver_action: str = "resolve_behavioral_correctness"
    handoff_action: str = "graph_handoff"
    resource_prefix: str = "pipeline-run"
    escalation_outcomes: frozenset[str] = frozenset()
    encode_records: Callable[[object], Any] = field(default=_identity, repr=False, compare=False)
    decode_records: Callable[[Any], object] = field(default=_identity, repr=False, compare=False)

    def __post_init__(self) -> None:
        declared = {agent.agent for agent in self.agents}
        if len(declared) != len(self.agents):
            raise ValueError("System description contains duplicate agent identifiers")
        if self.final_agent not in declared:
            raise ValueError(f"Final agent {self.final_agent!r} is not declared")
        if set(self.topology) != declared:
            raise ValueError("Topology must explicitly declare parents for every agent")
        unknown_parents = {
            parent
            for parents in self.topology.values()
            for parent in parents
            if parent not in declared
        }
        if unknown_parents:
            raise ValueError(f"Topology references undeclared agents: {sorted(unknown_parents)}")

    def agent(self, agent_id: str) -> AgentDescription:
        try:
            return next(agent for agent in self.agents if agent.agent == agent_id)
        except StopIteration as exc:
            raise KeyError(f"Agent {agent_id!r} is not declared") from exc

    def parents_for(self, agent_id: str) -> tuple[str, ...]:
        return self.topology[agent_id]


@dataclass(frozen=True)
class TraceStep:
    """One observed customer step before trusted declarations are attached."""

    agent: str
    action_type: str
    finding: str
    flagged: bool = False
    used_amounts: tuple[float, ...] = ()
    used_aggregate: float = 0.0
    disposition: str | None = None
    acted_account_id: str | None = None
    received_amounts_by_parent: dict[str, tuple[float, ...]] = field(default_factory=dict)
    received_findings_by_parent: dict[str, str] = field(default_factory=dict)
    finding_version: int = 1
    received_versions_by_parent: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineRunTrace:
    records: object
    steps: tuple[TraceStep, ...]
    resource_id: str
    is_star_case: bool = False
    fault: str | None = None
    is_conflict_case: bool = False


def adapt_trace(description: SystemDescription, trace: PipelineRunTrace) -> GraphOutcome:
    """Normalize a run using only explicitly declared topology/permissions."""

    observed_agents = [step.agent for step in trace.steps]
    if len(set(observed_agents)) != len(observed_agents):
        raise ValueError("A run trace must contain at most one step per agent")
    for agent_id in observed_agents:
        description.agent(agent_id)
    if description.final_agent not in observed_agents:
        raise ValueError("Run trace does not contain the declared final agent")

    steps = tuple(
        GraphStep(
            agent=step.agent,
            node_id=description.agent(step.agent).node_id,
            action_type=step.action_type,
            allowed_actions=description.agent(step.agent).allowed_actions,
            received_from=description.parents_for(step.agent),
            finding=step.finding,
            flagged=step.flagged,
            used_amounts=step.used_amounts,
            used_aggregate=step.used_aggregate,
            disposition=step.disposition,
            acted_account_id=step.acted_account_id,
            received_amounts_by_parent=step.received_amounts_by_parent,
            received_findings_by_parent=step.received_findings_by_parent,
            finding_version=step.finding_version,
            received_versions_by_parent=step.received_versions_by_parent,
        )
        for step in trace.steps
    )
    return GraphOutcome(
        alert=trace.records,
        steps=steps,
        is_star_case=trace.is_star_case,
        fault=trace.fault,
        is_conflict_case=trace.is_conflict_case,
        resource_id=trace.resource_id,
    )


def encode_graph_context(outcome: GraphOutcome, description: SystemDescription) -> dict[str, Any]:
    return {
        "schema": description.schema,
        "resource_id": outcome.resource_id,
        "records": description.encode_records(outcome.alert),
        "steps": [{
            "agent": step.agent,
            "node_id": step.node_id,
            "action_type": step.action_type,
            "allowed_actions": sorted(step.allowed_actions),
            "received_from": list(step.received_from),
            "finding": step.finding,
            "flagged": step.flagged,
            "used_amounts": list(step.used_amounts),
            "used_aggregate": step.used_aggregate,
            "disposition": step.disposition,
            "acted_account_id": step.acted_account_id,
            "received_amounts_by_parent": {
                key: list(value) for key, value in step.received_amounts_by_parent.items()
            },
            "received_findings_by_parent": dict(step.received_findings_by_parent),
            "finding_version": step.finding_version,
            "received_versions_by_parent": dict(step.received_versions_by_parent),
        } for step in outcome.steps],
    }


def decode_graph_context(context: dict[str, Any], description: SystemDescription) -> GraphOutcome:
    if context.get("schema") != description.schema:
        raise ValueError(
            f"Expected graph schema {description.schema!r}; received {context.get('schema')!r}"
        )
    records = description.decode_records(context.get("records", context.get("alert")))
    steps = tuple(
        GraphStep(
            agent=step["agent"],
            node_id=description.agent(step["agent"]).node_id,
            action_type=step["action_type"],
            allowed_actions=description.agent(step["agent"]).allowed_actions,
            received_from=description.parents_for(step["agent"]),
            finding=step["finding"],
            flagged=bool(step["flagged"]),
            used_amounts=tuple(step.get("used_amounts", ())),
            used_aggregate=float(step.get("used_aggregate", 0.0)),
            disposition=step.get("disposition"),
            acted_account_id=step.get("acted_account_id"),
            received_amounts_by_parent={
                key: tuple(value)
                for key, value in step.get("received_amounts_by_parent", {}).items()
            },
            received_findings_by_parent=dict(step.get("received_findings_by_parent", {})),
            finding_version=int(step.get("finding_version", 1)),
            received_versions_by_parent={
                key: int(value)
                for key, value in step.get("received_versions_by_parent", {}).items()
            },
        )
        for step in context["steps"]
    )
    return GraphOutcome(
        alert=records,
        steps=steps,
        resource_id=context.get("resource_id"),
    )


__all__ = [
    "AgentDescription",
    "PipelineRunTrace",
    "SystemDescription",
    "TraceStep",
    "adapt_trace",
    "decode_graph_context",
    "encode_graph_context",
]
