"""Backend ingestion for customer pipeline-governance bundles.

The Streamlit simulator is deliberately thin: this module validates customer
configuration, attaches trusted topology and permissions, and returns the
normalized objects consumed by ``GraphGovernor``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from correctness import CorrectnessResult
from domain import GraphOutcome
from nemo_guardrails import DEFAULT_CONFIG_PATH, NemoRuntimeConfig
from system_description import (
    AgentDescription,
    PipelineRunTrace,
    SystemDescription,
    TraceStep,
    adapt_trace,
)


@dataclass(frozen=True)
class IngestedPipeline:
    """Validated backend input ready for governance execution."""

    description: SystemDescription
    outcome: GraphOutcome
    graph_policy: str
    action_policy: str
    nemo_settings: NemoRuntimeConfig
    correctness_component: object | None
    source_name: str


class ExpectedResultComponent:
    """Demo bridge for a correctness result included in an ingestion bundle.

    Production integrations inject their own ``CorrectnessComponent`` instead.
    """

    def __init__(self, final_agent: str, config: dict[str, Any]) -> None:
        self.final_agent = final_agent
        self.expected = config["expected"]
        self.origin_agent = config.get("origin_agent")
        self.failure_reason = str(config.get("reason", "Customer correctness check failed."))

    def check(self, trace: GraphOutcome, records: object) -> CorrectnessResult:
        final_step = trace.step(self.final_agent)
        actual = final_step.disposition or final_step.finding
        passed = actual == self.expected
        origin = None if passed else (self.origin_agent or self.final_agent)
        return CorrectnessResult(
            passed=passed,
            expected=self.expected,
            actual=actual,
            origin_agent=origin,
            origin_node=trace.step(origin).node_id if origin else None,
            category=None if passed else "customer-correctness-failure",
            reason=(
                f"Customer correctness component confirmed {actual!r}."
                if passed else self.failure_reason
            ),
            metadata={"records_supplied": records is not None},
        )


def _parse_system(value: dict[str, Any]) -> SystemDescription:
    agents = tuple(
        AgentDescription(
            agent=str(item["id"]),
            node_id=str(item.get("node_id", index + 1)),
            allowed_actions=frozenset(str(action) for action in item["allowed_actions"]),
        )
        for index, item in enumerate(value["agents"])
    )
    topology = {
        str(agent): tuple(str(parent) for parent in parents)
        for agent, parents in value["topology"].items()
    }
    return SystemDescription(
        name=str(value["name"]),
        schema=str(value["schema"]),
        agents=agents,
        topology=topology,
        final_agent=str(value["final_agent"]),
        final_action=str(value.get("final_action", "commit_result")),
        resolver_action=str(value.get("resolver_action", "resolve_behavioral_correctness")),
        handoff_action=str(value.get("handoff_action", "graph_handoff")),
        resource_prefix=str(value.get("resource_prefix", "pipeline-run")),
        escalation_outcomes=frozenset(str(item) for item in value.get("escalation_outcomes", ())),
    )


def _parse_trace(value: dict[str, Any]) -> PipelineRunTrace:
    steps = tuple(
        TraceStep(
            agent=str(item["agent"]),
            action_type=str(item["action_type"]),
            finding=str(item["finding"]),
            flagged=bool(item.get("flagged", False)),
            used_amounts=tuple(float(number) for number in item.get("used_amounts", ())),
            used_aggregate=float(item.get("used_aggregate", 0.0)),
            disposition=item.get("disposition"),
            acted_account_id=item.get("acted_account_id"),
            received_amounts_by_parent={
                str(parent): tuple(float(number) for number in numbers)
                for parent, numbers in item.get("received_amounts_by_parent", {}).items()
            },
            received_findings_by_parent={
                str(parent): str(finding)
                for parent, finding in item.get("received_findings_by_parent", {}).items()
            },
            finding_version=int(item.get("finding_version", 1)),
            received_versions_by_parent={
                str(parent): int(version)
                for parent, version in item.get("received_versions_by_parent", {}).items()
            },
        )
        for item in value["steps"]
    )
    return PipelineRunTrace(
        records=value.get("records", {}),
        steps=steps,
        resource_id=str(value["resource_id"]),
    )


def ingest_bundle(payload: dict[str, Any], source_name: str = "customer bundle") -> IngestedPipeline:
    """Validate and normalize one customer governance bundle."""

    description = _parse_system(payload["system_description"])
    trace = _parse_trace(payload["trace"])
    outcome = adapt_trace(description, trace)

    policies = payload["policies"]
    graph_policy = policies["graph"]
    action_policy = policies["action"]
    if not isinstance(graph_policy, dict) or not isinstance(action_policy, dict):
        raise TypeError("Both customer policies must be JSON objects")

    nemo = payload.get("nemo", {})
    nemo_settings = NemoRuntimeConfig(
        config_path=Path(nemo.get("config_path", DEFAULT_CONFIG_PATH)),
        model=nemo.get("model") or None,
        prompts={str(task): str(prompt) for task, prompt in nemo.get("prompts", {}).items()},
        review_instruction=str(
            nemo.get(
                "review_instruction",
                "Review this authorized agent output using the customer content policy.",
            )
        ),
    )

    correctness_config = payload.get("correctness")
    correctness_component = (
        ExpectedResultComponent(description.final_agent, correctness_config)
        if isinstance(correctness_config, dict) else None
    )
    return IngestedPipeline(
        description=description,
        outcome=outcome,
        graph_policy=json.dumps(graph_policy),
        action_policy=json.dumps(action_policy),
        nemo_settings=nemo_settings,
        correctness_component=correctness_component,
        source_name=source_name,
    )


__all__ = ["IngestedPipeline", "ingest_bundle"]
