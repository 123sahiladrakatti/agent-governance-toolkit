"""Domain-neutral multi-agent context-handoff simulation.

The simulator models a common operational failure: a downstream agent receives
only the portion of its parent's context that fits the declared handoff budget.
No failure flag is planted. Context is lost only when the submitted envelope is
larger than that budget.
"""

from __future__ import annotations

from typing import Any


AGENT_NAMES = (
    "intake-agent",
    "analysis-agent",
    "review-agent",
    "coordination-agent",
    "decision-agent",
)
DEFAULT_AGENT_COUNT = 4
DEFAULT_HANDOFF_CAPACITY = 500


def _graph_policy(schema: str) -> dict[str, Any]:
    return {
        "apiVersion": "governance.toolkit/v1",
        "version": "1.0",
        "name": "context-simulation-graph-policy",
        "agents": ["*"],
        "scope": "global",
        "default_action": "allow",
        "rules": [
            {"name": "allow-authorized-handoff", "stage": "pre_tool", "condition": "handoff.status == 'authorized'", "action": "allow", "priority": 100},
            {"name": "deny-unauthorized-handoff", "stage": "pre_tool", "condition": "handoff.status != 'authorized'", "action": "deny", "priority": 100},
            {"name": "deny-invalid-trace-schema", "stage": "post_tool", "condition": f"governance_graph.schema != '{schema}'", "action": "deny", "priority": 100},
        ],
    }


ACTION_POLICY = {
    "apiVersion": "governance.toolkit/v1",
    "version": "1.0",
    "name": "context-simulation-action-governance",
    "agents": ["*"],
    "scope": "global",
    "default_action": "deny",
    "rules": [
        {"name": "deny-unauthorized", "stage": "post_tool", "condition": "action.status == 'unauthorized'", "action": "deny", "priority": 200},
        {"name": "review-context-conflict", "stage": "post_tool", "condition": "governance.status == 'conflict'", "action": "require_approval", "priority": 180},
        {"name": "deny-incorrect", "stage": "post_tool", "condition": "governance.status == 'failed'", "action": "deny", "priority": 150},
        {"name": "review-content", "stage": "post_tool", "condition": "nemo.status == 'flagged'", "action": "require_approval", "priority": 100},
        {"name": "allow-clean", "stage": "post_tool", "condition": "action.status == 'authorized' and governance.status == 'passed' and nemo.status == 'passed'", "action": "allow", "priority": 50},
        {"name": "allow-content-unavailable", "stage": "post_tool", "condition": "action.status == 'authorized' and governance.status == 'passed' and nemo.status == 'unavailable'", "action": "allow", "priority": 40},
    ],
}


def build_context_bundle(
    user_input: str,
    critical_requirements: list[str],
    agent_count: int = DEFAULT_AGENT_COUNT,
    handoff_capacity: int = DEFAULT_HANDOFF_CAPACITY,
) -> dict[str, Any]:
    """Build a customer bundle whose handoffs obey a fixed context capacity."""

    request = user_input.strip()
    if not request:
        raise ValueError("User input is required")
    if not 2 <= agent_count <= len(AGENT_NAMES):
        raise ValueError(f"agent_count must be between 2 and {len(AGENT_NAMES)}")
    if handoff_capacity < 1:
        raise ValueError("handoff_capacity must be positive")

    requirements = [item.strip() for item in critical_requirements if item.strip()]
    requirements_text = "\n".join(f"- {item}" for item in requirements) or "- none declared"
    envelope = f"USER REQUEST\n{request}\n\nMUST-PRESERVE CONTEXT\n{requirements_text}"

    agents = list(AGENT_NAMES[: agent_count - 1]) + [AGENT_NAMES[-1]]
    action_for = {
        agent: "accept_context" if index == 0 else
        "commit_result" if index == len(agents) - 1 else "process_context"
        for index, agent in enumerate(agents)
    }
    system_agents = [
        {"id": agent, "node_id": chr(65 + index), "allowed_actions": [action_for[agent]]}
        for index, agent in enumerate(agents)
    ]
    topology = {
        agent: [] if index == 0 else [agents[index - 1]]
        for index, agent in enumerate(agents)
    }

    steps: list[dict[str, Any]] = []
    emitted_context = envelope
    steps.append({
        "agent": agents[0],
        "action_type": action_for[agents[0]],
        "finding": emitted_context,
        "finding_version": 1,
    })
    losses: list[dict[str, Any]] = []
    for index, agent in enumerate(agents[1:], start=1):
        parent = agents[index - 1]
        received_context = emitted_context[:handoff_capacity]
        missing_requirements = [item for item in requirements if item not in received_context]
        lost_characters = len(emitted_context) - len(received_context)
        if lost_characters:
            losses.append({
                "sender": parent,
                "recipient": agent,
                "sent_characters": len(emitted_context),
                "received_characters": len(received_context),
                "lost_characters": lost_characters,
                "missing_requirements": missing_requirements,
            })
        step = {
            "agent": agent,
            "action_type": action_for[agent],
            "finding": received_context,
            "finding_version": 1,
            "received_findings_by_parent": {parent: received_context},
            "received_versions_by_parent": {parent: 1},
        }
        if index == len(agents) - 1:
            step["disposition"] = "CONTINUE"
        steps.append(step)
        emitted_context = received_context

    schema = "context-handoff-simulation/v1"
    return {
        "system_description": {
            "name": "context-handoff-simulation",
            "schema": schema,
            "agents": system_agents,
            "topology": topology,
            "final_agent": agents[-1],
            "final_action": "commit_result",
            "resource_prefix": "context-run",
        },
        "trace": {
            "resource_id": "context-run:interactive",
            "records": {
                "original_input": request,
                "critical_requirements": requirements,
                "context_envelope": envelope,
                "handoff_capacity": handoff_capacity,
                "handoff_losses": losses,
            },
            "steps": steps,
        },
        "policies": {"graph": _graph_policy(schema), "action": ACTION_POLICY},
        "nemo": {
            "review_instruction": "Review this authorized agent output using the customer content policy."
        },
        "correctness": None,
    }


__all__ = ["build_context_bundle"]
