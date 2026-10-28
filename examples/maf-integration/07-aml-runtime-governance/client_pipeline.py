"""Client-pipeline JSON: the external payload a client sends for one pipeline run.

The format is domain-agnostic (see ``client_pipeline_schema.md``). This module
validates a payload and normalizes it into the ``GraphOutcome`` that governance
consumes. Topology and permissions come only from the payload's declared
``system`` block and are never inferred from the observed ``run``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from domain import GraphOutcome, GraphStep
from system_description import (
    AgentDescription,
    PipelineRunTrace,
    SystemDescription,
    TraceStep,
    adapt_trace,
)

CLIENT_PIPELINE_FORMAT = "client-pipeline/v1"

# Keys a run step may never carry: authority is declared, not observed.
_FORBIDDEN_STEP_KEYS = frozenset({"permissions", "allowed_actions", "received_from"})

RecordCodec = tuple[Callable[[object], Any], Callable[[Any], object]]


class ClientPipelineError(ValueError):
    """The client-pipeline payload is malformed; nothing was ingested."""


@dataclass(frozen=True)
class IngestedClientPipeline:
    description: SystemDescription
    outcome: GraphOutcome
    graph_policy: str | None
    action_policy: str | None


def _require(container: Mapping[str, Any], key: str, kind: type | tuple[type, ...], where: str) -> Any:
    if key not in container:
        raise ClientPipelineError(f"{where}: missing required field {key!r}")
    value = container[key]
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        names = kind.__name__ if isinstance(kind, type) else " or ".join(k.__name__ for k in kind)
        raise ClientPipelineError(f"{where}.{key}: expected {names}, got {type(value).__name__}")
    return value


def _optional(container: Mapping[str, Any], key: str, kind: type | tuple[type, ...], where: str, default: Any) -> Any:
    if container.get(key) is None:
        return default
    return _require(container, key, kind, where)


def _string_list(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ClientPipelineError(f"{where}: expected a list of strings")
    return tuple(value)


def _numbers(value: Any, where: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    ):
        raise ClientPipelineError(f"{where}: expected a list of numbers")
    return tuple(float(item) for item in value)


def _parse_system(system: Mapping[str, Any], codecs: Mapping[str, RecordCodec]) -> SystemDescription:
    where = "system"
    schema = _require(system, "schema", str, where)
    agents_raw = _require(system, "agents", list, where)
    topology_raw = _require(system, "topology", dict, where)
    permissions_raw = _require(system, "permissions", dict, where)
    if not agents_raw:
        raise ClientPipelineError("system.agents: at least one agent is required")

    agent_ids: list[str] = []
    node_ids: dict[str, str] = {}
    for index, agent in enumerate(agents_raw):
        if not isinstance(agent, dict):
            raise ClientPipelineError(f"system.agents[{index}]: expected an object")
        agent_id = _require(agent, "id", str, f"system.agents[{index}]")
        agent_ids.append(agent_id)
        node_ids[agent_id] = str(_optional(agent, "node_id", str, f"system.agents[{index}]", index + 1))

    declared = set(agent_ids)
    if set(permissions_raw) != declared:
        missing = sorted(declared - set(permissions_raw))
        extra = sorted(set(permissions_raw) - declared)
        raise ClientPipelineError(
            f"system.permissions must declare every agent exactly once (missing={missing}, undeclared={extra})"
        )
    topology = {
        agent: _string_list(parents, f"system.topology.{agent}")
        for agent, parents in topology_raw.items()
    }
    encode, decode = codecs.get(schema, (lambda value: value, lambda value: value))
    try:
        return SystemDescription(
            name=_require(system, "name", str, where),
            schema=schema,
            agents=tuple(
                AgentDescription(
                    agent=agent_id,
                    node_id=node_ids[agent_id],
                    allowed_actions=frozenset(
                        _string_list(permissions_raw[agent_id], f"system.permissions.{agent_id}")
                    ),
                )
                for agent_id in agent_ids
            ),
            topology=topology,
            final_agent=_require(system, "final_agent", str, where),
            final_action=_optional(system, "final_action", str, where, "commit_result"),
            resolver_action=_optional(system, "resolver_action", str, where, "resolve_behavioral_correctness"),
            handoff_action=_optional(system, "handoff_action", str, where, "graph_handoff"),
            resource_prefix=_optional(system, "resource_prefix", str, where, "pipeline-run"),
            escalation_outcomes=frozenset(
                _string_list(system.get("escalation_outcomes", []), "system.escalation_outcomes")
            ),
            encode_records=encode,
            decode_records=decode,
        )
    except ClientPipelineError:
        raise
    except ValueError as exc:  # SystemDescription's own structural checks
        raise ClientPipelineError(f"system: {exc}") from exc


def _parse_step(step: Any, index: int, description: SystemDescription) -> TraceStep:
    where = f"run.steps[{index}]"
    if not isinstance(step, dict):
        raise ClientPipelineError(f"{where}: expected an object")
    forbidden = sorted(_FORBIDDEN_STEP_KEYS & set(step))
    if forbidden:
        raise ClientPipelineError(
            f"{where}: {forbidden} may not appear in a run step; declare them in system.permissions/topology"
        )
    agent = _require(step, "agent", str, where)
    try:
        parents = description.parents_for(agent)
    except KeyError:
        raise ClientPipelineError(f"{where}.agent: {agent!r} is not declared in system.agents") from None
    output = _require(step, "output", dict, where)
    evidence = _optional(output, "evidence", dict, f"{where}.output", {})
    received = _optional(step, "received", dict, where, {})

    findings: dict[str, str] = {}
    versions: dict[str, int] = {}
    values: dict[str, tuple[float, ...]] = {}
    for parent, handoff in received.items():
        handoff_where = f"{where}.received.{parent}"
        if parent not in parents:
            raise ClientPipelineError(
                f"{handoff_where}: {agent!r} has no declared handoff from {parent!r} in system.topology"
            )
        if not isinstance(handoff, dict):
            raise ClientPipelineError(f"{handoff_where}: expected an object")
        if "finding" in handoff:
            findings[parent] = _require(handoff, "finding", str, handoff_where)
        if "version" in handoff:
            versions[parent] = _require(handoff, "version", int, handoff_where)
        if "values" in handoff:
            values[parent] = _numbers(handoff["values"], f"{handoff_where}.values")

    return TraceStep(
        agent=agent,
        action_type=_require(step, "action", str, where),
        finding=_require(output, "finding", str, f"{where}.output"),
        flagged=_optional(output, "flagged", bool, f"{where}.output", False),
        used_amounts=_numbers(evidence.get("values", []), f"{where}.output.evidence.values"),
        used_aggregate=float(_optional(evidence, "aggregate", (int, float), f"{where}.output.evidence", 0.0)),
        disposition=_optional(output, "disposition", str, f"{where}.output", None),
        acted_account_id=_optional(output, "target", str, f"{where}.output", None),
        received_amounts_by_parent=values,
        received_findings_by_parent=findings,
        finding_version=_optional(output, "version", int, f"{where}.output", 1),
        received_versions_by_parent=versions,
    )


def  ingest_client_pipeline(
    payload: Mapping[str, Any] | str,
    record_codecs: Mapping[str, RecordCodec] | None = None,
) -> IngestedClientPipeline:
    """Validate one client-pipeline payload and return its declared system and outcome.

    ``record_codecs`` maps a ``system.schema`` to ``(encode, decode)`` functions for
    domains whose correctness plug-in needs typed records; others keep raw JSON.
    """

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ClientPipelineError(f"payload is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ClientPipelineError("payload: expected a JSON object")
    if payload.get("format") != CLIENT_PIPELINE_FORMAT:
        raise ClientPipelineError(
            f"payload.format: expected {CLIENT_PIPELINE_FORMAT!r}, got {payload.get('format')!r}"
        )

    description = _parse_system(_require(payload, "system", dict, "payload"), record_codecs or {})
    run = _require(payload, "run", dict, "payload")
    steps_raw = _require(run, "steps", list, "run")
    steps = tuple(_parse_step(step, index, description) for index, step in enumerate(steps_raw))

    records_raw = _require(payload, "records", (dict, list), "payload")
    try:
        records = description.decode_records(records_raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ClientPipelineError(f"records: could not decode for schema {description.schema!r}: {exc}") from exc

    resource_id = _optional(
        run, "resource_id", str, "run",
        f"{description.resource_prefix}:{_optional(run, 'run_id', str, 'run', 'unknown')}",
    )
    try:
        outcome = adapt_trace(description, PipelineRunTrace(records=records, steps=steps, resource_id=resource_id))
    except (KeyError, ValueError) as exc:
        raise ClientPipelineError(f"run: {exc}") from exc

    policy = _optional(payload, "policy", dict, "payload", {})
    graph_policy = _optional(policy, "graph", dict, "policy", None)
    action_policy = _optional(policy, "action", dict, "policy", None)
    return IngestedClientPipeline(
        description=description,
        outcome=outcome,
        graph_policy=json.dumps(graph_policy) if graph_policy is not None else None,
        action_policy=json.dumps(action_policy) if action_policy is not None else None,
    )


def ingest_client_json(
    payload: Mapping[str, Any] | str,
    record_codecs: Mapping[str, RecordCodec] | None = None,
) -> GraphOutcome:
    """Ingestion entry point: client-pipeline JSON -> normalized ``GraphOutcome``."""

    return ingest_client_pipeline(payload, record_codecs).outcome


def run_to_json(
    steps: tuple[GraphStep, ...],
    run_id: str,
    resource_id: str,
) -> dict[str, Any]:
    """Serialize an observed run (no declarations) into the client-pipeline ``run`` block."""

    def run_step(step: GraphStep) -> dict[str, Any]:
        output: dict[str, Any] = {
            "finding": step.finding,
            "flagged": step.flagged,
            "version": step.finding_version,
        }
        if step.disposition is not None:
            output["disposition"] = step.disposition
        if step.acted_account_id is not None:
            output["target"] = step.acted_account_id
        if step.used_amounts or step.used_aggregate:
            output["evidence"] = {"values": list(step.used_amounts), "aggregate": step.used_aggregate}
        entry: dict[str, Any] = {"agent": step.agent, "action": step.action_type, "output": output}
        parents = list(dict.fromkeys([
            *step.received_findings_by_parent,
            *step.received_versions_by_parent,
            *step.received_amounts_by_parent,
        ]))
        if parents:
            entry["received"] = {}
            for parent in parents:
                handoff: dict[str, Any] = {}
                if parent in step.received_findings_by_parent:
                    handoff["finding"] = step.received_findings_by_parent[parent]
                if parent in step.received_versions_by_parent:
                    handoff["version"] = step.received_versions_by_parent[parent]
                if parent in step.received_amounts_by_parent:
                    handoff["values"] = list(step.received_amounts_by_parent[parent])
                entry["received"][parent] = handoff
        return entry

    return {"run_id": run_id, "resource_id": resource_id, "steps": [run_step(step) for step in steps]}


def to_client_json(
    description: SystemDescription,
    steps: tuple[GraphStep, ...],
    records: object,
    run_id: str,
    resource_id: str | None = None,
    roles: Mapping[str, str] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a pipeline's declared system and observed run into client-pipeline JSON.

    Run steps carry only what was observed; declared permissions and topology are
    taken from ``description``, never from the steps.
    """

    payload: dict[str, Any] = {
        "format": CLIENT_PIPELINE_FORMAT,
        "system": {
            "name": description.name,
            "schema": description.schema,
            "agents": [
                {"id": agent.agent, "node_id": agent.node_id,
                 **({"role": roles[agent.agent]} if roles and agent.agent in roles else {})}
                for agent in description.agents
            ],
            "topology": {agent: list(parents) for agent, parents in description.topology.items()},
            "permissions": {agent.agent: sorted(agent.allowed_actions) for agent in description.agents},
            "final_agent": description.final_agent,
            "final_action": description.final_action,
            "resolver_action": description.resolver_action,
            "handoff_action": description.handoff_action,
            "resource_prefix": description.resource_prefix,
            "escalation_outcomes": sorted(description.escalation_outcomes),
        },
        "run": run_to_json(steps, run_id, resource_id or f"{description.resource_prefix}:{run_id}"),
        "records": description.encode_records(records),
    }
    if policy is not None:
        payload["policy"] = dict(policy)
    return payload


# ---------------------------------------------------------------------------
# Pipeline definitions: the client's static system + policy, supplied once.
# Each live case contributes its own run and records at runtime.
# ---------------------------------------------------------------------------
PIPELINE_DEFINITION_FORMAT = "client-pipeline-definition/v1"


def load_pipeline_definition(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load and validate a client pipeline definition (system + policy, no run)."""

    if isinstance(source, Mapping):
        definition: Any = dict(source)
        where = "pipeline definition"
    else:
        where = Path(source).name
        try:
            definition = json.loads(Path(source).read_text(encoding="utf-8"))
        except OSError as exc:
            raise ClientPipelineError(f"{where}: could not read pipeline definition: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ClientPipelineError(f"{where}: not valid JSON: {exc}") from exc
    if not isinstance(definition, dict):
        raise ClientPipelineError(f"{where}: expected a JSON object")
    if definition.get("format") != PIPELINE_DEFINITION_FORMAT:
        raise ClientPipelineError(
            f"{where}.format: expected {PIPELINE_DEFINITION_FORMAT!r}, got {definition.get('format')!r}"
        )
    case_keys = sorted({"run", "records"} & set(definition))
    if case_keys:
        raise ClientPipelineError(
            f"{where}: a pipeline definition may not contain {case_keys}; runs are supplied per case"
        )
    _parse_system(_require(definition, "system", dict, where), {})
    policy = _require(definition, "policy", dict, where)
    _require(policy, "graph", dict, f"{where}.policy")
    _require(policy, "action", dict, f"{where}.policy")
    return definition


def compose_client_payload(
    definition: Mapping[str, Any],
    run: Mapping[str, Any],
    records: Any,
) -> dict[str, Any]:
    """Combine a pipeline definition with one case's run and records into a payload."""

    return {
        "format": CLIENT_PIPELINE_FORMAT,
        "system": definition["system"],
        "policy": definition["policy"],
        "run": dict(run),
        "records": records,
    }


__all__ = [
    "CLIENT_PIPELINE_FORMAT",
    "PIPELINE_DEFINITION_FORMAT",
    "ClientPipelineError",
    "IngestedClientPipeline",
    "compose_client_payload",
    "ingest_client_json",
    "ingest_client_pipeline",
    "load_pipeline_definition",
    "run_to_json",
    "to_client_json",
]
