"""Healthcare care-path diamond executed as a LangGraph ``StateGraph``.

Same pipeline as ``healthcare.client_payload.build_healthcare_steps``, but graph
construction and execution are LangGraph's: one node per agent,
``patient-intake`` fans out to the three review nodes (run in parallel in one
superstep), and ``care-decision`` fans in once all three have written their
output to the shared state.

The agents stay deterministic: each node is a thin wrapper around the existing
healthcare logic (``allergy_finding``, ``coverage_finding``,
``care_disposition``). No LLM is called. After execution the LangGraph state is
normalized with ``adapt_trace`` into the same ``GraphStep`` tuple that the
reference steps produce.
"""

from __future__ import annotations

import operator
from functools import lru_cache
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from domain import GraphStep
from healthcare.client_payload import (
    INTAKE_FINDING,
    MEDICATION_FINDING,
    PRIOR_ALLERGY_STATE,
    allergy_finding,
    allergy_version,
    care_disposition,
    coverage_finding,
)
from healthcare.system import (
    ALLERGY_AGENT,
    CARE_DECISION_AGENT,
    COVERAGE_AGENT,
    HEALTHCARE_NODE_ID,
    HEALTHCARE_SYSTEM_DESCRIPTION,
    HealthcareCase,
    INTAKE_AGENT,
    MEDICATION_AGENT,
)
from system_description import PipelineRunTrace, TraceStep, adapt_trace

REVIEW_BRANCHES = (ALLERGY_AGENT, MEDICATION_AGENT, COVERAGE_AGENT)
HEALTHCARE_ACTION = {
    INTAKE_AGENT: "route_care_request",
    ALLERGY_AGENT: "evaluate_allergy_safety",
    MEDICATION_AGENT: "validate_medication_order",
    COVERAGE_AGENT: "verify_coverage",
    CARE_DECISION_AGENT: "merge_care_decision",
}


class CarePathState(TypedDict):
    """Shared LangGraph state for one patient case's run."""

    case: HealthcareCase
    stale_allergy_view: bool
    # Each node writes its own entry; the reducer merges the parallel branch writes.
    steps: Annotated[dict[str, TraceStep], operator.or_]
    # Order in which LangGraph executed the nodes (parallel branches share a superstep).
    executed: Annotated[list[str], operator.add]


def _step(agent: str, finding: str, case: HealthcareCase, flagged: bool = False, **extra) -> TraceStep:
    return TraceStep(
        agent=agent,
        action_type=HEALTHCARE_ACTION[agent],
        finding=finding,
        flagged=flagged,
        acted_account_id=case.patient_id,
        **extra,
    )


def _write(agent: str, step: TraceStep) -> dict:
    return {"steps": {agent: step}, "executed": [agent]}


def intake_node(state: CarePathState) -> dict:
    return _write(INTAKE_AGENT, _step(INTAKE_AGENT, INTAKE_FINDING, state["case"]))


def allergy_node(state: CarePathState) -> dict:
    case = state["case"]
    finding, flagged = allergy_finding(case)
    return _write(ALLERGY_AGENT, _step(
        ALLERGY_AGENT, finding, case, flagged, finding_version=allergy_version(flagged),
    ))


def medication_node(state: CarePathState) -> dict:
    return _write(MEDICATION_AGENT, _step(MEDICATION_AGENT, MEDICATION_FINDING, state["case"]))


def coverage_node(state: CarePathState) -> dict:
    case = state["case"]
    return _write(COVERAGE_AGENT, _step(
        COVERAGE_AGENT, coverage_finding(case), case, not case.coverage_authorized,
    ))


def care_decision_node(state: CarePathState) -> dict:
    """Fan-in: merge the three review outputs as this agent's case-state view holds them."""

    case = state["case"]
    view = {
        agent: {"finding": state["steps"][agent].finding, "version": state["steps"][agent].finding_version}
        for agent in REVIEW_BRANCHES
    }
    if state["stale_allergy_view"]:
        view[ALLERGY_AGENT] = dict(PRIOR_ALLERGY_STATE)
    disposition = care_disposition(view[ALLERGY_AGENT]["finding"], view[COVERAGE_AGENT]["finding"])
    return _write(CARE_DECISION_AGENT, _step(
        CARE_DECISION_AGENT,
        f"care-decision:{disposition.lower()}",
        case,
        disposition != "APPROVE_TREATMENT",
        disposition=disposition,
        received_findings_by_parent={agent: view[agent]["finding"] for agent in REVIEW_BRANCHES},
        received_versions_by_parent={agent: view[agent]["version"] for agent in REVIEW_BRANCHES},
    ))


NODES = {
    INTAKE_AGENT: intake_node,
    ALLERGY_AGENT: allergy_node,
    MEDICATION_AGENT: medication_node,
    COVERAGE_AGENT: coverage_node,
    CARE_DECISION_AGENT: care_decision_node,
}


@lru_cache(maxsize=1)
def build_care_path_graph():
    """Compile the care-path StateGraph: intake -> (3 parallel reviews) -> care decision."""

    builder = StateGraph(CarePathState)
    for agent, node in NODES.items():
        builder.add_node(agent, node)
    builder.add_edge(START, INTAKE_AGENT)
    for branch in REVIEW_BRANCHES:
        builder.add_edge(INTAKE_AGENT, branch)             # fan-out
    builder.add_edge(list(REVIEW_BRANCHES), CARE_DECISION_AGENT)  # fan-in: waits for all three
    builder.add_edge(CARE_DECISION_AGENT, END)
    return builder.compile()


def run_care_path_graph(case: HealthcareCase, stale_allergy_view: bool = False) -> CarePathState:
    """Execute the compiled graph for one case and return the final LangGraph state."""

    return build_care_path_graph().invoke({
        "case": case,
        "stale_allergy_view": stale_allergy_view,
        "steps": {},
        "executed": [],
    })


def build_langgraph_healthcare_steps(
    case: HealthcareCase,
    stale_allergy_view: bool = False,
) -> tuple[GraphStep, ...]:
    """LangGraph counterpart of ``build_healthcare_steps``: same signature, same output."""

    state = run_care_path_graph(case, stale_allergy_view)
    return adapt_trace(
        HEALTHCARE_SYSTEM_DESCRIPTION,
        PipelineRunTrace(
            records=case,
            steps=tuple(state["steps"][agent] for agent in HEALTHCARE_NODE_ID),
            resource_id=f"healthcare-case:{case.case_id}",
        ),
    ).steps


__all__ = [
    "CarePathState",
    "build_care_path_graph",
    "build_langgraph_healthcare_steps",
    "run_care_path_graph",
]
