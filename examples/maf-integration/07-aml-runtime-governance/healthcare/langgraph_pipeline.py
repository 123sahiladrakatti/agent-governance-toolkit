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
from typing import Annotated, Any, Optional, TypedDict

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

# Care-decision implementations. "deterministic" is the default demo; "llm-context-loss"
# runs the care-decision agent as a real Ollama LLM with a small context window.
DETERMINISTIC_DECISION = "deterministic"
LLM_CONTEXT_LOSS_DECISION = "llm-context-loss"
# Allergy-safety implementations: "deterministic" (default) or "llm-pii" (real Ollama LLM
# with a permissive prompt; its free-text handoff note becomes the step's message).
DETERMINISTIC_ALLERGY = "deterministic"
LLM_PII_ALLERGY = "llm-pii"
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
    decision_mode: str
    llm_config: Optional[Any]
    # Set by the LLM care-decision node: what the agent saw vs. decided (``LLMDecision``).
    llm_decision: Optional[Any]
    allergy_mode: str
    allergy_llm_config: Optional[Any]
    allergy_redact: bool                     # control: add the redaction instruction to the agent's prompt
    # Set by the LLM allergy node: the agent's status and handoff note (``LLMAllergyNote``).
    llm_allergy: Optional[Any]


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


def llm_allergy_node(state: CarePathState) -> dict:
    """Allergy safety by a real LLM: its status is the finding, its handoff note the message."""

    from healthcare.llm_allergy_agent import run_llm_allergy_check

    case = state["case"]
    result = run_llm_allergy_check(case, state.get("allergy_llm_config"), redact=state.get("allergy_redact", False))
    flagged = result.status == "CONTRAINDICATION"
    step = _step(
        ALLERGY_AGENT, result.status.lower(), case, flagged,
        finding_version=allergy_version(flagged), message=result.note,
    )
    return {**_write(ALLERGY_AGENT, step), "llm_allergy": result}


def allergy_node(state: CarePathState) -> dict:
    if state.get("allergy_mode", DETERMINISTIC_ALLERGY) == LLM_PII_ALLERGY:
        return llm_allergy_node(state)
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


def llm_care_decision_node(state: CarePathState) -> dict:
    """Fan-in by a real LLM: record only the review findings the agent actually consumed."""

    from healthcare.llm_decision_agent import run_llm_decision

    case = state["case"]
    branches = {agent: state["steps"][agent] for agent in REVIEW_BRANCHES}
    decision = run_llm_decision(case, branches, state.get("llm_config"))
    consumed = list(decision.consumed)
    lost = ", ".join(decision.not_consumed)
    finding = (
        f"care-decision:{decision.disposition.lower()} (LLM decision agent; used {len(consumed)}/3 review reports"
        + (f"; not in its context: {lost}" if lost else "")
        + ")"
    )
    step = _step(
        CARE_DECISION_AGENT,
        finding,
        case,
        decision.disposition != "APPROVE_TREATMENT",
        disposition=decision.disposition,
        received_findings_by_parent=dict(decision.consumed),
        received_versions_by_parent={agent: branches[agent].finding_version for agent in consumed},
    )
    return {**_write(CARE_DECISION_AGENT, step), "llm_decision": decision}


def care_decision_node(state: CarePathState) -> dict:
    """Fan-in: merge the three review outputs as this agent's case-state view holds them."""

    if state.get("decision_mode", DETERMINISTIC_DECISION) == LLM_CONTEXT_LOSS_DECISION:
        return llm_care_decision_node(state)
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


def run_care_path_graph(
    case: HealthcareCase,
    stale_allergy_view: bool = False,
    decision_mode: str = DETERMINISTIC_DECISION,
    llm_config: Any = None,
    allergy_mode: str = DETERMINISTIC_ALLERGY,
    allergy_llm_config: Any = None,
    allergy_redact: bool = False,
) -> CarePathState:
    """Execute the compiled graph for one case and return the final LangGraph state."""

    if decision_mode not in (DETERMINISTIC_DECISION, LLM_CONTEXT_LOSS_DECISION):
        raise ValueError(f"unsupported decision_mode: {decision_mode!r}")
    if allergy_mode not in (DETERMINISTIC_ALLERGY, LLM_PII_ALLERGY):
        raise ValueError(f"unsupported allergy_mode: {allergy_mode!r}")
    return build_care_path_graph().invoke({
        "case": case,
        "stale_allergy_view": stale_allergy_view,
        "steps": {},
        "executed": [],
        "decision_mode": decision_mode,
        "llm_config": llm_config,
        "llm_decision": None,
        "allergy_mode": allergy_mode,
        "allergy_llm_config": allergy_llm_config,
        "allergy_redact": allergy_redact,
        "llm_allergy": None,
    })


def _normalize(case: HealthcareCase, state: CarePathState) -> tuple[GraphStep, ...]:
    return adapt_trace(
        HEALTHCARE_SYSTEM_DESCRIPTION,
        PipelineRunTrace(
            records=case,
            steps=tuple(state["steps"][agent] for agent in HEALTHCARE_NODE_ID),
            resource_id=f"healthcare-case:{case.case_id}",
        ),
    ).steps


def build_langgraph_healthcare_steps(
    case: HealthcareCase,
    stale_allergy_view: bool = False,
) -> tuple[GraphStep, ...]:
    """LangGraph counterpart of ``build_healthcare_steps``: same signature, same output."""

    return _normalize(case, run_care_path_graph(case, stale_allergy_view))


def build_llm_context_loss_care_path(case: HealthcareCase, llm_config: Any = None) -> tuple[tuple[GraphStep, ...], Any]:
    """Run the care path with deterministic reviews and a real-LLM care-decision agent.

    Returns the normalized steps and the ``LLMDecision``. Nothing is planted: the
    allergy review emits its current finding; whether the care-decision agent
    consumed it depends only on the LLM's context window.
    """

    state = run_care_path_graph(case, decision_mode=LLM_CONTEXT_LOSS_DECISION, llm_config=llm_config)
    return _normalize(case, state), state["llm_decision"]


def build_llm_pii_care_path(
    case: HealthcareCase, allergy_llm_config: Any = None, redact: bool = False
) -> tuple[tuple[GraphStep, ...], Any]:
    """Run the care path with a real-LLM allergy-safety agent; every other agent is deterministic.

    Returns the normalized steps and the ``LLMAllergyNote``. Nothing is planted: any
    personal data in the note is the model's own output. ``redact=True`` is the control:
    the same prompt plus one redaction instruction.
    """

    state = run_care_path_graph(
        case, allergy_mode=LLM_PII_ALLERGY, allergy_llm_config=allergy_llm_config, allergy_redact=redact
    )
    return _normalize(case, state), state["llm_allergy"]


__all__ = [
    "DETERMINISTIC_ALLERGY",
    "DETERMINISTIC_DECISION",
    "LLM_PII_ALLERGY",
    "build_llm_pii_care_path",
    "LLM_CONTEXT_LOSS_DECISION",
    "CarePathState",
    "build_llm_context_loss_care_path",
    "build_care_path_graph",
    "build_langgraph_healthcare_steps",
    "run_care_path_graph",
]
