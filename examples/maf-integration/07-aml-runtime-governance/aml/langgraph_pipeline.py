"""AML diamond executed as a LangGraph ``StateGraph``.

Same pipeline as ``agent_sim.build_diamond``, but graph construction and
execution are LangGraph's: one node per agent, ``intake`` fans out to the three
branch nodes (run in parallel in one superstep), and ``decision`` fans in once
all three have written their output to the shared state.

The agents stay deterministic: each node is a thin wrapper around the existing
AML logic (``recompute_sanctions``, ``recompute_structuring`` /
``_disposition_for``, ``recompute_kyc``, ``merge_disposition``). No LLM is called.
After execution the LangGraph state is normalized with ``adapt_trace`` into the
same ``GraphStep`` tuple that ``build_diamond`` returns.
"""

from __future__ import annotations

import operator
from functools import lru_cache
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_sim import _disposition_for, _misread
from aml.system import (
    AML_SYSTEM_DESCRIPTION,
    DIAMOND_ACTION,
    DIAMOND_BRANCHES,
    DIAMOND_NODE_ID,
    DISPOSITION_AGENT,
    KYC_AGENT,
    SANCTIONS_AGENT,
    TRANSACTION_AGENT,
    TRIAGE_AGENT,
)
from domain import (
    Alert,
    GraphOutcome,
    GraphStep,
    merge_disposition,
    recompute_kyc,
    recompute_sanctions,
    recompute_structuring,
)
from system_description import PipelineRunTrace, TraceStep, adapt_trace

# The case-state screening entry before the screening agent's update (version 1).
# A decision agent holding a stale view consumes this instead of the current v2.
PRIOR_SCREENING_STATE = {"finding": "no-hit", "version": 1, "flagged": False}

# Decision-node implementations. "deterministic" is the default demo; "llm-context-loss"
# runs the decision agent as a real Ollama LLM with a small context window.
DETERMINISTIC_DECISION = "deterministic"
LLM_CONTEXT_LOSS_DECISION = "llm-context-loss"


class DiamondState(TypedDict):
    """Shared LangGraph state for one alert's run."""

    alert: Alert
    faulted_branch: Optional[str]
    stale_screening_view: bool
    # Each node writes its own entry; the reducer merges the parallel branch writes.
    steps: Annotated[dict[str, TraceStep], operator.or_]
    # Order in which LangGraph executed the nodes (parallel branches share a superstep).
    executed: Annotated[list[str], operator.add]
    decision_mode: str
    llm_config: Optional[Any]
    # Set by the LLM decision node: what the agent saw vs. decided (``LLMDecision``).
    llm_decision: Optional[Any]


def _step(agent: str, finding: str, flagged: bool, alert: Alert, **extra) -> TraceStep:
    return TraceStep(
        agent=agent,
        action_type=DIAMOND_ACTION[agent],
        finding=finding,
        flagged=flagged,
        acted_account_id=alert.account_id,
        **extra,
    )


def _write(agent: str, step: TraceStep) -> dict:
    return {"steps": {agent: step}, "executed": [agent]}


def intake_node(state: DiamondState) -> dict:
    alert = state["alert"]
    return _write(TRIAGE_AGENT, _step(TRIAGE_AGENT, f"routed {alert.alert_id} to 3 branches", False, alert))


def screening_node(state: DiamondState) -> dict:
    alert = state["alert"]
    sanctions = recompute_sanctions(alert)
    return _write(SANCTIONS_AGENT, _step(
        SANCTIONS_AGENT, sanctions.finding, sanctions.flagged, alert, finding_version=2,
    ))


def transaction_node(state: DiamondState) -> dict:
    alert = state["alert"]
    truth = recompute_structuring(alert)
    used = (
        _misread(truth.qualifying_amounts)
        if state["faulted_branch"] == TRANSACTION_AGENT else truth.qualifying_amounts
    )
    disposition = _disposition_for(used)
    flagged = disposition == "FILE_SAR"
    return _write(TRANSACTION_AGENT, _step(
        TRANSACTION_AGENT, "structuring" if flagged else "not-structuring", flagged, alert,
        used_amounts=used, used_aggregate=sum(used), disposition=disposition, finding_version=2,
    ))


def kyc_node(state: DiamondState) -> dict:
    alert = state["alert"]
    kyc = recompute_kyc(alert)
    return _write(KYC_AGENT, _step(KYC_AGENT, kyc.finding, kyc.flagged, alert, finding_version=2))


def llm_decision_node(state: DiamondState) -> dict:
    """Fan-in by a real LLM: record only the branch findings the agent actually consumed."""

    from aml.llm_decision_agent import run_llm_decision

    alert = state["alert"]
    branches = {agent: state["steps"][agent] for agent in DIAMOND_BRANCHES}
    decision = run_llm_decision(alert, branches, state.get("llm_config"))
    consumed = list(decision.consumed)
    lost = ", ".join(decision.not_consumed)
    finding = (
        f"{decision.disposition} (LLM decision agent; used {len(consumed)}/3 branch reports"
        + (f"; not in its context: {lost}" if lost else "")
        + ")"
    )
    transactions = branches[TRANSACTION_AGENT]
    step = _step(
        DISPOSITION_AGENT, finding, decision.disposition != "CLEAR", alert,
        used_amounts=transactions.used_amounts, used_aggregate=transactions.used_aggregate,
        disposition=decision.disposition,
        received_amounts_by_parent={agent: branches[agent].used_amounts for agent in consumed},
        received_findings_by_parent=dict(decision.consumed),
        finding_version=3,
        received_versions_by_parent={agent: branches[agent].finding_version for agent in consumed},
    )
    return {**_write(DISPOSITION_AGENT, step), "llm_decision": decision}


def decision_node(state: DiamondState) -> dict:
    """Fan-in: merge the three branch outputs as this agent's case-state view holds them."""

    if state.get("decision_mode", DETERMINISTIC_DECISION) == LLM_CONTEXT_LOSS_DECISION:
        return llm_decision_node(state)
    alert = state["alert"]
    branches = [state["steps"][agent] for agent in DIAMOND_BRANCHES]
    view = {
        step.agent: {"finding": step.finding, "version": step.finding_version, "flagged": step.flagged}
        for step in branches
    }
    stale = state["stale_screening_view"]
    if stale:
        view[SANCTIONS_AGENT] = dict(PRIOR_SCREENING_STATE)
    merged = merge_disposition(*(view[agent]["flagged"] for agent in DIAMOND_BRANCHES))
    clean = sum(not view[agent]["flagged"] for agent in DIAMOND_BRANCHES)
    finding = f"{merged} ({clean}/3 branches clean)" + (" - stale screening view" if stale else "")
    transactions = state["steps"][TRANSACTION_AGENT]
    return _write(DISPOSITION_AGENT, _step(
        DISPOSITION_AGENT, finding, merged != "CLEAR", alert,
        used_amounts=transactions.used_amounts, used_aggregate=transactions.used_aggregate,
        disposition=merged,
        received_amounts_by_parent={step.agent: step.used_amounts for step in branches},
        received_findings_by_parent={agent: view[agent]["finding"] for agent in DIAMOND_BRANCHES},
        finding_version=3,
        received_versions_by_parent={agent: view[agent]["version"] for agent in DIAMOND_BRANCHES},
    ))


NODES = {
    TRIAGE_AGENT: intake_node,
    SANCTIONS_AGENT: screening_node,
    TRANSACTION_AGENT: transaction_node,
    KYC_AGENT: kyc_node,
    DISPOSITION_AGENT: decision_node,
}


@lru_cache(maxsize=1)
def build_diamond_graph():
    """Compile the diamond StateGraph: intake -> (3 parallel branches) -> decision."""

    builder = StateGraph(DiamondState)
    for agent, node in NODES.items():
        builder.add_node(agent, node)
    builder.add_edge(START, TRIAGE_AGENT)
    for branch in DIAMOND_BRANCHES:
        builder.add_edge(TRIAGE_AGENT, branch)       # fan-out
    builder.add_edge(list(DIAMOND_BRANCHES), DISPOSITION_AGENT)  # fan-in: waits for all three
    builder.add_edge(DISPOSITION_AGENT, END)
    return builder.compile()


def run_diamond_graph(
    alert: Alert,
    faulted_branch: str | None = None,
    stale_screening_view: bool = False,
    decision_mode: str = DETERMINISTIC_DECISION,
    llm_config: Any = None,
) -> DiamondState:
    """Execute the compiled graph for one alert and return the final LangGraph state."""

    if faulted_branch == "C":
        faulted_branch = TRANSACTION_AGENT
    if faulted_branch not in (None, TRANSACTION_AGENT):
        raise ValueError(f"unsupported faulted_branch: {faulted_branch!r}")
    if decision_mode not in (DETERMINISTIC_DECISION, LLM_CONTEXT_LOSS_DECISION):
        raise ValueError(f"unsupported decision_mode: {decision_mode!r}")
    return build_diamond_graph().invoke({
        "alert": alert,
        "faulted_branch": faulted_branch,
        "stale_screening_view": stale_screening_view,
        "steps": {},
        "executed": [],
        "decision_mode": decision_mode,
        "llm_config": llm_config,
        "llm_decision": None,
    })


def _normalize(alert: Alert, state: DiamondState) -> tuple[GraphStep, ...]:
    return adapt_trace(
        AML_SYSTEM_DESCRIPTION,
        PipelineRunTrace(
            records=alert,
            steps=tuple(state["steps"][agent] for agent in DIAMOND_NODE_ID),
            resource_id=f"aml-alert:{alert.alert_id}",
        ),
    ).steps


def build_langgraph_diamond(
    alert: Alert,
    faulted_branch: str | None = None,
    stale_screening_view: bool = False,
) -> tuple[GraphStep, ...]:
    """LangGraph counterpart of ``agent_sim.build_diamond``: same signature, same output."""

    return _normalize(alert, run_diamond_graph(alert, faulted_branch, stale_screening_view))


def build_llm_context_loss_diamond(alert: Alert, llm_config: Any = None) -> tuple[tuple[GraphStep, ...], Any]:
    """Run the diamond with deterministic branches and a real-LLM decision agent.

    Returns the normalized steps (same shape as ``build_langgraph_diamond``) and the
    ``LLMDecision`` describing what the agent saw vs. decided. Nothing is planted:
    the screening branch emits its current finding; whether the decision agent
    consumed it depends only on the LLM's context window.
    """

    state = run_diamond_graph(alert, decision_mode=LLM_CONTEXT_LOSS_DECISION, llm_config=llm_config)
    return _normalize(alert, state), state["llm_decision"]


def build_langgraph_outcome(
    alert: Alert,
    faulted_branch: str | None = None,
    stale_screening_view: bool = False,
) -> GraphOutcome:
    return GraphOutcome(
        alert=alert,
        steps=build_langgraph_diamond(alert, faulted_branch, stale_screening_view),
        resource_id=f"aml-alert:{alert.alert_id}",
    )


__all__ = [
    "DETERMINISTIC_DECISION",
    "LLM_CONTEXT_LOSS_DECISION",
    "DiamondState",
    "build_llm_context_loss_diamond",
    "PRIOR_SCREENING_STATE",
    "build_diamond_graph",
    "build_langgraph_diamond",
    "build_langgraph_outcome",
    "run_diamond_graph",
]
