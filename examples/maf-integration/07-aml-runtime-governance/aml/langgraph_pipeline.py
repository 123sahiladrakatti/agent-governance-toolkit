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
from typing import Annotated, Optional, TypedDict

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


class DiamondState(TypedDict):
    """Shared LangGraph state for one alert's run."""

    alert: Alert
    faulted_branch: Optional[str]
    stale_screening_view: bool
    # Each node writes its own entry; the reducer merges the parallel branch writes.
    steps: Annotated[dict[str, TraceStep], operator.or_]
    # Order in which LangGraph executed the nodes (parallel branches share a superstep).
    executed: Annotated[list[str], operator.add]


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


def decision_node(state: DiamondState) -> dict:
    """Fan-in: merge the three branch outputs as this agent's case-state view holds them."""

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
) -> DiamondState:
    """Execute the compiled graph for one alert and return the final LangGraph state."""

    if faulted_branch == "C":
        faulted_branch = TRANSACTION_AGENT
    if faulted_branch not in (None, TRANSACTION_AGENT):
        raise ValueError(f"unsupported faulted_branch: {faulted_branch!r}")
    return build_diamond_graph().invoke({
        "alert": alert,
        "faulted_branch": faulted_branch,
        "stale_screening_view": stale_screening_view,
        "steps": {},
        "executed": [],
    })


def build_langgraph_diamond(
    alert: Alert,
    faulted_branch: str | None = None,
    stale_screening_view: bool = False,
) -> tuple[GraphStep, ...]:
    """LangGraph counterpart of ``agent_sim.build_diamond``: same signature, same output."""

    state = run_diamond_graph(alert, faulted_branch, stale_screening_view)
    return adapt_trace(
        AML_SYSTEM_DESCRIPTION,
        PipelineRunTrace(
            records=alert,
            steps=tuple(state["steps"][agent] for agent in DIAMOND_NODE_ID),
            resource_id=f"aml-alert:{alert.alert_id}",
        ),
    ).steps


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
    "DiamondState",
    "PRIOR_SCREENING_STATE",
    "build_diamond_graph",
    "build_langgraph_diamond",
    "build_langgraph_outcome",
    "run_diamond_graph",
]
