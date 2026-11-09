"""What a governed run shows: the event stream, WHERE & WHY, and the verdict.

Pure data (no Streamlit), shared by the single-run view in ``app.py`` and the
per-case PDF report, so both present the same governance detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agt_resolver import GraphReview
from domain import GraphStep
from graph_integrity import GraphIntegrityResult

FW, PLUG = "fw", "plug"
VERDICT_LABELS = {"allow": "CONTINUE", "require_approval": "HUMAN REVIEW", "deny": "BLOCK"}
VERDICT_CSS = {"allow": "continue", "require_approval": "review", "deny": "block"}

# Which layer catches which real-LLM failure: (key, failure, deterministic trace check, NeMo content rail).
CONTRAST_ROWS = (
    ("context-loss", "Context loss: the decision agent loses a branch finding",
     "Catches it: what the agent consumed differs from what the parent emitted",
     "Misses it: the output has no personal data or injection to flag"),
    ("pii-leak", "PII leak: the allergy agent over-shares in its note",
     "Misses it: provenance and versions are consistent",
     "Catches it: personal data in the agent's own output"),
)

# (layer, phase, status, css, detail) -- layer is FW (framework) or PLUG (domain plug-in).
Event = tuple[str, str, str, str, str]


@dataclass(frozen=True)
class GovernanceView:
    steps: dict[str, GraphStep]
    decision: GraphStep                 # the final (fan-in) agent's step
    consumer: GraphStep                 # the agent integrity attributes the conflict to
    producer: GraphStep | None          # the parent whose finding was not reconciled
    conflict_parent: str | None
    consumed_finding: Any
    consumed_version: Any
    action: str
    action_label: str
    verdict_css: str
    origin: str
    events: list[Event]


def build_governance_view(
    domain: str,
    review: GraphReview,
    integrity: GraphIntegrityResult,
    final_agent: str,
    run_engine: str,
    run_engine_fallback: bool,
    llm_decision: Any = None,
    llm_allergy: Any = None,
) -> GovernanceView:
    steps = {step.agent: step for step in review.outcome.steps}
    decision = steps[final_agent]
    consumer = steps.get(integrity.origin_agent, decision)
    conflict_parents = (
        *integrity.stale_parents,
        *integrity.altered_parents,
        *integrity.future_version_parents,
        *integrity.missing_parents,
    )
    conflict_parent = next((parent for parent in conflict_parents if parent in steps), None)
    producer = steps.get(conflict_parent) if conflict_parent else None
    consumed_finding = (
        consumer.received_findings_by_parent.get(conflict_parent, "missing")
        if conflict_parent else "not applicable"
    )
    consumed_version = (
        consumer.received_versions_by_parent.get(conflict_parent, "missing")
        if conflict_parent else "not applicable"
    )
    action = str(review.policy_decision.get("action", "deny"))
    action_label = VERDICT_LABELS.get(action, action.upper())
    verdict_css = VERDICT_CSS.get(action, "block")

    # Execution events first, in pipeline order: the client agents run, a real-LLM branch
    # agent (allergy safety) runs with the branches, and the real-LLM decision agent runs
    # last because it merges the branch findings. The framework's governance evaluation follows.
    events: list[Event] = [
        (PLUG, f"Client agents run case ({domain})", "RAN", "pass",
         f"Simulated {domain} agents executed the live case through the client pipeline's topology "
         f"(engine: {run_engine}{', FALLBACK' if run_engine_fallback else ''})."),
        *(
            [(PLUG, "Allergy agent (real LLM)", "WROTE NOTE", "pass",
              f"{llm_allergy.model} returned status {llm_allergy.status} and a handoff note "
              f"({'redaction instruction in prompt' if llm_allergy.redacted else 'permissive prompt'}); "
              f"record identifiers found verbatim in the note: "
              f"{', '.join(llm_allergy.identifiers_in_note) or 'none'}.")]
            if llm_allergy is not None else []
        ),
        *(
            [(PLUG, "Decision agent (real LLM)",
              "LOST CONTEXT" if llm_decision.not_consumed else "DECIDED",
              "review" if llm_decision.not_consumed else "pass",
              f"{llm_decision.model} with num_ctx={llm_decision.num_ctx} kept "
              f"{llm_decision.kept_tokens} prompt tokens and decided {llm_decision.disposition}; "
              f"not in its context: {', '.join(llm_decision.not_consumed) or 'none'}.")]
            if llm_decision is not None else []
        ),
        (FW, "JSON ingestion", "PASS", "pass",
         f"Definition + live run validated into a PipelineTrace of {len(review.outcome.steps)} agent steps; "
         "permissions and policies taken from the client pipeline definition."),
    ]
    for handoff in review.handoffs:
        events.append((
            FW, f"Authorization · {handoff.sender} → {handoff.recipient}",
            "AUTHORIZED" if handoff.passed else "DENIED",
            "pass" if handoff.passed else "fail",
            handoff.reason or handoff.rule or "Handoff policy evaluated.",
        ))
    events.append((
        FW, "Generic trace integrity", "PASS" if integrity.passed else "STALE / CONFLICT",
        "pass" if integrity.passed else "review", integrity.reason,
    ))
    for check in review.content_checks:
        if not check["passed"]:
            status, css = "FLAG", "review"
        elif check["rail"] == "unavailable":
            status, css = "UNAVAILABLE", "review"
        else:
            status, css = "PASS", "pass"
        events.append((FW, f"NeMo content · {check['agent']}", status, css, str(check["reason"])))
    events.append((
        PLUG, f"Correctness plug-in ({domain})",
        "PASS" if review.attribution.passed else "FLAG",
        "pass" if review.attribution.passed else "review",
        review.attribution.reason,
    ))
    events.append((
        FW, "Action Governance gate", action_label,
        "pass" if action == "allow" else ("review" if action == "require_approval" else "fail"),
        str(review.policy_decision.get("reason", "Final policy evaluated.")),
    ))

    origin = (
        f"{review.attribution.origin_node or '-'} · {review.attribution.origin_agent}"
        if review.attribution.origin_agent else "No failing agent attributed"
    )
    return GovernanceView(
        steps=steps, decision=decision, consumer=consumer, producer=producer,
        conflict_parent=conflict_parent, consumed_finding=consumed_finding, consumed_version=consumed_version,
        action=action, action_label=action_label, verdict_css=verdict_css, origin=origin, events=events,
    )


__all__ = ["CONTRAST_ROWS", "FW", "PLUG", "Event", "GovernanceView", "VERDICT_LABELS", "build_governance_view"]
