"""Behavioral governance for the AML diamond, packaged as AGT's AuthorityResolver.

AGT's PolicyEngine ships an authority-resolver hook (``set_authority_resolver``)
with no behavioral logic in it: ``DefaultAuthorityResolver`` allows everything.
This module fills that slot. The verdict on a diamond's merged disposition is
produced by ``PolicyEngine.evaluate``; the logic inside it is ours.

How the decision travels through AGT (agentmesh.governance.policy):
  1. Static JSON rules run first. GRAPH_POLICY has pre_tool rules for per-hop
     handoff authorization, and one post_tool rule that denies when no graph is
     in the context (fail closed). Otherwise no post_tool static rule matches.
  2. With no static match, AGT builds an ``AuthorityRequest`` (passing our
     context dict through untouched) and calls ``resolver.resolve(request)``.
  3. A resolver ``deny`` becomes a PolicyDecision whose reason AGT prefixes with
     "Authority resolver denied: ...". A resolver ``allow`` falls through to the
     policy default (allow), so the resolver is the gate for the merge commit.

If AGT is not importable the module still works: the same resolver class runs
against a stand-in AuthorityDecision and the result is flagged ``agt_live=False``.
Deterministic and LLM-free.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from agent_sim import (
    DIAMOND_BRANCHES,
    DIAMOND_PARENTS,
    DISPOSITION_AGENT,
    KYC_AGENT,
    SANCTIONS_AGENT,
    TRANSACTION_AGENT,
)
from domain import (
    Alert,
    BranchCheck,
    ChainStep,
    CustomerProfile,
    GraphAttribution,
    GraphOutcome,
    GraphStep,
    Transaction,
    merge_disposition,
    recompute_kyc,
    recompute_sanctions,
    recompute_transactions,
)
from governance import ChainGovernance, ToolkitGovernanceAdapter
from nemo_guardrails import NemoVerdict, nemo_check, summarize_nemo_checks


def _import_agt():
    """Import AGT's resolver types, falling back to the in-repo source tree."""
    try:
        from agentmesh.governance import AuthorityDecision, AuthorityResolver
        return AuthorityDecision, AuthorityResolver
    except ImportError:
        local_source = Path(__file__).resolve().parents[3] / "agent-governance-python" / "agent-mesh" / "src"
        if local_source.is_dir():
            sys.path.insert(0, str(local_source))
            try:
                from agentmesh.governance import AuthorityDecision, AuthorityResolver
                return AuthorityDecision, AuthorityResolver
            except ImportError:
                pass
    return None, None


_AGTDecision, _AGTResolver = _import_agt()
AGT_AVAILABLE = _AGTResolver is not None

if AGT_AVAILABLE:
    AuthorityDecision = _AGTDecision
    _ResolverBase = _AGTResolver
else:
    @dataclass
    class AuthorityDecision:  # stand-in with the same fields as AGT's
        decision: str
        effective_scope: list[str] = field(default_factory=list)
        effective_spend_limit: Optional[float] = None
        narrowing_reason: Optional[str] = None
        trust_tier: str = "unknown"
        matched_invariants: list[str] = field(default_factory=list)
        timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    _ResolverBase = object


GRAPH_SCHEMA = "aml-diamond/v1"
POLICY_VERSION = "aml-graph-governance@1.0"

GRAPH_POLICY = json.dumps({
    "apiVersion": "governance.toolkit/v1",
    "version": "1.0",
    "name": "aml-graph-governance",
    "description": ("Per-hop handoff authorization (static rules) and merge-commit governance "
                    "delegated to the registered BehavioralAuthorityResolver."),
    "agents": ["*"],
    "scope": "global",
    "default_action": "allow",
    "rules": [
        {"name": "allow-authorized-handoff", "description": "Handoff is on a declared edge and the recipient's action is delegated.",
         "stage": "pre_tool", "condition": "handoff.status == 'authorized'", "action": "allow", "priority": 100},
        {"name": "deny-unauthorized-handoff", "description": "Any handoff not explicitly authorized is denied.",
         "stage": "pre_tool", "condition": "handoff.status != 'authorized'", "action": "deny", "priority": 100},
        {"name": "deny-commit-without-graph", "description": "Fail closed: a merge commit must carry its graph and records.",
         "stage": "post_tool", "condition": f"aml_graph.schema != '{GRAPH_SCHEMA}'", "action": "deny", "priority": 100},
    ],
})

ACTION_GOVERNANCE_POLICY = json.dumps({
    "apiVersion": "governance.toolkit/v1",
    "version": "1.0",
    "name": "aml-action-governance",
    "description": "Final authorization, correctness, and NeMo content decision.",
    "agents": ["*"],
    "scope": "global",
    "default_action": "deny",
    "rules": [
        {"name": "deny-unauthorized-action", "description": "Final action is not authorized.",
         "stage": "post_tool", "condition": "action.status == 'unauthorized'", "action": "deny", "priority": 200},
        {"name": "deny-incorrect-behavioral-result", "description": "Record-relative behavioral governance failed.",
         "stage": "post_tool", "condition": "governance.status == 'failed'", "action": "deny", "priority": 150},
        {"name": "require-review-for-nemo-flag", "description": "NeMo content flags require human approval.",
         "stage": "post_tool", "condition": "nemo.status == 'flagged'", "action": "require_approval", "priority": 100},
        {"name": "allow-clean-action", "description": "Authorization, correctness, and NeMo checks passed.",
         "stage": "post_tool", "condition": "action.status == 'authorized' and governance.status == 'passed' and nemo.status == 'passed'",
         "action": "allow", "priority": 50},
        {"name": "allow-with-nemo-unavailable", "description": "Allow with an explicit unavailable NeMo signal when deterministic checks pass.",
         "stage": "post_tool", "condition": "action.status == 'authorized' and governance.status == 'passed' and nemo.status == 'unavailable'",
         "action": "allow", "priority": 40},
    ],
})


# ---------------------------------------------------------------------------
# JSON-safe context (AGT's AuditLog hashes the context with json.dumps)
# ---------------------------------------------------------------------------
def graph_context(outcome: GraphOutcome) -> dict[str, Any]:
    alert = outcome.alert
    return {
        "schema": GRAPH_SCHEMA,
        "alert": {
            "alert_id": alert.alert_id, "account_id": alert.account_id, "monitoring_rule": alert.monitoring_rule,
            "customer": {"account_id": alert.customer.account_id, "name": alert.customer.name,
                         "stated_occupation": alert.customer.stated_occupation,
                         "expected_monthly_cash_volume": alert.customer.expected_monthly_cash_volume},
            "transactions": [{"id": t.id, "account_id": t.account_id, "amount": t.amount, "type": t.type,
                              "timestamp": t.timestamp.isoformat(), "branch": t.branch} for t in alert.transactions],
        },
        "steps": [{
            "agent": s.agent, "node_id": s.node_id, "action_type": s.action_type,
            "allowed_actions": sorted(s.allowed_actions), "received_from": list(s.received_from),
            "finding": s.finding, "flagged": s.flagged, "used_amounts": list(s.used_amounts),
            "used_aggregate": s.used_aggregate, "disposition": s.disposition, "acted_account_id": s.acted_account_id,
            "received_amounts_by_parent": {k: list(v) for k, v in s.received_amounts_by_parent.items()},
            "received_findings_by_parent": dict(s.received_findings_by_parent),
        } for s in outcome.steps],
    }


def _decode(ctx: dict[str, Any]) -> tuple[Alert, tuple[GraphStep, ...]]:
    a = ctx["alert"]
    c = a["customer"]
    alert = Alert(
        a["alert_id"], a["account_id"],
        tuple(Transaction(t["id"], t["account_id"], float(t["amount"]), t["type"],
                          datetime.fromisoformat(t["timestamp"]), t["branch"]) for t in a["transactions"]),
        CustomerProfile(c["account_id"], c["name"], c["stated_occupation"], float(c["expected_monthly_cash_volume"])),
        a["monitoring_rule"],
    )
    steps = tuple(GraphStep(
        agent=s["agent"], node_id=s["node_id"], action_type=s["action_type"],
        allowed_actions=frozenset(s["allowed_actions"]), received_from=tuple(s["received_from"]),
        finding=s["finding"], flagged=bool(s["flagged"]), used_amounts=tuple(s["used_amounts"]),
        used_aggregate=float(s["used_aggregate"]), disposition=s["disposition"], acted_account_id=s["acted_account_id"],
        received_amounts_by_parent={k: tuple(v) for k, v in s["received_amounts_by_parent"].items()},
        received_findings_by_parent=dict(s["received_findings_by_parent"]),
    ) for s in ctx["steps"])
    return alert, steps


# ---------------------------------------------------------------------------
# Branch-level attribution (pure; shared by the AGT path and the fallback)
# ---------------------------------------------------------------------------
def _money(v: float) -> str:
    return f"${v:,.0f}"


def _check_transactions(alert: Alert, step: GraphStep) -> BranchCheck:
    """Reuse ChainGovernance's record-relative check by projecting branch C as a
    one-hop chain that reads the records itself (received_from=None)."""
    projected = ChainStep(
        agent=step.agent, step_index=1, used_amounts=step.used_amounts, used_aggregate=step.used_aggregate,
        disposition=step.disposition or "CLEAR", action_type=step.action_type, allowed_actions=step.allowed_actions,
        received_from=None, received_amounts=None, acted_account_id=step.acted_account_id,
    )
    gov = ChainGovernance().check(alert, (projected,))
    truth = recompute_transactions(alert)
    if gov.passed:
        return BranchCheck(step.agent, step.node_id, step.finding, truth.finding, True, truth.detail)
    if gov.category != "transitive-corruption":
        return BranchCheck(step.agent, step.node_id, step.finding, truth.finding, False, gov.reason)
    misread = next(((r, u) for r, u in zip(gov.record_amounts, step.used_amounts) if r != u), None)
    detail = (f"misread {_money(misread[0])} as {_money(misread[1])}" if misread else "used amounts differ from records")
    detail += (f"; used {_money(step.used_aggregate)} total vs {_money(gov.record_aggregate)} in the records "
               f"(records say {truth.finding})")
    return BranchCheck(step.agent, step.node_id, step.finding, truth.finding, False, detail)


def _check_simple(alert: Alert, step: GraphStep, recompute) -> BranchCheck:
    truth = recompute(alert)
    if step.acted_account_id != alert.account_id:
        return BranchCheck(step.agent, step.node_id, step.finding, truth.finding, False,
                           f"acted on {step.acted_account_id}; alert belongs to {alert.account_id}")
    ok = step.finding == truth.finding and step.flagged == truth.flagged
    return BranchCheck(step.agent, step.node_id, step.finding, truth.finding, ok, truth.detail)


_BRANCH_CHECKERS = {
    SANCTIONS_AGENT: lambda alert, s: _check_simple(alert, s, recompute_sanctions),
    TRANSACTION_AGENT: _check_transactions,
    KYC_AGENT: lambda alert, s: _check_simple(alert, s, recompute_kyc),
}
_RECOMPUTE = {SANCTIONS_AGENT: recompute_sanctions, TRANSACTION_AGENT: recompute_transactions, KYC_AGENT: recompute_kyc}


def attribute_graph(alert: Alert, steps: tuple[GraphStep, ...]) -> GraphAttribution:
    """Recompute every branch from the records, then judge the merged disposition."""
    by_agent = {s.agent: s for s in steps}
    merge = by_agent[DISPOSITION_AGENT]
    checks = [_BRANCH_CHECKERS[agent](alert, by_agent[agent]) for agent in DIAMOND_BRANCHES]
    truth = {agent: _RECOMPUTE[agent](alert) for agent in DIAMOND_BRANCHES}
    expected = merge_disposition(*(truth[a].flagged for a in DIAMOND_BRANCHES))
    merged = merge.disposition or "CLEAR"

    # Fan-in provenance: did E receive exactly what each parent reported?
    tampered = [p for p in merge.received_from
                if merge.received_findings_by_parent.get(p) != by_agent[p].finding]
    from_inputs = merge_disposition(*(by_agent[a].flagged for a in DIAMOND_BRANCHES))
    merge_ok = not tampered and merged == from_inputs
    checks.append(BranchCheck(
        merge.agent, merge.node_id, merged, expected, merge_ok and merged == expected,
        ("merged its three inputs faithfully" if merge_ok else
         f"merge diverged from its inputs (inputs imply {from_inputs}; altered: {', '.join(tampered) or 'none'})"),
    ))
    checks_t = tuple(checks)
    diverged = [c for c in checks_t[:-1] if not c.matches]

    if merged == expected:
        note = (f" Branch divergence did not change the outcome: {', '.join(c.agent for c in diverged)}."
                if diverged else "")
        return GraphAttribution(True, None, None, None, (), f"Merged {merged} matches the records.{note}",
                                checks_t, merged, expected)

    if not merge_ok:
        origin = merge
        reason = (f"{merge.agent} (node {merge.node_id}) produced {merged}, but its branch inputs imply "
                  f"{from_inputs}; records require {expected}.")
        masked: tuple[str, ...] = ()
        category = "merge-error"
    else:
        bad = diverged[0] if diverged else checks_t[0]
        origin = by_agent[bad.agent]
        masked = tuple(c.agent for c in checks_t[:-1] if c.matches and not by_agent[c.agent].flagged)
        sib = ", ".join(f"{by_agent[a].node_id} ({a})" for a in masked)
        reason = (f"branch {origin.node_id} ({origin.agent}) {bad.detail}; merged {merged} is wrong "
                  f"(records require {expected})" + (f". Masked by clean siblings {sib}." if sib else "."))
        category = "masked-branch-error"
    return GraphAttribution(False, category, origin.agent, origin.node_id, masked, reason, checks_t, merged, expected)


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------
class BehavioralAuthorityResolver(_ResolverBase):
    """AGT AuthorityResolver whose verdict is record-relative behavioral governance."""

    def __init__(self) -> None:
        self.last_attribution: GraphAttribution | None = None
        self.last_decision: AuthorityDecision | None = None
        self.calls = 0

    def resolve(self, request) -> AuthorityDecision:
        self.calls += 1
        ctx = (getattr(request, "context", None) or {}).get("aml_graph")
        if not isinstance(ctx, dict) or ctx.get("schema") != GRAPH_SCHEMA:
            self.last_attribution = None
            self.last_decision = AuthorityDecision(decision="deny", trust_tier="record-relative",
                                                   narrowing_reason="no diamond graph/records in context (fail closed)")
            return self.last_decision
        alert, steps = _decode(ctx)
        verdict = attribute_graph(alert, steps)
        self.last_attribution = verdict
        self.last_decision = AuthorityDecision(
            decision="allow" if verdict.passed else "deny",
            effective_scope=list(getattr(getattr(request, "delegation", None), "delegated_capabilities", []) or []),
            narrowing_reason=verdict.reason,
            trust_tier="record-relative",
            matched_invariants=[f"{c.node_id}:{c.agent}:{'ok' if c.matches else 'diverged'}" for c in verdict.branch_checks],
        )
        return self.last_decision


def register(engine) -> BehavioralAuthorityResolver:
    """Install our resolver into an AGT PolicyEngine."""
    resolver = BehavioralAuthorityResolver()
    engine.set_authority_resolver(resolver)
    return resolver


# ---------------------------------------------------------------------------
# The single entry point the UI calls
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HandoffCheck:
    sender: str
    recipient: str
    passed: bool
    decision: str          # AGT PolicyDecision.action
    rule: str | None       # AGT matched_rule
    reason: str


@dataclass(frozen=True)
class GraphReview:
    outcome: GraphOutcome
    agt_live: bool
    handoffs: tuple[HandoffCheck, ...]
    attribution: GraphAttribution
    authority_decision: dict[str, Any]    # what our resolver returned
    policy_decision: dict[str, Any]       # what AGT's PolicyEngine returned
    status: str                           # "committed" | "halted" | "escalated"
    audit: tuple[dict[str, Any], ...]
    content_checks: tuple[dict[str, Any], ...] = ()
    nemo_live: bool = False


class GraphGovernor:
    """One AGT PolicyEngine (static handoff rules + our resolver) with an audit log."""

    def __init__(self, content_checker: Callable[[str], NemoVerdict] = nemo_check) -> None:
        self.toolkit = ToolkitGovernanceAdapter(GRAPH_POLICY)
        self.action_toolkit = ToolkitGovernanceAdapter(ACTION_GOVERNANCE_POLICY)
        self.engine_class = type(self.toolkit.engine).__module__ + "." + type(self.toolkit.engine).__name__ if self.toolkit.engine is not None else None
        self.resolver_engine_live = AGT_AVAILABLE and self.toolkit.engine is not None
        self.action_engine_live = AGT_AVAILABLE and self.action_toolkit.engine is not None
        self.agt_live = self.resolver_engine_live and self.action_engine_live
        self.content_checker = content_checker
        if self.resolver_engine_live:
            self.resolver = register(self.toolkit.engine)
        else:
            self.resolver = BehavioralAuthorityResolver()

    def _audit(self, agent: str, action: str, resource: str, context: dict, decision) -> dict[str, Any]:
        row = {"agent": agent, "action": action, "resource": resource, "decision": decision.action,
               "reason": getattr(decision, "reason", ""), "entry_hash": None}
        if self.resolver_engine_live:
            entry = self.toolkit.audit.log(
                event_type="policy_evaluation", agent_did=agent, action=action, resource=resource, data=context,
                outcome="success" if decision.allowed else "denied",
                policy_decision=decision.action, policy_version=POLICY_VERSION,
            )
            row["entry_hash"] = getattr(entry, "entry_hash", None)
        return row

    def review(self, outcome: GraphOutcome) -> GraphReview:
        alert = outcome.alert
        resource = f"aml-alert:{alert.alert_id}"
        by_agent = {s.agent: s for s in outcome.steps}
        audit: list[dict[str, Any]] = []

        # Run each node's deterministic incoming-edge gates before checking its output.
        handoffs = []
        content_checks: list[dict[str, Any]] = []
        for step in outcome.steps:
            for sender in step.received_from:
                recipient = step.agent
                ok = sender in DIAMOND_PARENTS.get(recipient, ()) and step.action_type in step.allowed_actions
                ctx = {"handoff": {"sender": sender, "recipient": recipient,
                                   "status": "authorized" if ok else "unauthorized"},
                       "action": {"type": "aml_graph_handoff", "authorized": ok}}
                if self.resolver_engine_live:
                    d = self.toolkit.engine.evaluate(recipient, ctx, stage="pre_tool")
                else:
                    d = SimpleNamespace(allowed=ok, action="allow" if ok else "deny",
                                        matched_rule="allow-authorized-handoff" if ok else "deny-unauthorized-handoff",
                                        reason="fallback: static handoff rule")
                audit.append(self._audit(recipient, "aml_graph_handoff", resource, ctx, d))
                handoffs.append(HandoffCheck(sender, recipient, bool(d.allowed), d.action,
                                             getattr(d, "matched_rule", None), d.reason or ""))

            incoming = [check for check in handoffs if check.recipient == step.agent]
            authorized = step.action_type in step.allowed_actions and all(check.passed for check in incoming)
            if authorized:
                output = (
                    f"Agent: {step.agent} finding: {step.finding}; disposition: {step.disposition or 'not set'}; "
                    f"evidence amounts: {list(step.used_amounts)}"
                )
                verdict = self.content_checker(output)
            else:
                verdict = {
                    "passed": True,
                    "rail": "skipped",
                    "reason": "The deterministic authorization gate denied this step before execution.",
                }
            content_checks.append({"agent": step.agent, **verdict})
        nemo_signal, nemo_live = summarize_nemo_checks(content_checks)

        # Lane 3a: preserve the existing record-relative resolver as the correctness check.
        self.resolver.last_attribution = None
        self.resolver.last_decision = None
        calls_before = self.resolver.calls
        merge = by_agent[DISPOSITION_AGENT]
        resolver_context = {"action": {"type": "resolve_behavioral_correctness"}, "tool_name": "commit_disposition",
                            "resource": resource, "capabilities": sorted(merge.allowed_actions),
                            "nemo": nemo_signal, "aml_graph": graph_context(outcome)}
        if self.resolver_engine_live:
            resolver_decision = self.toolkit.engine.evaluate(DISPOSITION_AGENT, resolver_context, stage="post_tool")
            attribution = self.resolver.last_attribution
            if attribution is None:   # static fail-closed rule fired before the resolver
                attribution = attribute_graph(alert, outcome.steps)
        else:
            auth = self.resolver.resolve(SimpleNamespace(context=resolver_context, delegation=None))
            denied = auth.decision == "deny"
            resolver_decision = SimpleNamespace(
                allowed=not denied,
                action="deny" if denied else "allow",
                matched_rule=None,
                policy_name=None,
                reason=(f"Authority resolver denied: {auth.narrowing_reason}" if denied
                        else "fallback: resolver allowed"),
            )
            attribution = self.resolver.last_attribution

        audit.append(self._audit(DISPOSITION_AGENT, "resolve_behavioral_correctness", resource,
                                 resolver_context, resolver_decision))
        action_authorized = all(handoff.passed for handoff in handoffs) and all(
            step.action_type in step.allowed_actions for step in outcome.steps
        )
        action_context = {
            "action": {
                "type": "commit_disposition",
                "authorized": action_authorized,
                "status": "authorized" if action_authorized else "unauthorized",
            },
            "governance": {
                "passed": attribution.passed,
                "status": "passed" if attribution.passed else "failed",
            },
            "nemo": nemo_signal,
            "resource": resource,
            "capabilities": sorted(merge.allowed_actions),
        }
        d = self.action_toolkit.evaluate(
            agent=DISPOSITION_AGENT,
            action="commit_disposition",
            context=action_context,
            stage="post_tool",
            resource=resource,
        )
        audit.append({"agent": DISPOSITION_AGENT, "action": "action_governance_decision",
                      "resource": resource, "decision": d.action, "reason": d.reason,
                      "entry_hash": None})

        policy_decision = {"allowed": bool(d.allowed), "action": d.action, "reason": d.reason,
                           "matched_rule": getattr(d, "matched_rule", None),
                           "policy_name": getattr(d, "policy_name", None),
                           "evaluation_ms": getattr(d, "evaluation_ms", None)}
        ad = self.resolver.last_decision
        authority_decision = {"invoked_by_engine": self.resolver.calls > calls_before,
                              "resolver": type(self.resolver).__name__,
                              "resolver_base": f"{_ResolverBase.__module__}.{_ResolverBase.__name__}",
                              "decision": getattr(ad, "decision", None),
                              "narrowing_reason": getattr(ad, "narrowing_reason", None),
                              "trust_tier": getattr(ad, "trust_tier", None),
                              "matched_invariants": list(getattr(ad, "matched_invariants", []) or []),
                              "type": f"{type(ad).__module__}.{type(ad).__name__}" if ad is not None else None}
        if d.action == "require_approval" or merge.disposition == "ESCALATE":
            status = "escalated"
        elif not d.allowed:
            status = "halted"
        else:
            status = "committed"
        return GraphReview(outcome, self.agt_live, tuple(handoffs), attribution, authority_decision,
                           policy_decision, status, tuple(audit), tuple(content_checks), nemo_live)


def govern_diamond(outcome: GraphOutcome, governor: GraphGovernor | None = None) -> GraphReview:
    """Review one diamond. Same GraphReview whether AGT is live or the fallback ran."""
    return (governor or GraphGovernor()).review(outcome)
