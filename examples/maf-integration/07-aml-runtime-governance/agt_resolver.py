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
from typing import Any, Callable, Optional

from aml.correctness import AmlCorrectnessComponent
from aml.system import AML_SYSTEM_DESCRIPTION
from correctness import CorrectnessComponent
from domain import (
    Alert,
    BranchCheck,
    GraphAttribution,
    GraphOutcome,
    GraphStep,
)
from governance import ToolkitGovernanceAdapter
from graph_integrity import check_graph_integrity
from nemo_guardrails import NemoVerdict, nemo_check, summarize_nemo_checks
from system_description import (
    SystemDescription,
    decode_graph_context,
    encode_graph_context,
)


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


GRAPH_SCHEMA = AML_SYSTEM_DESCRIPTION.schema
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
        {"name": "require-review-for-unreconciled-conflict", "description": "A stale or contradictory agent finding requires human reconciliation.",
         "stage": "post_tool", "condition": "governance.status == 'conflict'", "action": "require_approval", "priority": 180},
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
def graph_context(
    outcome: GraphOutcome,
    system_description: SystemDescription = AML_SYSTEM_DESCRIPTION,
) -> dict[str, Any]:
    """Encode a normalized outcome for AGT context/audit transport."""

    return encode_graph_context(outcome, system_description)


def _decode(
    ctx: dict[str, Any],
    system_description: SystemDescription = AML_SYSTEM_DESCRIPTION,
) -> tuple[object, tuple[GraphStep, ...]]:
    outcome = decode_graph_context(ctx, system_description)
    return outcome.alert, outcome.steps


# ---------------------------------------------------------------------------
# Generic graph integrity + optional customer correctness
# ---------------------------------------------------------------------------
_DEFAULT_CORRECTNESS = object()


def evaluate_graph(
    outcome: GraphOutcome,
    correctness_component: CorrectnessComponent | None,
    system_description: SystemDescription = AML_SYSTEM_DESCRIPTION,
) -> GraphAttribution:
    """Combine always-on graph integrity with optional domain correctness."""

    integrity = check_graph_integrity(outcome)
    by_agent = {step.agent: step for step in outcome.steps}
    merge = by_agent[system_description.final_agent]
    correctness = (
        correctness_component.check(outcome, outcome.alert)
        if correctness_component is not None else None
    )

    actual = str(
        correctness.actual
        if correctness is not None and correctness.actual is not None
        else merge.disposition or merge.finding
    )
    expected = str(
        correctness.expected
        if correctness is not None and correctness.expected is not None
        else actual
    )

    if not integrity.passed:
        domain_checks = tuple(
            check for check in (correctness.checks if correctness is not None else ())
            if getattr(check, "agent", None) != merge.agent
        )
        merge_check = BranchCheck(
            merge.agent,
            merge.node_id,
            actual,
            expected,
            False,
            integrity.reason,
        )
        return GraphAttribution(
            False,
            integrity.category,
            integrity.origin_agent,
            integrity.origin_node,
            integrity.related_agents,
            integrity.reason,
            (*domain_checks, merge_check),
            actual,
            expected,
        )

    if correctness is None:
        merge_check = BranchCheck(
            merge.agent,
            merge.node_id,
            actual,
            expected,
            True,
            "Domain correctness was not configured; generic graph integrity passed.",
        )
        return GraphAttribution(
            True,
            None,
            None,
            None,
            (),
            "Generic graph integrity passed; domain correctness was not configured.",
            (merge_check,),
            actual,
            expected,
        )

    return GraphAttribution(
        correctness.passed,
        correctness.category,
        correctness.origin_agent,
        correctness.origin_node,
        correctness.related_agents,
        correctness.reason,
        tuple(correctness.checks),
        actual,
        expected,
    )


def attribute_graph(alert: Alert, steps: tuple[GraphStep, ...]) -> GraphAttribution:
    """Compatibility entry point for the AML reference implementation."""

    return evaluate_graph(
        GraphOutcome(alert=alert, steps=steps),
        AmlCorrectnessComponent(),
        AML_SYSTEM_DESCRIPTION,
    )


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------
class BehavioralAuthorityResolver(_ResolverBase):
    """AGT hook combining graph integrity and optional domain correctness."""

    def __init__(
        self,
        correctness_component: CorrectnessComponent | None | object = _DEFAULT_CORRECTNESS,
        system_description: SystemDescription = AML_SYSTEM_DESCRIPTION,
    ) -> None:
        if correctness_component is _DEFAULT_CORRECTNESS:
            self.correctness_component = (
                AmlCorrectnessComponent()
                if system_description.schema == AML_SYSTEM_DESCRIPTION.schema
                else None
            )
        else:
            self.correctness_component = correctness_component
        self.system_description = system_description
        self.last_attribution: GraphAttribution | None = None
        self.last_decision: AuthorityDecision | None = None
        self.calls = 0

    def resolve(self, request) -> AuthorityDecision:
        self.calls += 1
        request_context = getattr(request, "context", None) or {}
        ctx = request_context.get("governance_graph") or request_context.get("aml_graph")
        if not isinstance(ctx, dict) or ctx.get("schema") != self.system_description.schema:
            self.last_attribution = None
            self.last_decision = AuthorityDecision(decision="deny", trust_tier="record-relative",
                                                   narrowing_reason="no valid graph/records in context (fail closed)")
            return self.last_decision
        outcome = decode_graph_context(ctx, self.system_description)
        verdict = evaluate_graph(outcome, self.correctness_component, self.system_description)
        self.last_attribution = verdict
        self.last_decision = AuthorityDecision(
            decision="allow" if verdict.passed else "deny",
            effective_scope=list(getattr(getattr(request, "delegation", None), "delegated_capabilities", []) or []),
            narrowing_reason=verdict.reason,
            trust_tier="record-relative" if self.correctness_component is not None else "graph-integrity",
            matched_invariants=[f"{c.node_id}:{c.agent}:{'ok' if c.matches else 'diverged'}" for c in verdict.branch_checks],
        )
        return self.last_decision


def register(
    engine,
    correctness_component: CorrectnessComponent | None | object = _DEFAULT_CORRECTNESS,
    system_description: SystemDescription = AML_SYSTEM_DESCRIPTION,
) -> BehavioralAuthorityResolver:
    """Install our resolver into an AGT PolicyEngine."""
    resolver = BehavioralAuthorityResolver(correctness_component, system_description)
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

    def __init__(
        self,
        content_checker: Callable[[str], NemoVerdict] = nemo_check,
        correctness_component: CorrectnessComponent | None | object = _DEFAULT_CORRECTNESS,
        graph_policy: str = GRAPH_POLICY,
        action_policy: str = ACTION_GOVERNANCE_POLICY,
        system_description: SystemDescription = AML_SYSTEM_DESCRIPTION,
    ) -> None:
        self.system_description = system_description
        self.toolkit = ToolkitGovernanceAdapter(graph_policy)
        self.action_toolkit = ToolkitGovernanceAdapter(action_policy)
        graph_policy_data = json.loads(graph_policy)
        self.policy_version = (
            f"{graph_policy_data.get('name', 'graph-governance')}@"
            f"{graph_policy_data.get('version', 'unknown')}"
        )
        self.engine_class = type(self.toolkit.engine).__module__ + "." + type(self.toolkit.engine).__name__ if self.toolkit.engine is not None else None
        self.resolver_engine_live = AGT_AVAILABLE and self.toolkit.engine is not None
        self.action_engine_live = AGT_AVAILABLE and self.action_toolkit.engine is not None
        self.agt_live = self.resolver_engine_live and self.action_engine_live
        self.content_checker = content_checker
        if self.resolver_engine_live:
            self.resolver = register(
                self.toolkit.engine,
                correctness_component,
                system_description,
            )
        else:
            self.resolver = BehavioralAuthorityResolver(
                correctness_component,
                system_description,
            )

    def _audit(self, agent: str, action: str, resource: str, context: dict, decision) -> dict[str, Any]:
        row = {"agent": agent, "action": action, "resource": resource, "decision": decision.action,
               "reason": getattr(decision, "reason", ""), "entry_hash": None}
        if self.resolver_engine_live:
            entry = self.toolkit.audit.log(
                event_type="policy_evaluation", agent_did=agent, action=action, resource=resource, data=context,
                outcome="success" if decision.allowed else "denied",
                policy_decision=decision.action, policy_version=self.policy_version,
            )
            row["entry_hash"] = getattr(entry, "entry_hash", None)
        return row

    def review(self, outcome: GraphOutcome) -> GraphReview:
        record_id = getattr(outcome.alert, "alert_id", "unknown")
        resource = outcome.resource_id or f"{self.system_description.resource_prefix}:{record_id}"
        by_agent = {s.agent: s for s in outcome.steps}
        final_agent = self.system_description.final_agent
        final_step = by_agent[final_agent]
        audit: list[dict[str, Any]] = []

        # Run each node's deterministic incoming-edge gates before checking its output.
        handoffs = []
        content_checks: list[dict[str, Any]] = []
        for step in outcome.steps:
            for sender in step.received_from:
                recipient = step.agent
                declared = self.system_description.agent(recipient)
                ok = (
                    sender in self.system_description.parents_for(recipient)
                    and step.action_type in declared.allowed_actions
                )
                ctx = {"handoff": {"sender": sender, "recipient": recipient,
                                   "status": "authorized" if ok else "unauthorized"},
                       "action": {"type": self.system_description.handoff_action, "authorized": ok}}
                if self.resolver_engine_live:
                    d = self.toolkit.engine.evaluate(recipient, ctx, stage="pre_tool")
                else:
                    fallback_rule = self.toolkit.rule_names.get(
                        "handoff.status == 'authorized'" if ok else "handoff.status != 'authorized'",
                        "allow-authorized-handoff" if ok else "deny-unauthorized-handoff",
                    )
                    d = SimpleNamespace(allowed=ok, action="allow" if ok else "deny",
                                        matched_rule=fallback_rule,
                                        reason="fallback: static handoff rule")
                audit.append(self._audit(recipient, self.system_description.handoff_action, resource, ctx, d))
                handoffs.append(HandoffCheck(sender, recipient, bool(d.allowed), d.action,
                                             getattr(d, "matched_rule", None), d.reason or ""))

            incoming = [check for check in handoffs if check.recipient == step.agent]
            authorized = (
                step.action_type in self.system_description.agent(step.agent).allowed_actions
                and all(check.passed for check in incoming)
            )
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
        encoded_graph = graph_context(outcome, self.system_description)
        final_capabilities = sorted(self.system_description.agent(final_agent).allowed_actions)
        resolver_context = {
            "action": {"type": self.system_description.resolver_action},
            "tool_name": self.system_description.final_action,
            "resource": resource,
            "capabilities": final_capabilities,
            "nemo": nemo_signal,
            "governance_graph": encoded_graph,
            # Compatibility key for the shipped AML fail-closed policy.
            "aml_graph": encoded_graph,
        }
        if self.resolver_engine_live:
            resolver_decision = self.toolkit.engine.evaluate(final_agent, resolver_context, stage="post_tool")
            attribution = self.resolver.last_attribution
            if attribution is None:   # static fail-closed rule fired before the resolver
                attribution = evaluate_graph(
                    outcome,
                    self.resolver.correctness_component,
                    self.system_description,
                )
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

        audit.append(self._audit(final_agent, self.system_description.resolver_action, resource,
                                 resolver_context, resolver_decision))
        action_authorized = all(handoff.passed for handoff in handoffs) and all(
            step.action_type in self.system_description.agent(step.agent).allowed_actions
            for step in outcome.steps
        )
        action_context = {
            "action": {
                "type": self.system_description.final_action,
                "authorized": action_authorized,
                "status": "authorized" if action_authorized else "unauthorized",
            },
            "governance": {
                "passed": attribution.passed,
                "status": (
                    "passed" if attribution.passed else
                    "conflict" if attribution.category == "unreconciled-conflict" else "failed"
                ),
                "category": attribution.category or "",
            },
            "nemo": nemo_signal,
            "resource": resource,
            "capabilities": final_capabilities,
        }
        d = self.action_toolkit.evaluate(
            agent=final_agent,
            action=self.system_description.final_action,
            context=action_context,
            stage="post_tool",
            resource=resource,
        )
        audit.append({"agent": final_agent, "action": "action_governance_decision",
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
        if (
            d.action == "require_approval"
            or final_step.disposition in self.system_description.escalation_outcomes
        ):
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
