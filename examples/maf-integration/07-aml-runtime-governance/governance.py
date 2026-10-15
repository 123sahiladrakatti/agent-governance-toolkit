"""Runtime governance pipeline for the AML agent architecture.

The module models governance in order:
1. Deterministic pre-agent authorization through AGT policy.
2. NeMo content monitoring on each authorized output during agent work.
3. Final AGT Action Governance over authorization, correctness, and content.

The behavioral checker owns AML correctness. NeMo is only a content signal
producer, not an AGT AuthorityResolver or final decision-maker.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Sequence

from domain import (
    Alert,
    ChainAttribution,
    ChainOutcome,
    ChainReview,
    ChainStep,
    SecurityCheck,
    recompute_structuring,
)
from nemo_guardrails import NemoVerdict, nemo_check, summarize_nemo_checks


AML_POLICY = json.dumps({
    "apiVersion": "governance.toolkit/v1",
    "version": "1.0",
    "name": "aml-runtime-governance",
    "description": "Deterministic entry gates and final action decisions for the AML pipeline.",
    "agents": ["*"],
    "scope": "global",
    "default_action": "deny",
    "rules": [
        {"name": "allow-authorized-agent-step", "description": "Only delegated agent actions may run.",
         "stage": "pre_tool", "condition": "action.authorized", "action": "allow", "priority": 100},
        {"name": "deny-unauthorized-final-action", "description": "The final action must be authorized.",
         "stage": "post_tool", "condition": "action.status == 'unauthorized'", "action": "deny", "priority": 200},
        {"name": "require-review-for-unreconciled-conflict", "description": "Unreconciled agent findings require human review.",
         "stage": "post_tool", "condition": "governance.status == 'conflict'", "action": "require_approval", "priority": 180},
        {"name": "deny-incorrect-record-relative-result", "description": "Incorrect results fail deterministic governance.",
         "stage": "post_tool", "condition": "governance.status == 'failed'", "action": "deny", "priority": 150},
        {"name": "require-review-for-nemo-content-flag", "description": "A NeMo content flag requires human review.",
         "stage": "post_tool", "condition": "nemo.status == 'flagged'", "action": "require_approval", "priority": 100},
        {"name": "allow-verified-action", "description": "Authorization, correctness, and NeMo content checks passed.",
         "stage": "post_tool",
         "condition": "action.status == 'authorized' and governance.status == 'passed' and nemo.status == 'passed'",
         "action": "allow", "priority": 50},
        {"name": "allow-verified-action-with-nemo-unavailable", "description": "Deterministic checks pass; NeMo unavailability remains visible.",
         "stage": "post_tool",
         "condition": "action.status == 'authorized' and governance.status == 'passed' and nemo.status == 'unavailable'",
         "action": "allow", "priority": 40},
    ],
})


class ToolkitGovernanceAdapter:
    """Load AGT's PolicyEngine and AuditLog, with a deterministic local fallback."""

    def __init__(self, policy_text: str = AML_POLICY) -> None:
        policy = json.loads(policy_text)
        self.policy_name = str(policy.get("name", "aml-runtime-governance"))
        self.rule_names = {
            str(rule.get("condition")): str(rule.get("name"))
            for rule in policy.get("rules", [])
        }
        toolkit = self._import_toolkit()
        self.engine = None
        self.audit = []
        if toolkit is not None:
            audit_log, policy_engine = toolkit
            self.engine = policy_engine(conflict_strategy="deny_overrides")
            self.engine.load_json(policy_text)
            self.audit = audit_log()

    @staticmethod
    def _import_toolkit():
        try:
            from agentmesh.governance import AuditLog, PolicyEngine
            return AuditLog, PolicyEngine
        except ImportError:
            local_source = (
                Path(__file__).resolve().parents[3]
                / "agent-governance-python" / "agent-mesh" / "src"
            )
            if not local_source.is_dir():
                return None
            sys.path.insert(0, str(local_source))
            try:
                from agentmesh.governance import AuditLog, PolicyEngine
                return AuditLog, PolicyEngine
            except ImportError:
                return None

    def _fallback_decision(self, context: dict[str, object], stage: str):
        action = context.get("action", {})
        governance = context.get("governance", {})
        nemo = context.get("nemo", {})
        authorized = bool(action.get("authorized"))

        def rule(condition: str, default: str | None) -> str | None:
            return self.rule_names.get(condition, default)

        if stage == "pre_tool":
            allowed = authorized
            result = "allow" if allowed else "deny"
            matched_rule = rule("action.authorized", "allow-authorized-agent-step") if allowed else None
        elif action.get("status") == "unauthorized":
            allowed, result = False, "deny"
            matched_rule = rule("action.status == 'unauthorized'", "deny-unauthorized-final-action")
        elif governance.get("status") == "conflict":
            allowed, result = False, "require_approval"
            matched_rule = rule("governance.status == 'conflict'", "require-review-for-unreconciled-conflict")
        elif governance.get("status") == "failed":
            allowed, result = False, "deny"
            matched_rule = rule("governance.status == 'failed'", "deny-incorrect-record-relative-result")
        elif nemo.get("status") == "flagged":
            allowed, result = False, "require_approval"
            matched_rule = rule("nemo.status == 'flagged'", "require-review-for-nemo-content-flag")
        elif nemo.get("status") in {"passed", "unavailable"} and governance.get("status") == "passed" and authorized:
            allowed, result = True, "allow"
            if nemo.get("status") == "passed":
                condition = "action.status == 'authorized' and governance.status == 'passed' and nemo.status == 'passed'"
                matched_rule = rule(condition, "allow-verified-action")
            else:
                condition = "action.status == 'authorized' and governance.status == 'passed' and nemo.status == 'unavailable'"
                matched_rule = rule(condition, "allow-verified-action-with-nemo-unavailable")
        else:
            allowed, result, matched_rule = False, "deny", None

        return SimpleNamespace(
            allowed=allowed, action=result, matched_rule=matched_rule,
            policy_name=self.policy_name,
            reason=matched_rule or "No matching rule; default deny", evaluation_ms=0.0,
        )

    def evaluate(self, agent: str, action: str, context: dict[str, object], stage: str, resource: str):
        if self.engine is None:
            decision = self._fallback_decision(context, stage)
            self.audit.append({
                "agent": agent, "action": action, "resource": resource,
                "decision": decision.action, "matched_rule": decision.matched_rule,
            })
            return decision

        decision = self.engine.evaluate(agent, context, stage=stage)
        self.audit.log(
            event_type="policy_evaluation", agent_did=agent, action=action,
            resource=resource, data=context,
            outcome="success" if decision.allowed else "denied",
            policy_decision=decision.action, policy_version="aml-runtime-governance@1.0",
        )
        return decision


class SecurityLane:
    """Top deterministic gate: check each agent action before content monitoring."""

    def __init__(self, toolkit: ToolkitGovernanceAdapter | None = None) -> None:
        self.toolkit = toolkit

    def check(self, chain: Sequence[ChainStep], alert_id: str = "unknown") -> tuple[SecurityCheck, ...]:
        checks = []
        for step in chain:
            authorized = step.action_type in step.allowed_actions
            reason = (
                f"{step.agent} performed '{step.action_type}', within its delegated actions."
                if authorized else
                f"{step.agent} performed '{step.action_type}', outside its delegated actions."
            )
            if self.toolkit is not None:
                policy = self.toolkit.evaluate(
                    agent=step.agent, action=step.action_type,
                    context={"action": {"type": step.action_type, "authorized": authorized},
                             "agent": {"step": step.step_index}},
                    stage="pre_tool", resource=f"aml-alert:{alert_id}",
                )
                authorized = authorized and bool(policy.allowed)
                if not authorized:
                    reason = getattr(policy, "reason", "AGT deterministic gate denied the action.")
            checks.append(SecurityCheck(step.agent, step.step_index, authorized, reason))
        return tuple(checks)


class ChainGovernance:
    """Deterministic correctness check against the original transaction records."""

    def check(self, alert: Alert, chain: Sequence[ChainStep]) -> ChainAttribution:
        if not chain:
            raise ValueError("A chain must contain at least one agent step.")

        truth = recompute_structuring(alert)
        final = chain[-1]
        expected = "FILE_SAR" if truth.is_structuring else "CLEAR"

        for step in chain:
            acted_account = getattr(step, "acted_account_id", None)
            if acted_account is not None and acted_account != alert.account_id:
                reason = f"{step.agent} at step {step.step_index} acted on {acted_account}; alert belongs to {alert.account_id}."
                return self._failed("wrong-target", step, reason, truth, final, expected)

        for step in chain:
            if step.action_type not in step.allowed_actions:
                reason = (f"{step.agent} at step {step.step_index} performed '{step.action_type}', outside "
                          f"delegated actions ({', '.join(sorted(step.allowed_actions))}).")
                return self._failed("mis-delegation", step, reason, truth, final, expected)

        if len({step.disposition for step in chain}) > 1:
            first = chain[0].disposition
            culprit = next((step for step in chain if step.disposition != first), chain[-1])
            disagreements = ", ".join(f"{step.agent}->{step.disposition}" for step in chain)
            return self._failed("conflict", culprit,
                                f"Agents produced contradictory dispositions without reconciliation: {disagreements}.",
                                truth, final, expected)

        if final.disposition == expected:
            return ChainAttribution(
                passed=True, category=None, origin_agent=None, origin_step=None, propagators=(),
                reason=(f"End-to-end outcome '{final.disposition}' matches the records: "
                        f"${truth.aggregate:,.0f} across {truth.qualifying_count} sub-threshold deposits."),
                record_amounts=truth.qualifying_amounts, record_aggregate=truth.aggregate,
                is_structuring=truth.is_structuring, final_disposition=final.disposition,
                expected_disposition=expected,
            )

        origin = self._find_origin(chain, truth.qualifying_amounts) or chain[0]
        propagators = tuple(step.agent for step in chain if step.step_index > origin.step_index)
        reason = (
            f"Final disposition '{final.disposition}' contradicts the records (recomputed "
            f"${truth.aggregate:,.0f}, requiring '{expected}'). {origin.agent} at step "
            f"{origin.step_index} first diverged from record-relative evidence; downstream agents "
            f"({', '.join(propagators) or 'none'}) propagated that result."
        )
        return ChainAttribution(
            passed=False, category="transitive-corruption", origin_agent=origin.agent,
            origin_step=origin.step_index, propagators=propagators, reason=reason,
            record_amounts=truth.qualifying_amounts, record_aggregate=truth.aggregate,
            is_structuring=truth.is_structuring, final_disposition=final.disposition,
            expected_disposition=expected,
        )

    @staticmethod
    def _find_origin(chain: Sequence[ChainStep], record_amounts: tuple[float, ...]) -> ChainStep | None:
        for step in chain:
            if step.received_from is None and tuple(step.used_amounts) != record_amounts:
                return step
            if step.received_from is not None and tuple(step.used_amounts) != tuple(step.received_amounts or ()):
                return step
        return None

    @staticmethod
    def _failed(category, step, reason, truth, final, expected) -> ChainAttribution:
        return ChainAttribution(
            passed=False, category=category, origin_agent=step.agent, origin_step=step.step_index,
            propagators=(), reason=reason, record_amounts=truth.qualifying_amounts,
            record_aggregate=truth.aggregate, is_structuring=truth.is_structuring,
            final_disposition=final.disposition, expected_disposition=expected,
        )


class AgentContentMonitor:
    """Middle-layer monitor: inspect output content, not the correctness of AML decisions."""

    def __init__(self, check_output: Callable[[str], NemoVerdict] = nemo_check) -> None:
        self.check_output = check_output

    def check(self, step: ChainStep, authorized: bool) -> dict[str, object]:
        if not authorized:
            verdict: NemoVerdict = {
                "passed": True, "rail": "skipped",
                "reason": "The deterministic top gate denied this step before execution.",
            }
        else:
            output = (
                f"Agent: {step.agent}\nAction: {step.action_type}\n"
                f"Disposition: {step.disposition}\nEvidence amounts: {list(step.used_amounts)}"
            )
            verdict = self.check_output(output)
        return {"agent": step.agent, **verdict}


class ActionGovernance:
    """Bottom stage: let AGT consume signals and produce the final PolicyDecision."""

    def __init__(self, toolkit: ToolkitGovernanceAdapter) -> None:
        self.toolkit = toolkit

    def decide(self, agent: str, action: str, resource: str, authorized: bool,
               correctness: ChainAttribution, nemo: dict[str, object],
               capabilities: Sequence[str] = ()) -> dict[str, object]:
        context = {
            "action": {"type": action, "authorized": authorized,
                       "status": "authorized" if authorized else "unauthorized"},
            "governance": {
                "passed": correctness.passed,
                "status": (
                    "passed" if correctness.passed else
                    "conflict" if correctness.category == "unreconciled-conflict" else "failed"
                ),
                "category": correctness.category or "",
            },
            "nemo": nemo,
            "resource": resource,
            "capabilities": list(capabilities),
        }
        decision = self.toolkit.evaluate(agent, action, context, "post_tool", resource)
        return {
            "allowed": bool(decision.allowed), "action": decision.action,
            "reason": getattr(decision, "reason", ""),
            "matched_rule": getattr(decision, "matched_rule", None),
            "policy_name": getattr(decision, "policy_name", None),
            "evaluation_ms": getattr(decision, "evaluation_ms", None),
        }


@dataclass
class ChainSessionMetrics:
    """Bounded aggregate state retained across pipeline evaluations."""

    context_window: int = 8
    checkpoint_interval: int = 5
    turns: int = 0
    retained_alerts: list[str] = field(default_factory=list)
    checkpoints: list[int] = field(default_factory=list)
    security_intact_but_wrong: int = 0
    governance_flags: int = 0
    security_flags: int = 0
    nemo_flags: int = 0
    nemo_unavailable: int = 0

    @property
    def context_pressure(self) -> float:
        return self.turns / self.context_window

    @property
    def retained_context(self) -> int:
        return len(self.retained_alerts)


class AMLRuntimeGovernance:
    """Compose the architecture's top gate, middle monitor, and bottom decision."""

    def __init__(self, context_window: int = 8, checkpoint_interval: int = 5,
                 content_checker: Callable[[str], NemoVerdict] = nemo_check) -> None:
        if context_window < 1 or checkpoint_interval < 1:
            raise ValueError("context_window and checkpoint_interval must be positive")
        self.metrics = ChainSessionMetrics(context_window, checkpoint_interval)
        self.toolkit = ToolkitGovernanceAdapter()
        self.security = SecurityLane(self.toolkit)
        self.content = AgentContentMonitor(content_checker)
        self.correctness = ChainGovernance()
        self.action_governance = ActionGovernance(self.toolkit)

    def evaluate(self, outcome: ChainOutcome) -> ChainReview:
        resource = f"aml-alert:{outcome.alert.alert_id}"
        security = self.security.check(outcome.chain, outcome.alert.alert_id)
        content_checks = tuple(
            self.content.check(step, check.passed)
            for step, check in zip(outcome.chain, security)
        )
        nemo_signal, nemo_live = summarize_nemo_checks(list(content_checks))
        correctness = self.correctness.check(outcome.alert, outcome.chain)
        authorized = all(check.passed for check in security)
        action_decision = self.action_governance.decide(
            agent="aml-action-governance", action="commit_disposition", resource=resource,
            authorized=all(check.passed for check in security), correctness=correctness,
            nemo=nemo_signal,
        )
        self._update_metrics(outcome, security, correctness, content_checks)
        return ChainReview(
            outcome=outcome, security=security, governance=correctness,
            content_checks=content_checks, action_decision=action_decision,
            agt_live=self.toolkit.engine is not None, nemo_live=nemo_live,
        )

    def _update_metrics(self, outcome, security, correctness, content_checks) -> None:
        metrics = self.metrics
        metrics.turns += 1
        metrics.retained_alerts.append(outcome.alert.alert_id)
        del metrics.retained_alerts[:-metrics.context_window]
        if metrics.turns % metrics.checkpoint_interval == 0:
            metrics.checkpoints.append(metrics.turns)

        all_authorized = all(check.passed for check in security)
        has_nemo_flag = any(not check["passed"] for check in content_checks)
        has_unavailable = any(check["rail"] == "unavailable" for check in content_checks)
        metrics.security_flags += not all_authorized
        metrics.governance_flags += not correctness.passed
        metrics.nemo_flags += has_nemo_flag
        metrics.nemo_unavailable += has_unavailable
        metrics.security_intact_but_wrong += all_authorized and not correctness.passed


class ChainMonitor(AMLRuntimeGovernance):
    """Streamlit entry point retained for the existing chain review UI."""


__all__ = [
    "AML_POLICY", "AMLRuntimeGovernance", "ActionGovernance", "AgentContentMonitor",
    "ChainGovernance", "ChainMonitor", "ChainSessionMetrics", "SecurityLane",
    "ToolkitGovernanceAdapter",
]
