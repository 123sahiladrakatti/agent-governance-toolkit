"""Cheap deterministic runtime governance lanes for the AML agent demo.

Two lanes, both LLM-free and record-relative:
  * Security lane   - was each hop authorized? (permission only)
  * Governance lane - is the chain's outcome correct vs the original records,
                      and if not, which agent originated the error?

The single-decision checkers (RegularGovernance / EnhancedGovernance) are kept
for reference and the 2-agent path; the chain checkers add multi-hop attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Callable

from domain import (
    AgentAction,
    Alert,
    ChainAttribution,
    ChainOutcome,
    ChainReview,
    ChainStep,
    GovernanceVerdict,
    SecurityCheck,
    recompute_structuring,
)
from nemo_guardrails import NemoVerdict, nemo_check, summarize_nemo_checks


AML_POLICY = """
{"apiVersion":"governance.toolkit/v1","version":"1.0","name":"aml-runtime-chain","description":"Pre-agent authorization and final Action Governance over record-relative correctness and NeMo content signals.","agents":["*"],"scope":"global","default_action":"deny","rules":[{"name":"allow-authorized-chain-hop","description":"Allow only actions explicitly present in the agent delegation.","stage":"pre_tool","condition":"action.authorized","action":"allow","priority":100},{"name":"deny-unauthorized-final-action","description":"The final action requires successful deterministic authorization.","stage":"post_tool","condition":"action.status == 'unauthorized'","action":"deny","priority":200},{"name":"deny-record-relative-governance-failure","description":"Incorrect or corrupted AML decisions are denied by deterministic governance.","stage":"post_tool","condition":"governance.status == 'failed'","action":"deny","priority":150},{"name":"require-review-for-nemo-content-flag","description":"A NeMo content flag requires human approval.","stage":"post_tool","condition":"nemo.status == 'flagged'","action":"require_approval","priority":100},{"name":"allow-record-relative-and-content-pass","description":"Allow only when deterministic correctness, authorization, and NeMo content checks pass.","stage":"post_tool","condition":"action.status == 'authorized' and governance.status == 'passed' and nemo.status == 'passed'","action":"allow","priority":50},{"name":"allow-with-nemo-unavailable","description":"Allow when correctness and authorization pass while the UI explicitly reports NeMo unavailable.","stage":"post_tool","condition":"action.status == 'authorized' and governance.status == 'passed' and nemo.status == 'unavailable'","action":"allow","priority":40}]}
"""


class ToolkitGovernanceAdapter:
    """Apply Agent Governance Toolkit policy and append-only audit logging."""

    def __init__(self, policy_text: str = AML_POLICY) -> None:
        toolkit = None
        try:
            from agentmesh.governance import AuditLog, PolicyEngine
            toolkit = (AuditLog, PolicyEngine)
        except ImportError as exc:
            source_root = Path(__file__).resolve().parents[3]
            local_source = source_root / "agent-governance-python" / "agent-mesh" / "src"
            if local_source.is_dir():
                sys.path.insert(0, str(local_source))
                try:
                    from agentmesh.governance import AuditLog, PolicyEngine
                    toolkit = (AuditLog, PolicyEngine)
                except ImportError:
                    pass

        self.engine = None
        self.audit = []
        if toolkit is not None:
            audit_log, policy_engine = toolkit
            self.engine = policy_engine(conflict_strategy="deny_overrides")
            self.engine.load_json(policy_text)
            self.audit = audit_log()

    def evaluate(
        self,
        agent: str,
        action: str,
        context: dict[str, object],
        stage: str,
        resource: str,
    ):
        if self.engine is None:
            # Keep the standalone demo usable with its original Streamlit
            # interpreter; the full toolkit path is used when installed.
            authorized = bool(context.get("action", {}).get("authorized"))
            governance_passed = bool(context.get("governance", {}).get("passed"))
            nemo_passed = bool(context.get("nemo", {}).get("passed", True))
            if stage == "pre_tool":
                allowed, action_name = authorized, "allow" if authorized else "deny"
            elif not authorized or not governance_passed:
                allowed, action_name = False, "deny"
            elif not nemo_passed:
                allowed, action_name = False, "require_approval"
            else:
                allowed, action_name = True, "allow"
            decision = SimpleNamespace(
                allowed=allowed,
                action=action_name,
                reason="fallback Action Governance policy",
            )
            self.audit.append({"agent": agent, "action": action, "resource": resource, "decision": decision.action})
            return decision

        decision = self.engine.evaluate(agent, context, stage=stage)
        self.audit.log(
            event_type="policy_evaluation", agent_did=agent, action=action,
            resource=resource, data=context,
            outcome="success" if decision.allowed else "denied",
            policy_decision=decision.action, policy_version="aml-runtime-chain@1.0",
        )
        return decision


# ---------------------------------------------------------------------------
# Single-decision lanes (unchanged; used by the 2-agent path)
# ---------------------------------------------------------------------------
class RegularGovernance:
    """Check permission and access only; intentionally ignores semantic correctness."""

    def check(self, action: AgentAction, alert: Alert) -> GovernanceVerdict:
        if action.action_type not in action.delegation.allowed_actions:
            return GovernanceVerdict(False, "mis-delegation", action.responsible_agent, action.step, "Tool/action is outside the triage delegation.", {"allowed_actions": sorted(action.delegation.allowed_actions), "action_type": action.action_type})
        if action.delegation.data_access_scope != "portfolio":
            return GovernanceVerdict(False, "mis-delegation", action.responsible_agent, action.step, "Account data is outside the delegated access scope.", {"scope": action.delegation.data_access_scope})
        return GovernanceVerdict(True, None, action.responsible_agent, action.step, "Authorized action and portfolio data access.", {"action_type": action.action_type})


class EnhancedGovernance:
    """Recompute record-relative truth and compare it with the permitted action."""

    def check(self, action: AgentAction, alert: Alert) -> GovernanceVerdict:
        actual = recompute_structuring(alert)
        if action.acted_account_id != alert.account_id:
            return GovernanceVerdict(False, "wrong-target", action.responsible_agent, action.step, f"Agent acted on {action.acted_account_id}; alert belongs to {alert.account_id}.", {})
        if action.action_type not in action.delegation.allowed_actions:
            return GovernanceVerdict(False, "mis-delegation", action.responsible_agent, action.step, f"Agent recorded {action.action_type}; triage delegated only {', '.join(sorted(action.delegation.allowed_actions))}.", {})
        if tuple(action.claimed_amounts) != actual.qualifying_amounts or action.claimed_threshold != actual.threshold:
            reason = f"Agent claimed amounts {list(action.claimed_amounts)}; record amounts are {list(actual.qualifying_amounts)}."
            return GovernanceVerdict(False, "semantic-drift", action.responsible_agent, action.step, reason, {})
        expected = "FILE_SAR" if actual.is_structuring else "CLEAR"
        if action.disposition != expected or (action.disposition == "FILE_SAR" and not action.sar_record_id) or (action.disposition == "CLEAR" and actual.is_structuring):
            return GovernanceVerdict(False, "false-completion", action.responsible_agent, action.step, f"Agent reported {action.disposition_status}, but records recompute to {expected}.", {})
        return GovernanceVerdict(True, None, action.responsible_agent, action.step, f"Record-relative evidence supports {expected}.", {})


# ---------------------------------------------------------------------------
# Chain lanes (multi-hop: security per hop + governance with origin attribution)
# ---------------------------------------------------------------------------
class SecurityLane:
    """Per-hop authorization. This is the 'security intact' view: every hop is
    checked in isolation against what that agent was permitted to do."""

    def __init__(self, toolkit: ToolkitGovernanceAdapter | None = None) -> None:
        self.toolkit = toolkit

    def check(self, chain: tuple[ChainStep, ...], alert_id: str = "unknown") -> tuple[SecurityCheck, ...]:
        checks = []
        for step in chain:
            permitted = step.action_type in step.allowed_actions
            if self.toolkit is not None:
                policy = self.toolkit.evaluate(
                    agent=step.agent,
                    action=step.action_type,
                    context={
                        "action": {
                            "type": "aml_chain_handoff",
                            "authorized": permitted,
                        },
                        "agent": {"step": step.step_index},
                    },
                    stage="pre_tool",
                    resource=f"aml-alert:{alert_id}",
                )
                permitted = permitted and policy.allowed
            reason = (f"{step.agent} performed '{step.action_type}', which is within its permitted actions."
                      if permitted else
                      f"{step.agent} performed '{step.action_type}', outside its permitted actions.")
            checks.append(SecurityCheck(step.agent, step.step_index, permitted, reason))
        return checks


class ChainGovernance:
    """Governance over a whole chain.

    Recomputes the correct outcome from the ORIGINAL records, and if the chain's
    final disposition contradicts it, walks the provenance backward to attribute
    the error to the agent that first diverged from the records (the origin),
    distinguishing it from downstream agents that merely trusted and forwarded
    the bad value (the propagators). No LLM calls; pure record-relative arithmetic.
    """

    def check(self, alert: Alert, chain: tuple[ChainStep, ...]) -> ChainAttribution:
        truth = recompute_structuring(alert)
        final = chain[-1]
        expected = "FILE_SAR" if truth.is_structuring else "CLEAR"

        # --- New distinct fault classes (checked before the corruption/outcome check) ---
        # 1) Wrong-target: an agent acted on an account other than the alert's.
        for step in chain:
            acted = getattr(step, "acted_account_id", None)
            if acted is not None and acted != alert.account_id:
                reason = (f"{step.agent} at step {step.step_index} acted on account {acted}, "
                          f"but the alert belongs to {alert.account_id}. Authorized to access accounts, "
                          f"but the wrong entity - the decision does not concern this alert.")
                return ChainAttribution(False, "wrong-target", step.agent, step.step_index, (), reason,
                                        truth.qualifying_amounts, truth.aggregate, truth.is_structuring,
                                        final.disposition, expected)
        # 2) Scope / mis-delegation: an agent took an action outside its allowed_actions.
        #    (The security lane also catches this - the overlap is intentional.)
        for step in chain:
            if step.action_type not in step.allowed_actions:
                reason = (f"{step.agent} at step {step.step_index} performed '{step.action_type}', "
                          f"which is outside its delegated actions "
                          f"({', '.join(sorted(step.allowed_actions))}). Action taken beyond granted authority.")
                return ChainAttribution(False, "mis-delegation", step.agent, step.step_index, (), reason,
                                        truth.qualifying_amounts, truth.aggregate, truth.is_structuring,
                                        final.disposition, expected)
        # 3) Conflict: two agents recorded contradictory dispositions for the same alert.
        dispositions = {step.step_index: step.disposition for step in chain}
        if len(set(dispositions.values())) > 1:
            disagreeing = ", ".join(f"{s.agent}->{s.disposition}" for s in chain)
            reason = (f"Agents disagree on the disposition with no reconciliation ({disagreeing}). "
                      f"Two authorized agents reached contradictory conclusions for one alert.")
            first_disp = chain[0].disposition
            culprit = next((s for s in chain if s.disposition != first_disp), chain[-1])
            return ChainAttribution(False, "conflict", culprit.agent, culprit.step_index, (), reason,
                                    truth.qualifying_amounts, truth.aggregate, truth.is_structuring,
                                    final.disposition, expected)

        if final.disposition == expected:
            return ChainAttribution(
                passed=True, category=None, origin_agent=None, origin_step=None, propagators=(),
                reason=(f"End-to-end outcome '{final.disposition}' matches the records: "
                        f"${truth.aggregate:,.0f} across {truth.qualifying_count} sub-threshold deposits."),
                record_amounts=truth.qualifying_amounts, record_aggregate=truth.aggregate,
                is_structuring=truth.is_structuring, final_disposition=final.disposition, expected_disposition=expected,
            )

        # Outcome is wrong. Find the first hop whose used value diverged from its source of truth.
        origin = None
        for step in chain:
            if step.received_from is None:
                # The record reader: compare against the actual records.
                if tuple(step.used_amounts) != truth.qualifying_amounts:
                    origin = step
                    break
            else:
                # A propagator: it only originates error if it altered what it received.
                if tuple(step.used_amounts) != tuple(step.received_amounts or ()):
                    origin = step
                    break
        if origin is None:
            origin = chain[0]

        propagators = tuple(step.agent for step in chain if step.step_index > origin.step_index)
        prop_text = ", ".join(propagators) if propagators else "none"
        reason = (
            f"Final disposition '{final.disposition}' contradicts the records "
            f"(recomputed ${truth.aggregate:,.0f} across {truth.qualifying_count} sub-threshold deposits, "
            f"which requires '{expected}'). Walking the delegation chain back: {origin.agent} at step "
            f"{origin.step_index} used {[round(x) for x in origin.used_amounts]} while the records show "
            f"{[round(x) for x in truth.qualifying_amounts]}. Downstream agents ({prop_text}) trusted and "
            f"forwarded this value without re-checking it."
        )
        return ChainAttribution(
            passed=False, category="transitive-corruption", origin_agent=origin.agent, origin_step=origin.step_index,
            propagators=propagators, reason=reason, record_amounts=truth.qualifying_amounts,
            record_aggregate=truth.aggregate, is_structuring=truth.is_structuring,
            final_disposition=final.disposition, expected_disposition=expected,
        )


@dataclass
class ChainSessionMetrics:
    """Bounded state retained across a long-running chain session."""

    context_window: int = 8
    checkpoint_interval: int = 5
    turns: int = 0
    retained_alerts: list[str] = field(default_factory=list)
    checkpoints: list[int] = field(default_factory=list)
    security_intact_but_wrong: int = 0   # chains where every hop passed security but governance flagged
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


class ChainMonitor:
    """Evaluate each chain while retaining compact state across the full run."""

    def __init__(
        self,
        context_window: int = 8,
        checkpoint_interval: int = 5,
        content_checker: Callable[[str], NemoVerdict] = nemo_check,
    ) -> None:
        self.metrics = ChainSessionMetrics(context_window, checkpoint_interval)
        self.toolkit = ToolkitGovernanceAdapter()
        self.security = SecurityLane(self.toolkit)
        self.governance = ChainGovernance()
        self.content_checker = content_checker

    def evaluate(self, outcome: ChainOutcome) -> ChainReview:
        sec = []
        content_checks: list[dict[str, object]] = []
        for step in outcome.chain:
            security_check = self.security.check((step,), outcome.alert.alert_id)[0]
            sec.append(security_check)
            if security_check.passed:
                agent_output = (
                    f"Agent: {step.agent}\nAction: {step.action_type}\n"
                    f"Disposition: {step.disposition}\nEvidence amounts: {list(step.used_amounts)}"
                )
                verdict = self.content_checker(agent_output)
            else:
                verdict = {
                    "passed": True,
                    "rail": "skipped",
                    "reason": "The deterministic authorization gate denied this step before execution.",
                }
            content_checks.append({"agent": step.agent, **verdict})

        gov = self.governance.check(outcome.alert, outcome.chain)
        nemo_signal, nemo_live = summarize_nemo_checks(content_checks)
        nemo_failures = [check for check in content_checks if not check["passed"]]
        nemo_unavailable = [check for check in content_checks if check["rail"] == "unavailable"]
        decision = self.toolkit.evaluate(
            agent="aml-governance-monitor",
            action="aml_chain_review",
            context={
                "action": {
                    "authorized": all(check.passed for check in sec),
                    "status": "authorized" if all(check.passed for check in sec) else "unauthorized",
                },
                "governance": {
                    "failed": not gov.passed,
                    "passed": gov.passed,
                    "status": "passed" if gov.passed else "failed",
                },
                "nemo": nemo_signal,
            },
            stage="post_tool",
            resource=f"aml-alert:{outcome.alert.alert_id}",
        )
        m = self.metrics
        m.turns += 1
        m.retained_alerts.append(outcome.alert.alert_id)
        if len(m.retained_alerts) > m.context_window:
            m.retained_alerts.pop(0)
        if m.turns % m.checkpoint_interval == 0:
            m.checkpoints.append(m.turns)
        all_security_passed = all(check.passed for check in sec)
        if not all_security_passed:
            m.security_flags += 1
        if not gov.passed:
            m.governance_flags += 1
        if nemo_failures:
            m.nemo_flags += 1
        if nemo_unavailable:
            m.nemo_unavailable += 1
        if all_security_passed and not gov.passed:
            m.security_intact_but_wrong += 1
        from domain import ChainReview as _CR  # local import avoids cycle at module import
        action_decision = {
            "allowed": bool(decision.allowed),
            "action": decision.action,
            "reason": getattr(decision, "reason", ""),
            "matched_rule": getattr(decision, "matched_rule", None),
            "policy_name": getattr(decision, "policy_name", None),
        }
        return _CR(
            outcome=outcome,
            security=tuple(sec),
            governance=gov,
            content_checks=tuple(content_checks),
            action_decision=action_decision,
            agt_live=self.toolkit.engine is not None,
            nemo_live=nemo_live,
        )