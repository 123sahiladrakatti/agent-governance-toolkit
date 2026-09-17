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


AML_POLICY = """
{"apiVersion":"governance.toolkit/v1","version":"1.0","name":"aml-runtime-chain","description":"Explicit authorization and post-run review policy for the AML chain.","agents":["*"],"scope":"global","default_action":"deny","rules":[{"name":"allow-authorized-chain-hop","description":"Allow only actions explicitly present in the agent delegation.","stage":"pre_tool","condition":"action.authorized","action":"allow","priority":100},{"name":"require-review-for-governance-failure","description":"A chain that fails record-relative governance requires human review.","stage":"post_tool","condition":"governance.failed","action":"require_approval","priority":100},{"name":"allow-record-relative-governance-pass","description":"Allow a chain whose final outcome matches the original records.","stage":"post_tool","condition":"governance.passed","action":"allow","priority":50}]}
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
            allowed = (
                bool(context.get("action", {}).get("authorized"))
                if stage == "pre_tool"
                else bool(context.get("governance", {}).get("passed"))
            )
            decision = SimpleNamespace(
                allowed=allowed,
                action="allow" if allowed else ("require_approval" if stage == "post_tool" else "deny"),
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

    @property
    def context_pressure(self) -> float:
        return self.turns / self.context_window

    @property
    def retained_context(self) -> int:
        return len(self.retained_alerts)


class ChainMonitor:
    """Evaluate each chain while retaining compact state across the full run."""

    def __init__(self, context_window: int = 8, checkpoint_interval: int = 5) -> None:
        self.metrics = ChainSessionMetrics(context_window, checkpoint_interval)
        self.toolkit = ToolkitGovernanceAdapter()
        self.security = SecurityLane(self.toolkit)
        self.governance = ChainGovernance()

    def evaluate(self, outcome: ChainOutcome) -> ChainReview:
        sec = self.security.check(outcome.chain, outcome.alert.alert_id)
        gov = self.governance.check(outcome.alert, outcome.chain)
        self.toolkit.evaluate(
            agent="aml-governance-monitor",
            action="aml_chain_review",
            context={"governance": {"failed": not gov.passed, "passed": gov.passed}},
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
        if all_security_passed and not gov.passed:
            m.security_intact_but_wrong += 1
        from domain import ChainReview as _CR  # local import avoids cycle at module import
        return _CR(outcome=outcome, security=tuple(sec), governance=gov)