"""Cheap deterministic runtime governance lanes for the AML demo."""

from __future__ import annotations

from domain import AgentAction, Alert, GovernanceVerdict, recompute_structuring

class RegularGovernance:
    """Check permission and access only; intentionally ignores semantic correctness."""
    def check(self, action: AgentAction, alert: Alert) -> GovernanceVerdict:
        if action.action_type not in action.delegation.allowed_actions:
            return GovernanceVerdict(False, "mis-delegation", action.responsible_agent, action.step, "Tool/action is outside the triage delegation.", {"allowed_actions": sorted(action.delegation.allowed_actions), "action_type": action.action_type})
        if action.delegation.data_access_scope != "portfolio":
            return GovernanceVerdict(False, "mis-delegation", action.responsible_agent, action.step, "Account data is outside the delegated access scope.", {"scope": action.delegation.data_access_scope})
        return GovernanceVerdict(True, None, action.responsible_agent, action.step, "Authorized action and portfolio data access.", {"action_type": action.action_type, "data_scope": action.delegation.data_access_scope})

class EnhancedGovernance:
    """Recompute record-relative truth and compare it with the permitted action."""
    def check(self, action: AgentAction, alert: Alert) -> GovernanceVerdict:
        actual = recompute_structuring(alert)
        if action.acted_account_id != alert.account_id:
            return GovernanceVerdict(False, "wrong-target", action.responsible_agent, action.step, f"Agent acted on {action.acted_account_id}; alert belongs to {alert.account_id}.", {"alert_account": alert.account_id, "acted_account": action.acted_account_id})
        if action.action_type not in action.delegation.allowed_actions:
            return GovernanceVerdict(False, "mis-delegation", action.responsible_agent, action.step, f"Agent recorded {action.action_type}; triage delegated only {', '.join(sorted(action.delegation.allowed_actions))}.", {"allowed_actions": sorted(action.delegation.allowed_actions), "action_type": action.action_type})
        if tuple(action.claimed_amounts) != actual.qualifying_amounts or action.claimed_threshold != actual.threshold:
            mismatch = next(((claimed, recorded) for claimed, recorded in zip(action.claimed_amounts, actual.qualifying_amounts) if claimed != recorded), (None, None))
            reason = f"Agent claimed amounts {list(action.claimed_amounts)}; record amounts are {list(actual.qualifying_amounts)}. Recomputed aggregate ${actual.aggregate:,.0f} {'≥' if actual.is_structuring else '<'} ${30_000:,.0f} structuring rule."
            if mismatch[0] is not None:
                reason = f"Agent used ${mismatch[0]:,.0f}; record shows ${mismatch[1]:,.0f}. " + reason
            return GovernanceVerdict(False, "semantic-drift", action.responsible_agent, action.step, reason, {"claimed_amounts": action.claimed_amounts, "record_amounts": actual.qualifying_amounts, "claimed_threshold": action.claimed_threshold, "record_threshold": actual.threshold, "record_aggregate": actual.aggregate})
        expected = "FILE_SAR" if actual.is_structuring else "CLEAR"
        if action.disposition != expected or (action.disposition == "FILE_SAR" and not action.sar_record_id) or (action.disposition == "CLEAR" and actual.is_structuring):
            return GovernanceVerdict(False, "false-completion", action.responsible_agent, action.step, f"Agent reported {action.disposition_status}, but records recompute to {expected}: {actual.qualifying_count} deposits totaling ${actual.aggregate:,.0f}.", {"reported_status": action.disposition_status, "reported_disposition": action.disposition, "recomputed_disposition": expected, "record_aggregate": actual.aggregate, "sar_record_id": action.sar_record_id})
        return GovernanceVerdict(True, None, action.responsible_agent, action.step, f"Record-relative evidence supports {expected}: ${actual.aggregate:,.0f} across {actual.qualifying_count} qualifying deposits.", {"recomputed_disposition": expected, "record_aggregate": actual.aggregate, "record_amounts": actual.qualifying_amounts})
