"""AML reference implementation of the optional correctness contract."""

from __future__ import annotations

from correctness import CorrectnessResult
from domain import Alert, BranchCheck, ChainStep, GraphOutcome, GraphStep
from governance import ChainGovernance

from aml.calculations import (
    merge_disposition,
    recompute_kyc,
    recompute_sanctions,
    recompute_transactions,
)


SANCTIONS_AGENT = "customer-screening"
TRANSACTION_AGENT = "transaction-analysis"
KYC_AGENT = "kyc-entity"
DISPOSITION_AGENT = "decision"
AML_BRANCHES = (SANCTIONS_AGENT, TRANSACTION_AGENT, KYC_AGENT)


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _check_transactions(alert: Alert, step: GraphStep) -> BranchCheck:
    projected = ChainStep(
        agent=step.agent,
        step_index=1,
        used_amounts=step.used_amounts,
        used_aggregate=step.used_aggregate,
        disposition=step.disposition or "CLEAR",
        action_type=step.action_type,
        allowed_actions=step.allowed_actions,
        received_from=None,
        received_amounts=None,
        acted_account_id=step.acted_account_id,
    )
    governance = ChainGovernance().check(alert, (projected,))
    truth = recompute_transactions(alert)
    if governance.passed:
        return BranchCheck(step.agent, step.node_id, step.finding, truth.finding, True, truth.detail)
    if governance.category != "transitive-corruption":
        return BranchCheck(step.agent, step.node_id, step.finding, truth.finding, False, governance.reason)
    misread = next(
        ((recorded, used) for recorded, used in zip(governance.record_amounts, step.used_amounts) if recorded != used),
        None,
    )
    detail = (
        f"misread {_money(misread[0])} as {_money(misread[1])}"
        if misread else "used amounts differ from records"
    )
    detail += (
        f"; used {_money(step.used_aggregate)} total vs {_money(governance.record_aggregate)} "
        f"in the records (records say {truth.finding})"
    )
    return BranchCheck(step.agent, step.node_id, step.finding, truth.finding, False, detail)


def _check_simple(alert: Alert, step: GraphStep, recompute) -> BranchCheck:
    truth = recompute(alert)
    if step.acted_account_id != alert.account_id:
        return BranchCheck(
            step.agent,
            step.node_id,
            step.finding,
            truth.finding,
            False,
            f"acted on {step.acted_account_id}; alert belongs to {alert.account_id}",
        )
    matches = step.finding == truth.finding and step.flagged == truth.flagged
    return BranchCheck(step.agent, step.node_id, step.finding, truth.finding, matches, truth.detail)


class AmlCorrectnessComponent:
    """Recompute AML branch truth from records and attribute divergence."""

    name = "aml-record-relative-correctness"

    def check(self, trace: GraphOutcome, records: object) -> CorrectnessResult:
        if not isinstance(records, Alert):
            raise TypeError("AmlCorrectnessComponent requires an AML Alert as records")

        by_agent = {step.agent: step for step in trace.steps}
        merge = by_agent[DISPOSITION_AGENT]
        checkers = {
            SANCTIONS_AGENT: lambda step: _check_simple(records, step, recompute_sanctions),
            TRANSACTION_AGENT: lambda step: _check_transactions(records, step),
            KYC_AGENT: lambda step: _check_simple(records, step, recompute_kyc),
        }
        recomputers = {
            SANCTIONS_AGENT: recompute_sanctions,
            TRANSACTION_AGENT: recompute_transactions,
            KYC_AGENT: recompute_kyc,
        }
        checks = tuple(checkers[agent](by_agent[agent]) for agent in AML_BRANCHES)
        truth = {agent: recomputers[agent](records) for agent in AML_BRANCHES}
        expected = merge_disposition(*(truth[agent].flagged for agent in AML_BRANCHES))
        actual = merge.disposition or "CLEAR"
        diverged = [check for check in checks if not check.matches]

        merge_check = BranchCheck(
            merge.agent,
            merge.node_id,
            actual,
            expected,
            actual == expected,
            "merged disposition matches the record-relative branch results"
            if actual == expected else "merged disposition differs from the record-relative branch results",
        )
        all_checks = (*checks, merge_check)

        if actual == expected:
            note = (
                f" Branch divergence did not change the outcome: "
                f"{', '.join(check.agent for check in diverged)}."
                if diverged else ""
            )
            return CorrectnessResult(
                passed=True,
                expected=expected,
                actual=actual,
                origin_agent=None,
                reason=f"Merged {actual} matches the records.{note}",
                checks=all_checks,
                metadata={"component": self.name},
            )

        bad = diverged[0] if diverged else checks[0]
        origin = by_agent[bad.agent]
        masked = tuple(
            check.agent
            for check in checks
            if check.matches and not by_agent[check.agent].flagged
        )
        siblings = ", ".join(f"{by_agent[agent].node_id} ({agent})" for agent in masked)
        reason = (
            f"branch {origin.node_id} ({origin.agent}) {bad.detail}; merged {actual} is wrong "
            f"(records require {expected})"
            + (f". Masked by clean siblings {siblings}." if siblings else ".")
        )
        return CorrectnessResult(
            passed=False,
            expected=expected,
            actual=actual,
            origin_agent=origin.agent,
            origin_node=origin.node_id,
            related_agents=masked,
            reason=reason,
            category="masked-branch-error",
            checks=all_checks,
            metadata={"component": self.name},
        )


__all__ = ["AmlCorrectnessComponent"]
