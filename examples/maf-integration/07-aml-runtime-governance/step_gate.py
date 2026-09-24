"""Step-gated runtime governance: run one step, check it, gate the next.

This is the intervention path. Unlike ChainMonitor (which reviews a *completed*
chain), the runner here builds the chain one step at a time and runs a governance
check after each step. On a failing check it HALTS: downstream steps never run,
so the faulty result never reaches the step that would commit real harm.

Two intervention points, matching the two fault categories:
  * Permission fault (scope) -> blocked AT THE GATE, before the action executes.
    Knowable from the rules alone (action not in allowed_actions), so we refuse
    to run it at all.
  * Behavioral fault (corruption / wrong-target / conflict) -> detected AFTER the
    faulty agent produces output (we must see the output to check it), then the
    chain is halted before the next agent runs.

All checks are deterministic and record-relative. No LLM calls.
"""

from __future__ import annotations

from domain import (
    Alert,
    ChainStep,
    GatedRun,
    GatedStep,
    recompute_structuring,
)
from agent_sim import (
    INVESTIGATION_AGENT,
    CASE_AGENT,
    FILING_AGENT,
    CHAIN_ALLOWED,
    CHAIN_ACTION,
    _disposition_for,
    recompute_structuring as _rs,  # same function, explicit
)

# Which step, if reached and executed, is the consequential (harmful, irreversible)
# one. In this chain, filing (step 3) is the action that commits the disposition.
CONSEQUENTIAL_STEP_INDEX = 3


def _permission_ok(action_type: str, allowed: frozenset) -> bool:
    return action_type in allowed


def _build_step_1(alert: Alert, fault: str | None) -> ChainStep:
    truth = recompute_structuring(alert)
    if fault == "transitive-corruption" and truth.qualifying_amounts:
        used = tuple(2_900.0 if a == truth.qualifying_amounts[0] else a * 0.5 for a in truth.qualifying_amounts)
    else:
        used = truth.qualifying_amounts
    return ChainStep(
        agent=INVESTIGATION_AGENT, step_index=1, used_amounts=used, used_aggregate=sum(used),
        disposition=_disposition_for(used), action_type=CHAIN_ACTION[INVESTIGATION_AGENT],
        allowed_actions=CHAIN_ALLOWED[INVESTIGATION_AGENT], received_from=None, received_amounts=None,
        acted_account_id=alert.account_id,
    )


def _build_step_2(alert: Alert, s1: ChainStep) -> ChainStep:
    return ChainStep(
        agent=CASE_AGENT, step_index=2, used_amounts=s1.used_amounts, used_aggregate=s1.used_aggregate,
        disposition=s1.disposition, action_type=CHAIN_ACTION[CASE_AGENT],
        allowed_actions=CHAIN_ALLOWED[CASE_AGENT], received_from=INVESTIGATION_AGENT, received_amounts=s1.used_amounts,
        acted_account_id=alert.account_id,
    )


def _build_step_3(alert: Alert, s2: ChainStep, fault: str | None) -> ChainStep:
    filing_account = alert.account_id
    filing_action = CHAIN_ACTION[FILING_AGENT]
    filing_disposition = s2.disposition
    if fault == "wrong-target":
        filing_account = "ACCT-0099"
    elif fault == "scope":
        filing_action = "freeze_account"
    elif fault == "conflict":
        filing_disposition = "FILE_SAR" if s2.disposition == "CLEAR" else "CLEAR"
    return ChainStep(
        agent=FILING_AGENT, step_index=3, used_amounts=s2.used_amounts, used_aggregate=s2.used_aggregate,
        disposition=filing_disposition, action_type=filing_action,
        allowed_actions=CHAIN_ALLOWED[FILING_AGENT], received_from=CASE_AGENT, received_amounts=s2.used_amounts,
        acted_account_id=filing_account,
    )


def _check_step_behavioral(alert: Alert, step: ChainStep) -> tuple[bool, str | None, str]:
    """Record-relative check on a single step's output, as it is produced.

    Returns (ok, category_if_failed, reason). Behavioral only - permission/scope
    is handled by the gate before the action runs, not here.
    """
    truth = recompute_structuring(alert)

    # Wrong-target: this step acted on an account other than the alert's.
    if step.acted_account_id is not None and step.acted_account_id != alert.account_id:
        return (False, "wrong-target",
                f"{step.agent} acted on account {step.acted_account_id}, but the alert belongs to "
                f"{alert.account_id}. Wrong entity - halting before the decision commits.")

    # Corruption at the reader: the value it used diverges from the records.
    if step.received_from is None:
        if tuple(step.used_amounts) != truth.qualifying_amounts:
            return (False, "transitive-corruption",
                    f"{step.agent} used {[round(x) for x in step.used_amounts]} but the records show "
                    f"{[round(x) for x in truth.qualifying_amounts]} (aggregate ${truth.aggregate:,.0f}). "
                    f"Misread input detected at the source - halting before it propagates.")
    else:
        # A downstream step: did it faithfully carry what it received?
        if tuple(step.used_amounts) != tuple(step.received_amounts or ()):
            return (False, "transitive-corruption",
                    f"{step.agent} altered the value it received - halting.")

    return (True, None, f"{step.agent}'s output matches the records.")


def run_gated_chain(alert: Alert, fault: str | None) -> GatedRun:
    """Execute investigation -> case -> filing with a governance gate after each
    step and a permission gate before each action. Halt on the first failure."""
    steps: list[GatedStep] = []

    builders = [
        (1, INVESTIGATION_AGENT, lambda prev: _build_step_1(alert, fault)),
        (2, CASE_AGENT, lambda prev: _build_step_2(alert, prev)),
        (3, FILING_AGENT, lambda prev: _build_step_3(alert, prev, fault)),
    ]

    prev_step: ChainStep | None = None
    for idx, agent, build in builders:
        # ---- decide the action this step WOULD take, to run the pre-action gate ----
        # We build the step first to know its action_type/account, but treat a
        # permission failure as "blocked at the gate" (the action does not execute).
        candidate = build(prev_step)

        # 1) Permission gate (pre-action): knowable from rules alone.
        if not _permission_ok(candidate.action_type, candidate.allowed_actions):
            steps.append(GatedStep(
                step=None, agent=agent, step_index=idx, status="blocked_at_gate",
                reason=(f"{agent} attempted '{candidate.action_type}', which is outside its permitted "
                        f"actions ({', '.join(sorted(candidate.allowed_actions))}). Denied at the gate - "
                        f"the action never executed."),
            ))
            # mark the rest not-run
            for j, a, _ in builders[idx:]:
                steps.append(GatedStep(step=None, agent=a, step_index=j, status="not_run",
                                       reason="Chain already halted upstream."))
            prevented = idx <= CONSEQUENTIAL_STEP_INDEX
            return GatedRun(alert=alert, steps=tuple(steps), halted=True, halt_step_index=idx,
                            halt_kind="gate", category="mis-delegation",
                            reason=steps[idx - 1].reason, fault=fault,
                            consequential_prevented=prevented)

        # 2) Action executes -> behavioral check on its output.
        ok, category, reason = _check_step_behavioral(alert, candidate)
        if not ok:
            steps.append(GatedStep(step=candidate, agent=agent, step_index=idx,
                                   status="halted_after", reason=reason))
            for j, a, _ in builders[idx:]:
                steps.append(GatedStep(step=None, agent=a, step_index=j, status="not_run",
                                       reason="Downstream step never ran - chain halted by governance."))
            # consequential prevented if the harmful step (filing) had not yet run
            prevented = idx < CONSEQUENTIAL_STEP_INDEX or (idx == CONSEQUENTIAL_STEP_INDEX and category == "wrong-target")
            return GatedRun(alert=alert, steps=tuple(steps), halted=True, halt_step_index=idx,
                            halt_kind="behavioral", category=category, reason=reason, fault=fault,
                            consequential_prevented=prevented)

        # passed -> record and continue
        steps.append(GatedStep(step=candidate, agent=agent, step_index=idx, status="executed", reason=reason))
        prev_step = candidate

    # Reached the end with no halt: also check the whole-chain conflict condition,
    # since conflict is only visible when >1 disposition exists across steps.
    dispositions = {s.step.disposition for s in steps if s.step is not None}
    if len(dispositions) > 1:
        # attribute to the step that diverged from step 1
        first_disp = steps[0].step.disposition
        culprit = next((s for s in steps if s.step is not None and s.step.disposition != first_disp), steps[-1])
        # rebuild step list marking the culprit as halted and any after as not_run
        new_steps: list[GatedStep] = []
        for s in steps:
            if s.step_index < culprit.step_index:
                new_steps.append(s)
            elif s.step_index == culprit.step_index:
                new_steps.append(GatedStep(step=s.step, agent=s.agent, step_index=s.step_index,
                                           status="halted_after",
                                           reason=(f"{s.agent} reached a disposition ({s.step.disposition}) that "
                                                   f"contradicts the chain ({first_disp}) with no reconciliation - halting.")))
            else:
                new_steps.append(GatedStep(step=None, agent=s.agent, step_index=s.step_index,
                                           status="not_run", reason="Never ran - chain halted by governance."))
        return GatedRun(alert=alert, steps=tuple(new_steps), halted=True,
                        halt_step_index=culprit.step_index, halt_kind="behavioral", category="conflict",
                        reason=new_steps[culprit.step_index - 1].reason, fault=fault,
                        consequential_prevented=(culprit.step_index <= CONSEQUENTIAL_STEP_INDEX))

    return GatedRun(alert=alert, steps=tuple(steps), halted=False, halt_step_index=None,
                    halt_kind=None, category=None,
                    reason="All steps executed and passed governance; chain completed cleanly.",
                    fault=fault, consequential_prevented=False)


def run_gated_session(count: int = 40, seed: int = 7):
    """Run the whole session step-gated. Returns list[GatedRun]."""
    from agent_sim import generate_alerts, _fault_plan
    alerts = generate_alerts(count, seed)
    plan = _fault_plan(count)
    return [run_gated_chain(alert, plan.get(i)) for i, alert in enumerate(alerts)]