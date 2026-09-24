"""Deterministic long-running AML agent simulation.

Provides both the original single-decision investigation (kept for reference)
and the 3-agent delegation chain used by the demo, where a misread at the
record-reading agent propagates untouched through two trusting downstream agents.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from domain import (
    Alert,
    ChainOutcome,
    ChainStep,
    CustomerProfile,
    Disposition,
    Transaction,
    recompute_structuring,
)

# Chain roles: only the first agent reads raw records; the rest trust upstream.
INVESTIGATION_AGENT = "investigation-agent"
CASE_AGENT = "case-agent"
FILING_AGENT = "filing-agent"

CHAIN_ALLOWED = {
    INVESTIGATION_AGENT: frozenset({"investigate", "review_records"}),
    CASE_AGENT: frozenset({"prepare_package"}),
    FILING_AGENT: frozenset({"file_disposition"}),
}
CHAIN_ACTION = {
    INVESTIGATION_AGENT: "investigate",
    CASE_AGENT: "prepare_package",
    FILING_AGENT: "file_disposition",
}


def _make_alert(index: int, rng: random.Random) -> Alert:
    account_id = f"ACCT-{index + 1000}"
    start = datetime(2026, 1, 1) + timedelta(days=index * 2)
    is_structuring = index % 3 != 1
    if is_structuring:
        amounts = [
            round(rng.uniform(7_700, 9_800), -2),
            round(rng.uniform(8_100, 9_700), -2),
            round(rng.uniform(8_400, 9_800), -2),
            round(rng.uniform(7_900, 9_600), -2),
        ]
        transactions = tuple(
            Transaction(f"T-{index + 1:02d}-{position}", account_id, amount, "cash_deposit",
                        start + timedelta(days=position - 1), f"BR-{(index % 4) + 1:02d}")
            for position, amount in enumerate(amounts, start=1)
        )
        rule = "multiple sub-threshold cash deposits"
    else:
        transactions = (
            Transaction(f"T-{index + 1:02d}-1", account_id, 12_500.0, "cash_deposit", start, "BR-01"),
            Transaction(f"T-{index + 1:02d}-2", account_id, 1_400.0, "card_payment", start + timedelta(days=1), "BR-01"),
        )
        rule = "cash activity exceeds expected profile"
    customer = CustomerProfile(
        account_id,
        f"Customer {index + 1:02d}",
        "small business owner" if is_structuring else "software consultant",
        8_000.0 if is_structuring else 5_000.0,
    )
    return Alert(f"AML-{index + 1:03d}", account_id, transactions, customer, rule)


def generate_alerts(count: int = 40, seed: int = 7) -> list[Alert]:
    """Generate reproducible synthetic alerts; no expected disposition is stored."""
    rng = random.Random(seed)
    return [_make_alert(index, rng) for index in range(count)]


def _disposition_for(amounts: tuple[float, ...]) -> Disposition:
    """A local, records-free decision rule the agents apply to whatever values they hold."""
    aggregate = sum(amounts)
    return "FILE_SAR" if (len(amounts) >= 3 and aggregate >= 30_000) else "CLEAR"


def build_chain(alert: Alert, corrupt: bool, fault: str | None = None) -> tuple[ChainStep, ...]:
    """Run one alert through investigation -> case -> filing.

    Only the investigation agent reads the raw records. Faults planted here are all
    "authorized but wrong": every action stays within permissions (except the
    scope fault, which is deliberately out-of-scope so the security lane can catch
    that one). Supported `fault` values:
      * None                  - clean chain
      * "transitive-corruption" - investigation misreads a value; it propagates
      * "wrong-target"        - filing acts on a different account than the alert
      * "scope"               - filing takes an action outside its allowed_actions
      * "conflict"            - case and filing reach contradictory dispositions
    The legacy ``corrupt`` flag maps to fault="transitive-corruption".
    """
    if corrupt and fault is None:
        fault = "transitive-corruption"

    truth = recompute_structuring(alert)
    acct = alert.account_id

    # Step 1: investigation reads the records (correctly, or misreads if corrupt).
    if fault == "transitive-corruption" and truth.qualifying_amounts:
        used = tuple(2_900.0 if a == truth.qualifying_amounts[0] else a * 0.5 for a in truth.qualifying_amounts)
    else:
        used = truth.qualifying_amounts
    s1 = ChainStep(
        agent=INVESTIGATION_AGENT, step_index=1, used_amounts=used, used_aggregate=sum(used),
        disposition=_disposition_for(used), action_type=CHAIN_ACTION[INVESTIGATION_AGENT],
        allowed_actions=CHAIN_ALLOWED[INVESTIGATION_AGENT], received_from=None, received_amounts=None,
        acted_account_id=acct,
    )
    # Step 2: case trusts investigation's finding (does not re-read records).
    case_disposition = s1.disposition
    s2 = ChainStep(
        agent=CASE_AGENT, step_index=2, used_amounts=s1.used_amounts, used_aggregate=s1.used_aggregate,
        disposition=case_disposition, action_type=CHAIN_ACTION[CASE_AGENT],
        allowed_actions=CHAIN_ALLOWED[CASE_AGENT], received_from=INVESTIGATION_AGENT, received_amounts=s1.used_amounts,
        acted_account_id=acct,
    )
    # Step 3: filing trusts case's package - unless a fault diverts it.
    filing_account = acct
    filing_action = CHAIN_ACTION[FILING_AGENT]
    filing_disposition = s2.disposition
    if fault == "wrong-target":
        filing_account = "ACCT-0099"          # authorized to file, but on the wrong account
    elif fault == "scope":
        filing_action = "freeze_account"       # outside filing's allowed_actions (security WILL catch this)
    elif fault == "conflict":
        # filing contradicts the case package with no reconciliation
        filing_disposition = "FILE_SAR" if s2.disposition == "CLEAR" else "CLEAR"
    s3 = ChainStep(
        agent=FILING_AGENT, step_index=3, used_amounts=s2.used_amounts, used_aggregate=s2.used_aggregate,
        disposition=filing_disposition, action_type=filing_action,
        allowed_actions=CHAIN_ALLOWED[FILING_AGENT], received_from=CASE_AGENT, received_amounts=s2.used_amounts,
        acted_account_id=filing_account,
    )
    return (s1, s2, s3)


def _fault_plan(count: int) -> dict[int, str]:
    """Plant one instance of each fault class at distinct indices; the rest run clean.

    Indices are clamped so they always land inside the session and never collide.
    The star (transitive-corruption) keeps its original index for continuity.
    """
    star = min(6, count - 1)
    plan: dict[int, str] = {star: "transitive-corruption"}
    for idx, cls in [(min(10, count - 1), "wrong-target"),
                     (min(14, count - 1), "scope"),
                     (min(26, count - 1), "conflict")]:
        if idx not in plan:      # don't overwrite an earlier planting on tiny sessions
            plan[idx] = cls
    return plan


def run_chain_session(count: int = 40, seed: int = 7) -> list[ChainOutcome]:
    """Generate a long-running session of 3-agent chains with one planted instance
    of each fault class (transitive-corruption, wrong-target, scope, conflict).
    All other chains run clean."""
    alerts = generate_alerts(count, seed)
    plan = _fault_plan(count)
    star_index = min(6, count - 1)
    outcomes: list[ChainOutcome] = []
    for index, alert in enumerate(alerts):
        fault = plan.get(index)
        corrupt = (fault == "transitive-corruption")
        chain = build_chain(alert, corrupt, fault)
        outcomes.append(ChainOutcome(alert=alert, chain=chain, corrupt=corrupt,
                                     is_star_case=(index == star_index), fault=fault))
    return outcomes