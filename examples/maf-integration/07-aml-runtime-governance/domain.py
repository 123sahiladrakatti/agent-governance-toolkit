"""Domain models and deterministic AML structuring calculations.

Includes both the single-decision (2-agent) model and the 3-agent delegation
chain used to demonstrate transitive corruption with origin attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

CTR_THRESHOLD = 10_000.0
STRUCTURING_MIN_DEPOSITS = 3
STRUCTURING_AGGREGATE = 30_000.0
STRUCTURING_WINDOW_DAYS = 5

Disposition = Literal["CLEAR", "ESCALATE", "FILE_SAR"]


@dataclass(frozen=True)
class Transaction:
    id: str
    account_id: str
    amount: float
    type: str
    timestamp: datetime
    branch: str


@dataclass(frozen=True)
class CustomerProfile:
    account_id: str
    name: str
    stated_occupation: str
    expected_monthly_cash_volume: float


@dataclass(frozen=True)
class Alert:
    alert_id: str
    account_id: str
    transactions: tuple[Transaction, ...]
    customer: CustomerProfile
    monitoring_rule: str


@dataclass(frozen=True)
class Delegation:
    sender: str
    recipient: str
    account_id: str
    goal: str
    allowed_actions: frozenset[str]
    data_access_scope: str = "portfolio"


@dataclass(frozen=True)
class AgentAction:
    alert_id: str
    step: str
    responsible_agent: str
    delegation: Delegation
    acted_account_id: str
    action_type: str
    disposition: Disposition
    disposition_status: str
    rationale: str
    claimed_amounts: tuple[float, ...]
    claimed_threshold: float
    claimed_aggregate: float
    evidence_transactions: tuple[str, ...]
    sar_record_id: str | None = None
    is_star_case: bool = False
    fault_label: str | None = None


@dataclass(frozen=True)
class StructuringResult:
    qualifying_amounts: tuple[float, ...]
    qualifying_transaction_ids: tuple[str, ...]
    aggregate: float
    threshold: float
    qualifying_count: int
    within_window: bool
    is_structuring: bool


def recompute_structuring(alert: Alert, threshold: float = CTR_THRESHOLD) -> StructuringResult:
    """Derive structuring evidence only from the alert's transaction records."""
    cash_deposits = [
        transaction for transaction in alert.transactions
        if transaction.type == "cash_deposit" and transaction.amount < threshold
    ]
    cash_deposits.sort(key=lambda transaction: transaction.timestamp)
    within_window = bool(cash_deposits) and any(
        transaction.timestamp - cash_deposits[0].timestamp <= timedelta(days=STRUCTURING_WINDOW_DAYS)
        for transaction in cash_deposits
    )
    amounts = tuple(transaction.amount for transaction in cash_deposits)
    transaction_ids = tuple(transaction.id for transaction in cash_deposits)
    aggregate = sum(amounts)
    return StructuringResult(
        qualifying_amounts=amounts,
        qualifying_transaction_ids=transaction_ids,
        aggregate=aggregate,
        threshold=threshold,
        qualifying_count=len(amounts),
        within_window=within_window,
        is_structuring=(len(amounts) >= STRUCTURING_MIN_DEPOSITS and aggregate >= STRUCTURING_AGGREGATE and within_window),
    )


@dataclass(frozen=True)
class GovernanceVerdict:
    passed: bool
    category: str | None
    responsible_agent: str
    step: str
    reason: str
    details: dict[str, object]


@dataclass(frozen=True)
class AlertOutcome:
    alert: Alert
    action: AgentAction
    regular: GovernanceVerdict
    enhanced: GovernanceVerdict


# ---------------------------------------------------------------------------
# 3-agent delegation chain (transitive corruption + origin attribution)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChainStep:
    """One agent's hop in a delegation chain.

    Each step records the value the agent *used*, the value it *received* from
    upstream, and who it received from. This provenance is what lets governance
    walk the chain backward and attribute a corrupted value to its origin.
    """

    agent: str
    step_index: int              # 1-based position in the chain
    used_amounts: tuple[float, ...]
    used_aggregate: float
    disposition: Disposition
    action_type: str             # the permitted action this agent took
    allowed_actions: frozenset[str]
    received_from: str | None    # upstream agent trusted (None for the record reader)
    received_amounts: tuple[float, ...] | None


@dataclass(frozen=True)
class ChainOutcome:
    """Result of running one alert through the 3-agent chain."""

    alert: Alert
    chain: tuple[ChainStep, ...]
    corrupt: bool
    is_star_case: bool = False


@dataclass(frozen=True)
class SecurityCheck:
    """Per-hop deterministic authorization result (the 'security intact' lane)."""

    agent: str
    step_index: int
    passed: bool
    reason: str


@dataclass(frozen=True)
class ChainAttribution:
    """Governance verdict over a whole chain, with origin vs propagators."""

    passed: bool
    category: str | None
    origin_agent: str | None
    origin_step: int | None
    propagators: tuple[str, ...]
    reason: str
    record_amounts: tuple[float, ...]
    record_aggregate: float
    is_structuring: bool
    final_disposition: str
    expected_disposition: str


@dataclass(frozen=True)
class ChainReview:
    """Everything the UI needs for one chain: hops, security lane, governance lane."""

    outcome: ChainOutcome
    security: tuple[SecurityCheck, ...]
    governance: ChainAttribution