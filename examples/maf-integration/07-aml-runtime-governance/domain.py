"""Domain models and deterministic AML structuring calculations.

Includes both the single-decision (2-agent) model and the 3-agent delegation
chain used to demonstrate transitive corruption with origin attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    acted_account_id: str | None = None   # the account this hop acted on (for wrong-target)


@dataclass(frozen=True)
class ChainOutcome:
    """Result of running one alert through the 3-agent chain."""

    alert: Alert
    chain: tuple[ChainStep, ...]
    corrupt: bool
    is_star_case: bool = False
    fault: str | None = None   # which fault class was planted, if any


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


# ---------------------------------------------------------------------------
# Step-gated execution (runtime halt): produce a step, check it, gate the next.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GatedStep:
    """One step in a step-gated run, with its execution status.

    status is one of:
      * "executed"          - the step ran and passed its governance check
      * "blocked_at_gate"   - the step's action was denied before it ran
                              (permission/scope fault; the action never executed)
      * "halted_after"      - the step ran, its output failed the check, and the
                              chain was halted (this step is the point of detection)
      * "not_run"           - a downstream step that never executed because the
                              chain was already halted/blocked upstream
    """

    step: "ChainStep | None"           # None only for a gate-blocked action we refused to build
    agent: str
    step_index: int
    status: str
    reason: str


@dataclass(frozen=True)
class GatedRun:
    """Result of running one alert through the step-gated chain."""

    alert: "Alert"
    steps: tuple[GatedStep, ...]
    halted: bool                        # True if the chain was stopped early
    halt_step_index: int | None        # where it stopped (gate or post-step)
    halt_kind: str | None              # "gate" (pre-action deny) or "behavioral" (post-step halt)
    category: str | None               # the fault category that triggered the halt
    reason: str
    fault: str | None                  # the planted fault label (for the demo)
    consequential_prevented: bool      # True if a harmful downstream step was stopped

# ---------------------------------------------------------------------------
# Graph topology (fan-out / fan-in diamond). Additive: the linear ChainStep path
# above is unchanged. A GraphStep may have several parents, so fan-in provenance
# is kept per parent rather than as a single received_from string.
# ---------------------------------------------------------------------------
SANCTIONS_WATCHLIST = frozenset({"Viktor Petrov Holdings", "Al-Noor Exchange", "Northwind Shell Co"})


@dataclass(frozen=True)
class GraphStep:
    """One node in a fan-out/fan-in investigation graph."""

    agent: str
    node_id: str                         # "A".."E" (display label)
    action_type: str
    allowed_actions: frozenset[str]
    received_from: tuple[str, ...]       # upstream agents (empty for the root)
    finding: str                         # what this node reported, e.g. "no-hit", "not-structuring"
    flagged: bool                        # does this node's finding raise a concern?
    used_amounts: tuple[float, ...] = ()
    used_aggregate: float = 0.0
    disposition: Disposition | None = None    # only the merge node disposes
    acted_account_id: str | None = None
    received_amounts_by_parent: dict[str, tuple[float, ...]] = field(default_factory=dict)
    received_findings_by_parent: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphOutcome:
    """Result of running one alert through the diamond graph."""

    alert: Alert
    steps: tuple[GraphStep, ...]
    is_star_case: bool = False
    fault: str | None = None

    def step(self, agent: str) -> GraphStep:
        return next(s for s in self.steps if s.agent == agent)

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        return tuple((parent, s.agent) for s in self.steps for parent in s.received_from)


@dataclass(frozen=True)
class BranchFinding:
    """A branch's correct finding, recomputed from the alert records alone."""

    finding: str
    flagged: bool
    detail: str


def recompute_sanctions(alert: Alert) -> BranchFinding:
    hit = alert.customer.name in SANCTIONS_WATCHLIST
    return BranchFinding("hit" if hit else "no-hit", hit,
                         f"'{alert.customer.name}' {'is' if hit else 'is not'} on the sanctions watchlist")


def recompute_kyc(alert: Alert) -> BranchFinding:
    c = alert.customer
    complete = bool(c.name and c.stated_occupation and c.expected_monthly_cash_volume > 0)
    return BranchFinding("verified" if complete else "incomplete", not complete,
                         f"profile on file: {c.stated_occupation}, expected ${c.expected_monthly_cash_volume:,.0f}/mo cash")


def recompute_transactions(alert: Alert) -> BranchFinding:
    truth = recompute_structuring(alert)
    return BranchFinding("structuring" if truth.is_structuring else "not-structuring", truth.is_structuring,
                         f"{truth.qualifying_count} sub-threshold cash deposits totalling ${truth.aggregate:,.0f}")


def merge_disposition(sanctions_flagged: bool, transactions_flagged: bool, kyc_flagged: bool) -> Disposition:
    """The fan-in merge rule the disposition agent applies to its branch inputs."""
    if sanctions_flagged or transactions_flagged:
        return "FILE_SAR"
    if kyc_flagged:
        return "ESCALATE"
    return "CLEAR"


@dataclass(frozen=True)
class BranchCheck:
    """Record-relative check of one node in the graph."""

    agent: str
    node_id: str
    reported: str
    recomputed: str
    matches: bool
    detail: str


@dataclass(frozen=True)
class GraphAttribution:
    """Governance verdict over a diamond, with branch-level attribution."""

    passed: bool
    category: str | None
    origin_agent: str | None
    origin_node: str | None
    masked_by: tuple[str, ...]           # sibling branches whose clean results hid the bad one
    reason: str
    branch_checks: tuple[BranchCheck, ...]
    merged_disposition: str
    expected_disposition: str
