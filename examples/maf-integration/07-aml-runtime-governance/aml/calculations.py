"""Deterministic AML calculations used by the reference implementation."""

from __future__ import annotations

from datetime import timedelta

from domain import (
    Alert,
    BranchFinding,
    CTR_THRESHOLD,
    Disposition,
    SANCTIONS_WATCHLIST,
    STRUCTURING_AGGREGATE,
    STRUCTURING_MIN_DEPOSITS,
    STRUCTURING_WINDOW_DAYS,
    StructuringResult,
)


def recompute_structuring(alert: Alert, threshold: float = CTR_THRESHOLD) -> StructuringResult:
    cash_deposits = [
        transaction for transaction in alert.transactions
        if transaction.type == "cash_deposit" and transaction.amount < threshold
    ]
    cash_deposits.sort(key=lambda transaction: transaction.timestamp)
    within_window = bool(cash_deposits) and (
        cash_deposits[-1].timestamp - cash_deposits[0].timestamp
        <= timedelta(days=STRUCTURING_WINDOW_DAYS)
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
        is_structuring=(
            len(amounts) >= STRUCTURING_MIN_DEPOSITS
            and aggregate >= STRUCTURING_AGGREGATE
            and within_window
        ),
    )


def recompute_sanctions(alert: Alert) -> BranchFinding:
    hit = alert.customer.name in SANCTIONS_WATCHLIST
    return BranchFinding(
        "hit" if hit else "no-hit",
        hit,
        f"'{alert.customer.name}' {'is' if hit else 'is not'} on the sanctions watchlist",
    )


def recompute_kyc(alert: Alert) -> BranchFinding:
    customer = alert.customer
    complete = bool(
        customer.name
        and customer.stated_occupation
        and customer.expected_monthly_cash_volume > 0
    )
    return BranchFinding(
        "verified" if complete else "incomplete",
        not complete,
        f"profile on file: {customer.stated_occupation}, "
        f"expected ${customer.expected_monthly_cash_volume:,.0f}/mo cash",
    )


def recompute_transactions(alert: Alert) -> BranchFinding:
    truth = recompute_structuring(alert)
    return BranchFinding(
        "structuring" if truth.is_structuring else "not-structuring",
        truth.is_structuring,
        f"{truth.qualifying_count} sub-threshold cash deposits totalling ${truth.aggregate:,.0f}",
    )


def merge_disposition(
    sanctions_flagged: bool,
    transactions_flagged: bool,
    kyc_flagged: bool,
) -> Disposition:
    if sanctions_flagged or transactions_flagged:
        return "FILE_SAR"
    if kyc_flagged:
        return "ESCALATE"
    return "CLEAR"


__all__ = [
    "merge_disposition",
    "recompute_kyc",
    "recompute_sanctions",
    "recompute_structuring",
    "recompute_transactions",
]
