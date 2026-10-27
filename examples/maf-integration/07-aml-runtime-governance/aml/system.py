"""Declared AML reference pipeline and record codec."""

from __future__ import annotations

from datetime import datetime

from domain import Alert, CustomerProfile, Transaction
from system_description import AgentDescription, SystemDescription


TRIAGE_AGENT = "intake"
SANCTIONS_AGENT = "customer-screening"
TRANSACTION_AGENT = "transaction-analysis"
KYC_AGENT = "kyc-entity"
DISPOSITION_AGENT = "decision"

DIAMOND_NODE_ID = {
    TRIAGE_AGENT: "A",
    SANCTIONS_AGENT: "B",
    TRANSACTION_AGENT: "C",
    KYC_AGENT: "D",
    DISPOSITION_AGENT: "E",
}
DIAMOND_BRANCHES = (SANCTIONS_AGENT, TRANSACTION_AGENT, KYC_AGENT)
DIAMOND_PARENTS = {
    TRIAGE_AGENT: (),
    SANCTIONS_AGENT: (TRIAGE_AGENT,),
    TRANSACTION_AGENT: (TRIAGE_AGENT,),
    KYC_AGENT: (TRIAGE_AGENT,),
    DISPOSITION_AGENT: DIAMOND_BRANCHES,
}
DIAMOND_ALLOWED = {
    TRIAGE_AGENT: frozenset({"fan_out_alert"}),
    SANCTIONS_AGENT: frozenset({"screen_sanctions"}),
    TRANSACTION_AGENT: frozenset({"analyze_transactions", "review_records"}),
    KYC_AGENT: frozenset({"verify_kyc"}),
    DISPOSITION_AGENT: frozenset({"merge_disposition"}),
}
DIAMOND_ACTION = {
    TRIAGE_AGENT: "fan_out_alert",
    SANCTIONS_AGENT: "screen_sanctions",
    TRANSACTION_AGENT: "analyze_transactions",
    KYC_AGENT: "verify_kyc",
    DISPOSITION_AGENT: "merge_disposition",
}


def encode_alert(alert: object) -> dict[str, object]:
    if not isinstance(alert, Alert):
        raise TypeError("AML trace records must be an Alert")
    return {
        "alert_id": alert.alert_id,
        "account_id": alert.account_id,
        "monitoring_rule": alert.monitoring_rule,
        "customer": {
            "account_id": alert.customer.account_id,
            "name": alert.customer.name,
            "stated_occupation": alert.customer.stated_occupation,
            "expected_monthly_cash_volume": alert.customer.expected_monthly_cash_volume,
        },
        "transactions": [{
            "id": transaction.id,
            "account_id": transaction.account_id,
            "amount": transaction.amount,
            "type": transaction.type,
            "timestamp": transaction.timestamp.isoformat(),
            "branch": transaction.branch,
        } for transaction in alert.transactions],
    }


def decode_alert(value: object) -> Alert:
    if not isinstance(value, dict):
        raise TypeError("AML graph context must contain an encoded alert")
    customer = value["customer"]
    return Alert(
        value["alert_id"],
        value["account_id"],
        tuple(
            Transaction(
                transaction["id"],
                transaction["account_id"],
                float(transaction["amount"]),
                transaction["type"],
                datetime.fromisoformat(transaction["timestamp"]),
                transaction["branch"],
            )
            for transaction in value["transactions"]
        ),
        CustomerProfile(
            customer["account_id"],
            customer["name"],
            customer["stated_occupation"],
            float(customer["expected_monthly_cash_volume"]),
        ),
        value["monitoring_rule"],
    )


AML_SYSTEM_DESCRIPTION = SystemDescription(
    name="aml-diamond",
    schema="aml-diamond/v1",
    agents=tuple(
        AgentDescription(agent, DIAMOND_NODE_ID[agent], DIAMOND_ALLOWED[agent])
        for agent in DIAMOND_NODE_ID
    ),
    topology=DIAMOND_PARENTS,
    final_agent=DISPOSITION_AGENT,
    final_action="commit_disposition",
    resolver_action="resolve_behavioral_correctness",
    handoff_action="aml_graph_handoff",
    resource_prefix="aml-alert",
    escalation_outcomes=frozenset({"ESCALATE"}),
    encode_records=encode_alert,
    decode_records=decode_alert,
)


__all__ = [
    "AML_SYSTEM_DESCRIPTION",
    "DIAMOND_ACTION",
    "DIAMOND_ALLOWED",
    "DIAMOND_BRANCHES",
    "DIAMOND_NODE_ID",
    "DIAMOND_PARENTS",
    "DISPOSITION_AGENT",
    "KYC_AGENT",
    "SANCTIONS_AGENT",
    "TRANSACTION_AGENT",
    "TRIAGE_AGENT",
]
