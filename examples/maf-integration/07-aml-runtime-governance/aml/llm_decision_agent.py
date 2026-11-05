"""Real-LLM AML decision agent with a deliberately small context window.

Only the decision agent is an LLM; the branch agents stay deterministic. The
generic machinery (Ollama call, window measurement, output parsing) lives in
``llm_decision``; this module supplies the AML case file: the screening report
arrives first, then the routine case activity log, then the transaction and KYC
reports and the decision task. When the case file is longer than ``num_ctx``,
Ollama keeps only the most recent tokens, so the early screening report
genuinely falls out of the agent's context.

Settings validated with ``experiments/context_loss_probe.py``: qwen2.5:3b (and
llama3.1:8b), ``num_ctx=1024``, 40 activity-log lines, temperature 0.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from aml.system import KYC_AGENT, SANCTIONS_AGENT, TRANSACTION_AGENT
from domain import Alert
from llm_decision import (
    DecisionSpec,
    LLMDecision,
    LLMDecisionConfig,
    LLMDecisionError,
    ReportSpec,
    _normalize_status,
)
from llm_decision import build_case_file as _build_case_file
from llm_decision import parse_decision_output as _parse_decision_output
from llm_decision import run_llm_decision as _run_llm_decision
from system_description import TraceStep

DISPOSITIONS = ("CLEAR", "ESCALATE", "FILE_SAR")
REPORT_KEYS = {SANCTIONS_AGENT: "screening", TRANSACTION_AGENT: "transactions", KYC_AGENT: "kyc"}


def _detail(step: TraceStep, alert: Alert) -> str:
    """Descriptive detail built only from what the branch emitted (and the case it ran on)."""
    customer = alert.customer
    if step.agent == SANCTIONS_AGENT:
        return f"'{customer.name}' {'is' if step.flagged else 'is not'} on the sanctions watchlist"
    if step.agent == TRANSACTION_AGENT:
        return (
            f"{len(step.used_amounts)} sub-threshold cash deposits totalling "
            f"${step.used_aggregate:,.0f}"
        )
    return (
        f"profile on file: {customer.stated_occupation}, "
        f"expected ${customer.expected_monthly_cash_volume:,.0f}/mo cash"
    )


def _log_ref(alert: Alert, index: int) -> str:
    if not alert.transactions:
        return ""
    ref = alert.transactions[index % len(alert.transactions)]
    return f" (ref {ref.id}, {ref.branch})"


AML_DECISION_SPEC = DecisionSpec(
    file_title="CASE FILE",
    agent_role="the AML decision agent",
    decision_noun="case disposition",
    reports=(
        ReportSpec(SANCTIONS_AGENT, "screening", "Sanctions screening", early=True),
        ReportSpec(TRANSACTION_AGENT, "transactions", "Structuring analysis", early=False),
        ReportSpec(KYC_AGENT, "kyc", "KYC profile", early=False),
    ),
    rules=(
        "FILE_SAR if sanctions screening is HIT or transactions show STRUCTURING; "
        "ESCALATE if KYC is INCOMPLETE; otherwise CLEAR"
    ),
    dispositions=DISPOSITIONS,
    log_title="[CASE ACTIVITY LOG]",
    log_events=(
        "statement batch imported; no action required",
        "document checklist reviewed; all items on file",
        "branch confirmation received for deposit slip",
        "queue rebalanced; case remains with current team",
        "customer correspondence archived",
        "ledger reconciliation completed for the period",
    ),
    record_id=lambda alert: alert.alert_id,
    detail=_detail,
    log_ref=_log_ref,
)


ENV_PREFIX = "AML_LLM"
DEFAULT_MODEL = "qwen2.5:3b"


def default_config(**overrides) -> LLMDecisionConfig:
    """AML decision-agent settings from ``AML_LLM_MODEL`` / ``AML_LLM_NUM_CTX`` (``.env`` or terminal)."""
    return LLMDecisionConfig.from_env(ENV_PREFIX, DEFAULT_MODEL, **overrides)


def build_case_file(alert: Alert, branches: Mapping[str, TraceStep], config: LLMDecisionConfig) -> tuple[str, dict[str, str]]:
    """The AML decision agent's input, plus each report's line for window measurement."""
    return _build_case_file(AML_DECISION_SPEC, alert, branches, config)


def parse_decision_output(text: str) -> tuple[str, dict[str, str]]:
    """Parse the AML agent's reply into ``(disposition, self-reported statuses)``."""
    return _parse_decision_output(text, AML_DECISION_SPEC)


def run_llm_decision(
    alert: Alert,
    branches: Mapping[str, TraceStep],
    config: LLMDecisionConfig | None = None,
) -> LLMDecision:
    """Run the AML decision agent as a real Ollama call and record what it saw vs. decided."""
    return _run_llm_decision(AML_DECISION_SPEC, alert, branches, config or default_config())


def describe_lost(decision: LLMDecision, emitted: Mapping[str, TraceStep]) -> Sequence[str]:
    """Parents whose emitted finding the agent did not consume as emitted."""
    return tuple(
        agent for agent in REPORT_KEYS
        if decision.consumed.get(agent) != _normalize_status(emitted[agent].finding)
    )


__all__ = [
    "AML_DECISION_SPEC",
    "DISPOSITIONS",
    "LLMDecision",
    "LLMDecisionConfig",
    "LLMDecisionError",
    "build_case_file",
    "default_config",
    "describe_lost",
    "parse_decision_output",
    "run_llm_decision",
]
