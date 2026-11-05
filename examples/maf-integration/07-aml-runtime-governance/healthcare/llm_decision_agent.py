"""Real-LLM healthcare care-decision agent with a deliberately small context window.

Only the care-decision agent is an LLM; the review agents stay deterministic.
The generic machinery lives in ``llm_decision``; this module supplies the care
file: the allergy-safety report arrives first, then the routine care
coordination log, then the medication and coverage reports and the decision
task. When the care file is longer than ``num_ctx``, Ollama keeps only the most
recent tokens, so the early allergy report genuinely falls out of the agent's
context and a contraindication can be missed.

Validated settings: llama3.1:8b, ``num_ctx=1024``, temperature 0. The smaller
qwen2.5:3b drops or garbles these longer statuses even with the full care file
in view, so it is not a reliable full-context control here.
"""

from __future__ import annotations

from typing import Mapping

from healthcare.system import ALLERGY_AGENT, COVERAGE_AGENT, MEDICATION_AGENT, HealthcareCase
from llm_decision import DecisionSpec, LLMDecision, LLMDecisionConfig, ReportSpec
from llm_decision import build_case_file as _build_case_file
from llm_decision import parse_decision_output as _parse_decision_output
from llm_decision import run_llm_decision as _run_llm_decision
from system_description import TraceStep

DISPOSITIONS = ("APPROVE_TREATMENT", "HOLD_FOR_CLINICAL_REVIEW", "HOLD_FOR_AUTHORIZATION")


def _detail(step: TraceStep, case: HealthcareCase) -> str:
    """Descriptive detail built only from what the review emitted (and the case it ran on)."""
    if step.agent == ALLERGY_AGENT:
        relation = "conflicts with" if step.flagged else "has no conflict with"
        return f"{case.medication} {relation} the recorded allergy ({case.known_allergy})"
    if step.agent == MEDICATION_AGENT:
        return f"{case.medication} ordered for {case.diagnosis}"
    return f"prior authorization {'not confirmed' if step.flagged else 'confirmed'} for {case.medication}"


HEALTHCARE_DECISION_SPEC = DecisionSpec(
    file_title="CARE FILE",
    agent_role="the care-decision agent",
    decision_noun="care decision",
    reports=(
        ReportSpec(ALLERGY_AGENT, "allergy", "Allergy safety", early=True),
        ReportSpec(MEDICATION_AGENT, "medication", "Medication review", early=False),
        ReportSpec(COVERAGE_AGENT, "coverage", "Coverage review", early=False),
    ),
    rules=(
        "HOLD_FOR_CLINICAL_REVIEW if allergy safety is CONTRAINDICATION; "
        "HOLD_FOR_AUTHORIZATION if coverage is NOT-AUTHORIZED; otherwise APPROVE_TREATMENT. "
        "In the JSON, report the STATUS of every report that appears, including reports these rules do not use; "
        "copy each STATUS exactly as written in its report"
    ),
    status_hint="<STATUS copied exactly from that report, or not-present>",
    dispositions=DISPOSITIONS,
    log_title="[CARE COORDINATION LOG]",
    log_events=(
        "vitals recorded; within normal range",
        "nursing note filed; no change in condition",
        "pharmacy queue updated; order awaiting decision",
        "bed assignment confirmed",
        "patient education materials provided",
        "care team handoff completed",
    ),
    record_id=lambda case: case.case_id,
    detail=_detail,
    log_ref=lambda case, index: f" (patient {case.patient_id})",
)


ENV_PREFIX = "HEALTHCARE_LLM"
DEFAULT_MODEL = "llama3.1:8b"


def default_config(**overrides) -> LLMDecisionConfig:
    """Care-decision agent settings from ``HEALTHCARE_LLM_MODEL`` / ``HEALTHCARE_LLM_NUM_CTX``."""
    return LLMDecisionConfig.from_env(ENV_PREFIX, DEFAULT_MODEL, **overrides)


def build_case_file(
    case: HealthcareCase, branches: Mapping[str, TraceStep], config: LLMDecisionConfig
) -> tuple[str, dict[str, str]]:
    """The care-decision agent's input, plus each report's line for window measurement."""
    return _build_case_file(HEALTHCARE_DECISION_SPEC, case, branches, config)


def parse_decision_output(text: str) -> tuple[str, dict[str, str]]:
    """Parse the care-decision agent's reply into ``(disposition, self-reported statuses)``."""
    return _parse_decision_output(text, HEALTHCARE_DECISION_SPEC)


def run_llm_decision(
    case: HealthcareCase,
    branches: Mapping[str, TraceStep],
    config: LLMDecisionConfig | None = None,
) -> LLMDecision:
    """Run the care-decision agent as a real Ollama call and record what it saw vs. decided."""
    return _run_llm_decision(HEALTHCARE_DECISION_SPEC, case, branches, config or default_config())


__all__ = [
    "DISPOSITIONS",
    "HEALTHCARE_DECISION_SPEC",
    "build_case_file",
    "default_config",
    "parse_decision_output",
    "run_llm_decision",
]
