"""Simulated healthcare client agents and client-pipeline payload helpers."""

from __future__ import annotations

from typing import Any

from client_pipeline import RecordCodec, run_to_json
from domain import GraphStep
from healthcare.system import (
    ALLERGY_AGENT,
    CARE_DECISION_AGENT,
    COVERAGE_AGENT,
    HEALTHCARE_ALLOWED,
    HEALTHCARE_NODE_ID,
    HEALTHCARE_PARENTS,
    HEALTHCARE_SYSTEM_DESCRIPTION,
    HealthcareCase,
    INTAKE_AGENT,
    MEDICATION_AGENT,
    MEDICATION_ALLERGENS,
    decode_healthcare_case,
    encode_healthcare_case,
)


HEALTHCARE_RECORD_CODECS: dict[str, RecordCodec] = {
    HEALTHCARE_SYSTEM_DESCRIPTION.schema: (encode_healthcare_case, decode_healthcare_case),
}

try:
    # The healthcare agents run as a LangGraph StateGraph (deterministic nodes, no LLM).
    import langgraph.graph  # noqa: F401

    HEALTHCARE_RUN_ENGINE = "langgraph"
except ImportError:  # langgraph not installed: fall back to the hand-rolled reference steps
    HEALTHCARE_RUN_ENGINE = "build_healthcare_steps"

# Deterministic branch outputs shared by the reference steps and the LangGraph nodes.
INTAKE_FINDING = "care-request-routed"
MEDICATION_FINDING = "medication-order-validated"
# The case-state allergy entry before the safety agent's update (version 1).
PRIOR_ALLERGY_STATE = {"finding": "no-contraindication", "version": 1}


def allergy_finding(case: HealthcareCase) -> tuple[str, bool]:
    allergen = MEDICATION_ALLERGENS.get(case.medication)
    conflict = bool(allergen and case.known_allergy.casefold() == allergen.casefold())
    return ("contraindication", True) if conflict else ("no-contraindication", False)


def expected_disposition(case: HealthcareCase) -> str:
    if allergy_finding(case)[1]:
        return "HOLD_FOR_CLINICAL_REVIEW"
    if not case.coverage_authorized:
        return "HOLD_FOR_AUTHORIZATION"
    return "APPROVE_TREATMENT"


def allergy_version(flagged: bool) -> int:
    return 2 if flagged else 1


def coverage_finding(case: HealthcareCase) -> str:
    return "authorized" if case.coverage_authorized else "not-authorized"


def care_disposition(consumed_allergy: str, coverage: str) -> str:
    """Care-decision merge rule applied to whatever allergy/coverage view the agent holds."""
    if consumed_allergy == "contraindication":
        return "HOLD_FOR_CLINICAL_REVIEW"
    if coverage == "not-authorized":
        return "HOLD_FOR_AUTHORIZATION"
    return "APPROVE_TREATMENT"


def build_healthcare_steps(
    case: HealthcareCase,
    stale_allergy_view: bool = False,
) -> tuple[GraphStep, ...]:
    """Hand-rolled reference run of the healthcare diamond (parity reference and fallback).

    When ``stale_allergy_view`` is true, the safety agent emits its current v2
    contraindication but the final agent consumes an earlier v1 no-hit result.
    The governance framework detects this from provenance/version metadata.
    """

    allergy, allergy_flagged = allergy_finding(case)
    current_allergy_version = allergy_version(allergy_flagged)
    coverage = coverage_finding(case)
    intake_finding = INTAKE_FINDING
    medication_finding = MEDICATION_FINDING

    consumed_allergy = PRIOR_ALLERGY_STATE["finding"] if stale_allergy_view else allergy
    consumed_allergy_version = PRIOR_ALLERGY_STATE["version"] if stale_allergy_view else current_allergy_version
    disposition = care_disposition(consumed_allergy, coverage)

    def step(agent: str, action: str, finding: str, flagged: bool = False, **kwargs) -> GraphStep:
        return GraphStep(
            agent=agent,
            node_id=HEALTHCARE_NODE_ID[agent],
            action_type=action,
            allowed_actions=HEALTHCARE_ALLOWED[agent],
            received_from=HEALTHCARE_PARENTS[agent],
            finding=finding,
            flagged=flagged,
            acted_account_id=case.patient_id,
            **kwargs,
        )

    steps = (
        step(INTAKE_AGENT, "route_care_request", intake_finding),
        step(ALLERGY_AGENT, "evaluate_allergy_safety", allergy, allergy_flagged,
             finding_version=current_allergy_version),
        step(MEDICATION_AGENT, "validate_medication_order", medication_finding),
        step(COVERAGE_AGENT, "verify_coverage", coverage, not case.coverage_authorized),
        step(
            CARE_DECISION_AGENT,
            "merge_care_decision",
            f"care-decision:{disposition.lower()}",
            disposition != "APPROVE_TREATMENT",
            disposition=disposition,
            received_findings_by_parent={
                ALLERGY_AGENT: consumed_allergy,
                MEDICATION_AGENT: medication_finding,
                COVERAGE_AGENT: coverage,
            },
            received_versions_by_parent={
                ALLERGY_AGENT: consumed_allergy_version,
                MEDICATION_AGENT: 1,
                COVERAGE_AGENT: 1,
            },
        ),
    )
    return steps


def simulate_healthcare_case(
    case: HealthcareCase,
    stale_allergy_view: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the healthcare agents on one live case and return observed run + records."""

    if HEALTHCARE_RUN_ENGINE == "langgraph":
        from healthcare.langgraph_pipeline import build_langgraph_healthcare_steps as run_pipeline
    else:
        run_pipeline = build_healthcare_steps
    steps = run_pipeline(case, stale_allergy_view=stale_allergy_view)
    run = run_to_json(steps, run_id=case.case_id, resource_id=f"healthcare-case:{case.case_id}")
    return run, encode_healthcare_case(case)


__all__ = [
    "HEALTHCARE_RECORD_CODECS",
    "HEALTHCARE_RUN_ENGINE",
    "PRIOR_ALLERGY_STATE",
    "allergy_finding",
    "build_healthcare_steps",
    "care_disposition",
    "coverage_finding",
    "expected_disposition",
    "simulate_healthcare_case",
]
