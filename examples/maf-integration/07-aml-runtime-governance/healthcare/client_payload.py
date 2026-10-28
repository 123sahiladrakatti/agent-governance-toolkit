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


def simulate_healthcare_case(
    case: HealthcareCase,
    stale_allergy_view: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the reference healthcare agents and return observed run + records.

    When ``stale_allergy_view`` is true, the safety agent emits its current v2
    contraindication but the final agent consumes an earlier v1 no-hit result.
    The governance framework detects this from provenance/version metadata.
    """

    allergy, allergy_flagged = allergy_finding(case)
    allergy_version = 2 if allergy_flagged else 1
    coverage = "authorized" if case.coverage_authorized else "not-authorized"
    intake_finding = "care-request-routed"
    medication_finding = "medication-order-validated"

    consumed_allergy = "no-contraindication" if stale_allergy_view else allergy
    consumed_allergy_version = 1 if stale_allergy_view else allergy_version
    if consumed_allergy == "contraindication":
        disposition = "HOLD_FOR_CLINICAL_REVIEW"
    elif coverage == "not-authorized":
        disposition = "HOLD_FOR_AUTHORIZATION"
    else:
        disposition = "APPROVE_TREATMENT"

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
             finding_version=allergy_version),
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
    run = run_to_json(steps, run_id=case.case_id, resource_id=f"healthcare-case:{case.case_id}")
    return run, encode_healthcare_case(case)


__all__ = [
    "HEALTHCARE_RECORD_CODECS",
    "allergy_finding",
    "expected_disposition",
    "simulate_healthcare_case",
]
