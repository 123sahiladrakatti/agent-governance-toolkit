"""Healthcare reference records, declared pipeline, and record codec."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from system_description import AgentDescription, SystemDescription


INTAKE_AGENT = "patient-intake"
ALLERGY_AGENT = "allergy-safety"
MEDICATION_AGENT = "medication-review"
COVERAGE_AGENT = "coverage-review"
CARE_DECISION_AGENT = "care-decision"

HEALTHCARE_NODE_ID = {
    INTAKE_AGENT: "H-A",
    ALLERGY_AGENT: "H-B",
    MEDICATION_AGENT: "H-C",
    COVERAGE_AGENT: "H-D",
    CARE_DECISION_AGENT: "H-E",
}
HEALTHCARE_PARENTS = {
    INTAKE_AGENT: (),
    ALLERGY_AGENT: (INTAKE_AGENT,),
    MEDICATION_AGENT: (INTAKE_AGENT,),
    COVERAGE_AGENT: (INTAKE_AGENT,),
    CARE_DECISION_AGENT: (ALLERGY_AGENT, MEDICATION_AGENT, COVERAGE_AGENT),
}
HEALTHCARE_ALLOWED = {
    INTAKE_AGENT: frozenset({"route_care_request"}),
    ALLERGY_AGENT: frozenset({"evaluate_allergy_safety"}),
    MEDICATION_AGENT: frozenset({"validate_medication_order"}),
    COVERAGE_AGENT: frozenset({"verify_coverage"}),
    CARE_DECISION_AGENT: frozenset({"merge_care_decision"}),
}

# Tiny reference knowledge base. A real customer would connect its formulary,
# EHR, clinical rules, and authorization records through its correctness plug-in.
MEDICATION_ALLERGENS = {
    "Amoxicillin": "penicillin",
    "Penicillin V": "penicillin",
    "Cephalexin": "cephalosporin",
    "Ibuprofen": "NSAID",
}


@dataclass(frozen=True)
class HealthcareCase:
    case_id: str
    patient_id: str
    patient_name: str
    age: int
    diagnosis: str
    medication: str
    known_allergy: str
    coverage_authorized: bool


def encode_healthcare_case(value: object) -> dict[str, object]:
    if not isinstance(value, HealthcareCase):
        raise TypeError("Healthcare trace records must be a HealthcareCase")
    return asdict(value)


def decode_healthcare_case(value: object) -> HealthcareCase:
    if not isinstance(value, dict):
        raise TypeError("Healthcare graph context must contain an encoded case")
    return HealthcareCase(
        case_id=str(value["case_id"]),
        patient_id=str(value["patient_id"]),
        patient_name=str(value["patient_name"]),
        age=int(value["age"]),
        diagnosis=str(value["diagnosis"]),
        medication=str(value["medication"]),
        known_allergy=str(value["known_allergy"]),
        coverage_authorized=bool(value["coverage_authorized"]),
    )


HEALTHCARE_SYSTEM_DESCRIPTION = SystemDescription(
    name="healthcare-care-path",
    schema="healthcare-care-path/v1",
    agents=tuple(
        AgentDescription(agent, HEALTHCARE_NODE_ID[agent], HEALTHCARE_ALLOWED[agent])
        for agent in HEALTHCARE_NODE_ID
    ),
    topology=HEALTHCARE_PARENTS,
    final_agent=CARE_DECISION_AGENT,
    final_action="commit_care_decision",
    resolver_action="resolve_care_correctness",
    handoff_action="healthcare_handoff",
    resource_prefix="healthcare-case",
    escalation_outcomes=frozenset({"HOLD_FOR_CLINICAL_REVIEW", "HOLD_FOR_AUTHORIZATION"}),
    encode_records=encode_healthcare_case,
    decode_records=decode_healthcare_case,
)


__all__ = [
    "ALLERGY_AGENT",
    "CARE_DECISION_AGENT",
    "COVERAGE_AGENT",
    "HEALTHCARE_ALLOWED",
    "HEALTHCARE_NODE_ID",
    "HEALTHCARE_PARENTS",
    "HEALTHCARE_SYSTEM_DESCRIPTION",
    "HealthcareCase",
    "INTAKE_AGENT",
    "MEDICATION_AGENT",
    "MEDICATION_ALLERGENS",
    "decode_healthcare_case",
    "encode_healthcare_case",
]
