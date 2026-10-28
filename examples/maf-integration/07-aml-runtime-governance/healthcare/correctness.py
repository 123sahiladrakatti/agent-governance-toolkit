"""Healthcare implementation of the optional record-relative correctness API."""

from __future__ import annotations

from correctness import CorrectnessResult
from domain import BranchCheck, GraphOutcome
from healthcare.client_payload import allergy_finding, expected_disposition
from healthcare.system import (
    ALLERGY_AGENT,
    CARE_DECISION_AGENT,
    COVERAGE_AGENT,
    HealthcareCase,
    MEDICATION_AGENT,
)


class HealthcareCorrectnessComponent:
    """Validate healthcare findings against the submitted patient records."""

    name = "healthcare-record-relative-correctness"

    def check(self, trace: GraphOutcome, records: object) -> CorrectnessResult:
        if not isinstance(records, HealthcareCase):
            raise TypeError("HealthcareCorrectnessComponent requires HealthcareCase records")

        by_agent = {step.agent: step for step in trace.steps}
        allergy_expected, allergy_flagged = allergy_finding(records)
        coverage_expected = "authorized" if records.coverage_authorized else "not-authorized"
        medication_expected = "medication-order-validated"
        expected = expected_disposition(records)
        actual = by_agent[CARE_DECISION_AGENT].disposition

        checks = (
            BranchCheck(
                ALLERGY_AGENT,
                by_agent[ALLERGY_AGENT].node_id,
                by_agent[ALLERGY_AGENT].finding,
                allergy_expected,
                by_agent[ALLERGY_AGENT].finding == allergy_expected
                and by_agent[ALLERGY_AGENT].flagged == allergy_flagged,
                "Medication/allergy safety recomputed from the submitted clinical record.",
            ),
            BranchCheck(
                MEDICATION_AGENT,
                by_agent[MEDICATION_AGENT].node_id,
                by_agent[MEDICATION_AGENT].finding,
                medication_expected,
                by_agent[MEDICATION_AGENT].finding == medication_expected,
                "Medication-order validation compared with the submitted order.",
            ),
            BranchCheck(
                COVERAGE_AGENT,
                by_agent[COVERAGE_AGENT].node_id,
                by_agent[COVERAGE_AGENT].finding,
                coverage_expected,
                by_agent[COVERAGE_AGENT].finding == coverage_expected,
                "Coverage status recomputed from the submitted authorization record.",
            ),
            BranchCheck(
                CARE_DECISION_AGENT,
                by_agent[CARE_DECISION_AGENT].node_id,
                str(actual),
                expected,
                actual == expected,
                "Final care disposition compared with all record-relative branch results.",
            ),
        )
        passed = all(check.matches for check in checks)
        first_bad = next((check for check in checks if not check.matches), None)
        return CorrectnessResult(
            passed=passed,
            expected=expected,
            actual=actual,
            origin_agent=None if passed else first_bad.agent,
            origin_node=None if passed else first_bad.node_id,
            category=None if passed else "healthcare-correctness-failure",
            reason=(
                f"Care decision {actual} matches the submitted healthcare records."
                if passed else
                f"{first_bad.agent} reported {first_bad.reported!r}; records require "
                f"{first_bad.recomputed!r}. Final care decision is {actual!r}, expected {expected!r}."
            ),
            checks=checks,
            metadata={"component": self.name},
        )


__all__ = ["HealthcareCorrectnessComponent"]
