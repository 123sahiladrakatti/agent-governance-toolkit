"""Healthcare reference-pipeline ingestion and governance tests."""

import json
import unittest
from pathlib import Path

from agt_resolver import GraphGovernor
from client_pipeline import compose_client_payload, ingest_client_pipeline, load_pipeline_definition
from healthcare.client_payload import HEALTHCARE_RECORD_CODECS, simulate_healthcare_case
from healthcare.correctness import HealthcareCorrectnessComponent
from healthcare.system import HealthcareCase


PIPELINE_FILE = Path(__file__).parent / "client_pipelines" / "healthcare_pipeline.json"


def no_content_check(_output):
    return {"passed": True, "rail": "unavailable", "reason": "disabled in tests"}


def govern(case: HealthcareCase, stale_allergy_view: bool):
    definition = load_pipeline_definition(PIPELINE_FILE)
    run, records = simulate_healthcare_case(case, stale_allergy_view=stale_allergy_view)
    payload = json.dumps(compose_client_payload(definition, run, records))
    ingested = ingest_client_pipeline(payload, HEALTHCARE_RECORD_CODECS)
    return GraphGovernor(
        content_checker=no_content_check,
        correctness_component=HealthcareCorrectnessComponent(),
        system_description=ingested.description,
        graph_policy=ingested.graph_policy,
        action_policy=ingested.action_policy,
    ).review(ingested.outcome)


def make_case(allergy="none", medication="Amoxicillin", authorized=True):
    return HealthcareCase(
        "HC-TEST", "PAT-001", "Synthetic Patient", 45, "bacterial infection",
        medication, allergy, authorized,
    )


class HealthcarePipelineTests(unittest.TestCase):
    def test_definition_supplies_healthcare_agents_and_permissions(self):
        definition = load_pipeline_definition(PIPELINE_FILE)
        self.assertEqual(
            [agent["id"] for agent in definition["system"]["agents"]],
            [
                "patient-intake", "allergy-safety", "medication-review",
                "coverage-review", "care-decision",
            ],
        )
        self.assertEqual(
            definition["system"]["permissions"]["allergy-safety"],
            ["evaluate_allergy_safety"],
        )

    def test_clean_case_continues(self):
        review = govern(make_case(), stale_allergy_view=False)
        self.assertEqual(review.policy_decision["action"], "allow")
        self.assertTrue(review.attribution.passed)
        self.assertEqual(review.outcome.step("care-decision").disposition, "APPROVE_TREATMENT")

    def test_stale_allergy_context_requires_human_review(self):
        review = govern(
            make_case(allergy="penicillin", medication="Amoxicillin"),
            stale_allergy_view=True,
        )
        self.assertEqual(review.policy_decision["action"], "require_approval")
        self.assertEqual(review.attribution.category, "unreconciled-conflict")
        self.assertIn("allergy-safety v1 < v2", review.attribution.reason)

    def test_correctly_propagated_contraindication_is_not_a_governance_failure(self):
        review = govern(
            make_case(allergy="penicillin", medication="Amoxicillin"),
            stale_allergy_view=False,
        )
        self.assertTrue(review.attribution.passed)
        self.assertEqual(
            review.outcome.step("care-decision").disposition,
            "HOLD_FOR_CLINICAL_REVIEW",
        )


if __name__ == "__main__":
    unittest.main()
