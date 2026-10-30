"""Parity: the LangGraph care path must match the hand-rolled healthcare steps exactly."""

import itertools
import unittest
from dataclasses import fields

from domain import GraphStep
from healthcare.client_payload import (
    HEALTHCARE_RUN_ENGINE,
    build_healthcare_steps,
)
from healthcare.langgraph_pipeline import (
    REVIEW_BRANCHES,
    build_care_path_graph,
    build_langgraph_healthcare_steps,
    run_care_path_graph,
)
from healthcare.system import (
    ALLERGY_AGENT,
    CARE_DECISION_AGENT,
    INTAKE_AGENT,
    MEDICATION_ALLERGENS,
    HealthcareCase,
)

MEDICATIONS = ("Acetaminophen", *MEDICATION_ALLERGENS)
ALLERGIES = ("none", "penicillin", "cephalosporin", "NSAID")


def make_case(medication="Amoxicillin", allergy="none", authorized=True):
    return HealthcareCase(
        "HC-INTERACTIVE", "PAT-DEMO-001", "Synthetic Patient", 45, "bacterial infection",
        medication, allergy, authorized,
    )


class HealthcareLangGraphParityTests(unittest.TestCase):
    def assert_parity(self, case, stale):
        expected = build_healthcare_steps(case, stale_allergy_view=stale)
        actual = build_langgraph_healthcare_steps(case, stale_allergy_view=stale)
        self.assertEqual([s.agent for s in actual], [s.agent for s in expected])
        for want, got in zip(expected, actual):
            for field in fields(GraphStep):
                self.assertEqual(
                    getattr(got, field.name), getattr(want, field.name),
                    f"{want.agent}.{field.name} differs",
                )
        self.assertEqual(actual, expected)
        return actual

    def test_clean_case(self):
        steps = self.assert_parity(make_case("Acetaminophen", "none"), stale=False)
        self.assertEqual(steps[-1].disposition, "APPROVE_TREATMENT")

    def test_stale_allergy_conflict_case(self):
        steps = self.assert_parity(make_case("Amoxicillin", "penicillin"), stale=True)
        by_agent = {s.agent: s for s in steps}
        self.assertEqual(by_agent[ALLERGY_AGENT].finding_version, 2)
        self.assertEqual(by_agent[CARE_DECISION_AGENT].received_versions_by_parent[ALLERGY_AGENT], 1)
        self.assertEqual(
            by_agent[CARE_DECISION_AGENT].received_findings_by_parent[ALLERGY_AGENT], "no-contraindication"
        )

    def test_every_form_combination(self):
        for medication, allergy, authorized, stale in itertools.product(
            MEDICATIONS, ALLERGIES, (True, False), (False, True)
        ):
            with self.subTest(medication=medication, allergy=allergy, authorized=authorized, stale=stale):
                self.assert_parity(make_case(medication, allergy, authorized), stale)


class HealthcareLangGraphStructureTests(unittest.TestCase):
    def test_demo_uses_langgraph(self):
        self.assertEqual(HEALTHCARE_RUN_ENGINE, "langgraph")

    def test_graph_is_a_real_diamond(self):
        edges = {(edge.source, edge.target) for edge in build_care_path_graph().get_graph().edges}
        self.assertIn(("__start__", INTAKE_AGENT), edges)
        for branch in REVIEW_BRANCHES:
            self.assertIn((INTAKE_AGENT, branch), edges)
            self.assertIn((branch, CARE_DECISION_AGENT), edges)
        self.assertIn((CARE_DECISION_AGENT, "__end__"), edges)

    def test_reviews_run_between_intake_and_decision(self):
        executed = run_care_path_graph(make_case())["executed"]
        self.assertEqual(executed[0], INTAKE_AGENT)
        self.assertEqual(set(executed[1:4]), set(REVIEW_BRANCHES))
        self.assertEqual(executed[4], CARE_DECISION_AGENT)


if __name__ == "__main__":
    unittest.main()
