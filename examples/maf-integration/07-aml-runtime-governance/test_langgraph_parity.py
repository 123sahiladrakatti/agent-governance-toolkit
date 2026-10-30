"""Parity: the LangGraph diamond must match the hand-rolled build_diamond exactly."""

import unittest
from dataclasses import fields
from datetime import datetime

from agent_sim import TRANSACTION_AGENT, build_diamond, generate_alerts, run_diamond_session
from aml.langgraph_pipeline import build_diamond_graph, build_langgraph_diamond, run_diamond_graph
from aml.system import DIAMOND_BRANCHES, DISPOSITION_AGENT, SANCTIONS_AGENT, TRIAGE_AGENT
from domain import Alert, CustomerProfile, GraphStep, Transaction


def make_alert(name, amounts=(2_500.0, 1_500.0)):
    account = "ACCT-DEMO-001"
    return Alert(
        "AML-INTERACTIVE",
        account,
        tuple(
            Transaction(f"T-{i:03d}", account, amount, "cash_deposit", datetime(2026, 1, 4 + i), "BR-01")
            for i, amount in enumerate(amounts, start=1)
        ),
        CustomerProfile(account, name, "business owner", 8_000.0),
        "customer and transaction activity review",
    )


class LangGraphParityTests(unittest.TestCase):
    def assert_parity(self, alert, faulted_branch=None, stale=False):
        expected = build_diamond(alert, faulted_branch, stale_screening_view=stale)
        actual = build_langgraph_diamond(alert, faulted_branch, stale_screening_view=stale)
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
        steps = self.assert_parity(make_alert("Jane Smith"))
        self.assertEqual(steps[-1].disposition, "CLEAR")

    def test_stale_screening_conflict_case(self):
        steps = self.assert_parity(make_alert("Org 1"), stale=True)
        decision = {s.agent: s for s in steps}[DISPOSITION_AGENT]
        screening = {s.agent: s for s in steps}[SANCTIONS_AGENT]
        self.assertEqual(screening.finding_version, 2)
        self.assertEqual(decision.received_versions_by_parent[SANCTIONS_AGENT], 1)
        self.assertEqual(decision.received_findings_by_parent[SANCTIONS_AGENT], "no-hit")

    def test_structuring_case(self):
        self.assert_parity(make_alert("Jane Smith", (9_200.0, 9_500.0, 9_100.0, 9_400.0)))

    def test_misread_branch_case(self):
        self.assert_parity(make_alert("Jane Smith", (9_200.0, 9_500.0, 9_100.0, 9_400.0)), TRANSACTION_AGENT)

    def test_generated_alerts_all_variants(self):
        for alert in generate_alerts(12, 7):
            for faulted in (None, TRANSACTION_AGENT):
                for stale in (False, True):
                    with self.subTest(alert=alert.alert_id, faulted=faulted, stale=stale):
                        self.assert_parity(alert, faulted, stale)

    def test_session_outcomes_match(self):
        for outcome in run_diamond_session(8, 7):
            faulted = TRANSACTION_AGENT if outcome.is_star_case else None
            with self.subTest(alert=outcome.alert.alert_id):
                self.assertEqual(
                    build_langgraph_diamond(outcome.alert, faulted, outcome.is_conflict_case),
                    outcome.steps,
                )


class LangGraphStructureTests(unittest.TestCase):
    def test_graph_is_a_real_diamond(self):
        graph = build_diamond_graph().get_graph()
        edges = {(edge.source, edge.target) for edge in graph.edges}
        self.assertIn(("__start__", TRIAGE_AGENT), edges)
        for branch in DIAMOND_BRANCHES:
            self.assertIn((TRIAGE_AGENT, branch), edges)
            self.assertIn((branch, DISPOSITION_AGENT), edges)
        self.assertIn((DISPOSITION_AGENT, "__end__"), edges)

    def test_branches_run_between_intake_and_decision(self):
        executed = run_diamond_graph(make_alert("Jane Smith"))["executed"]
        self.assertEqual(executed[0], TRIAGE_AGENT)
        self.assertEqual(set(executed[1:4]), set(DIAMOND_BRANCHES))
        self.assertEqual(executed[4], DISPOSITION_AGENT)


if __name__ == "__main__":
    unittest.main()
