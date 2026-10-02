"""Tests for the diamond graph and the AGT BehavioralAuthorityResolver."""

import unittest

from agent_sim import TRANSACTION_AGENT, build_chain, build_diamond, generate_alerts, run_chain_session, run_diamond_session
from agt_resolver import AGT_AVAILABLE, BehavioralAuthorityResolver, GraphGovernor, graph_context
from domain import GraphOutcome


class DiamondGovernanceTests(unittest.TestCase):
    def setUp(self):
        self.governor = GraphGovernor()
        self.reviews = [self.governor.review(o) for o in run_diamond_session(40, 7)]
        self.star = next(r for r in self.reviews if r.outcome.is_star_case)

    def test_star_case_all_hops_authorized_but_denied_and_attributed_to_c(self):
        self.assertEqual(len(self.star.handoffs), 6)
        self.assertTrue(all(h.passed for h in self.star.handoffs))
        self.assertEqual(self.star.attribution.merged_disposition, "CLEAR")
        self.assertEqual(self.star.attribution.expected_disposition, "FILE_SAR")
        self.assertFalse(self.star.policy_decision["allowed"])
        self.assertEqual(self.star.attribution.origin_node, "C")
        self.assertEqual(self.star.attribution.masked_by, ("customer-screening", "kyc-entity"))
        self.assertIn("misread $9,200 as $2,900", self.star.attribution.reason)
        self.assertEqual(self.star.status, "halted")

    def test_deny_comes_through_agt_policy_engine(self):
        if not AGT_AVAILABLE:
            self.skipTest("agentmesh not installed")
        self.assertTrue(self.star.agt_live)
        self.assertTrue(self.star.authority_decision["invoked_by_engine"])
        self.assertTrue(self.star.policy_decision["reason"].startswith("Authority resolver denied: "))

    def test_clean_diamonds_commit(self):
        clean = [r for r in self.reviews if not r.outcome.is_star_case]
        self.assertTrue(all(r.attribution.passed and r.status == "committed" for r in clean))
        self.assertEqual(sum(not r.attribution.passed for r in self.reviews), 1)

    def test_resolver_fails_closed_without_graph(self):
        from types import SimpleNamespace
        decision = BehavioralAuthorityResolver().resolve(SimpleNamespace(context={}))
        self.assertEqual(decision.decision, "deny")

    def test_context_round_trip_matches_direct_build(self):
        alert = generate_alerts(10, 7)[3]
        outcome = GraphOutcome(alert, build_diamond(alert, TRANSACTION_AGENT))
        decision = BehavioralAuthorityResolver().resolve(type("R", (), {"context": {"aml_graph": graph_context(outcome)}})())
        self.assertEqual(decision.decision, "deny")

    def test_linear_chain_unchanged(self):
        chain = build_chain(generate_alerts(8, 7)[0], corrupt=True)
        self.assertEqual(len(chain), 3)
        self.assertIsNone(chain[0].received_from)
        self.assertEqual(chain[1].received_from, "investigation-agent")
        self.assertEqual(len(run_chain_session(40, 7)), 40)


if __name__ == "__main__":
    unittest.main()
