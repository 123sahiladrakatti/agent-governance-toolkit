"""Tests for optional customer-supplied correctness components."""

import unittest
import json

from agent_sim import run_diamond_session
from agt_resolver import ACTION_GOVERNANCE_POLICY, GRAPH_POLICY, GraphGovernor


def allow_content(_agent_output):
    return {"passed": True, "rail": "self_check_output", "reason": "test allow"}


class OptionalCorrectnessTests(unittest.TestCase):
    def test_customer_policies_are_injected_into_both_agt_adapters(self):
        graph_policy = json.loads(GRAPH_POLICY)
        graph_policy["name"] = "customer-graph-policy"
        graph_policy["rules"][0]["name"] = "customer-allow-handoff"
        action_policy = json.loads(ACTION_GOVERNANCE_POLICY)
        action_policy["name"] = "customer-action-policy"
        action_policy["rules"][3]["name"] = "customer-review-content"
        flagged = lambda _: {
            "passed": False,
            "rail": "self_check_output",
            "reason": "customer content flag",
        }
        outcome = run_diamond_session(8, 7)[0]

        review = GraphGovernor(
            content_checker=flagged,
            graph_policy=json.dumps(graph_policy),
            action_policy=json.dumps(action_policy),
        ).review(outcome)

        self.assertTrue(all(handoff.rule == "customer-allow-handoff" for handoff in review.handoffs))
        self.assertEqual(review.policy_decision["policy_name"], "customer-action-policy")
        self.assertEqual(review.policy_decision["matched_rule"], "customer-review-content")

    def test_without_correctness_clean_integrity_can_continue(self):
        outcome = next(item for item in run_diamond_session(8, 7) if item.is_star_case)

        review = GraphGovernor(
            content_checker=allow_content,
            correctness_component=None,
        ).review(outcome)

        self.assertTrue(review.attribution.passed)
        self.assertIn("not configured", review.attribution.reason)
        self.assertEqual(review.policy_decision["action"], "allow")

    def test_without_correctness_generic_stale_conflict_still_requires_review(self):
        outcome = next(item for item in run_diamond_session(8, 7) if item.is_conflict_case)

        review = GraphGovernor(
            content_checker=allow_content,
            correctness_component=None,
        ).review(outcome)

        self.assertFalse(review.attribution.passed)
        self.assertEqual(review.attribution.category, "unreconciled-conflict")
        self.assertEqual(review.policy_decision["action"], "require_approval")


if __name__ == "__main__":
    unittest.main()
