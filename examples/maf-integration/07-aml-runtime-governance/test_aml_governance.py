import unittest

from agent_sim import generate_alerts, run_chain_session
from domain import recompute_structuring
from governance import ChainGovernance, ChainMonitor, SecurityLane


def allow_content(_agent_output):
    return {"passed": True, "rail": "self_check_output", "reason": "test allow"}


class AmlGovernanceTests(unittest.TestCase):
    def test_star_case_is_authorized_but_record_relative_governance_flags_it(self):
        outcome = run_chain_session(40, seed=7)[6]
        security = SecurityLane().check(outcome.chain, outcome.alert.alert_id)
        governance = ChainGovernance().check(outcome.alert, outcome.chain)

        self.assertTrue(recompute_structuring(outcome.alert).is_structuring)
        self.assertTrue(all(check.passed for check in security))
        self.assertFalse(governance.passed)
        self.assertEqual(governance.category, "transitive-corruption")

    def test_planted_faults_are_attributed_by_record_relative_governance(self):
        reviews = {
            outcome.fault: ChainGovernance().check(outcome.alert, outcome.chain)
            for outcome in run_chain_session(40, seed=7)
            if outcome.fault
        }

        self.assertEqual(reviews["transitive-corruption"].category, "transitive-corruption")
        self.assertEqual(reviews["wrong-target"].category, "wrong-target")
        self.assertEqual(reviews["scope"].category, "mis-delegation")
        self.assertEqual(reviews["conflict"].category, "conflict")

    def test_no_hidden_expected_disposition_labels(self):
        alert = generate_alerts(1, seed=7)[0]
        self.assertFalse(hasattr(alert, "expected_disposition"))

    def test_monitor_retains_bounded_context_and_checkpoints(self):
        monitor = ChainMonitor(context_window=8, checkpoint_interval=5, content_checker=allow_content)
        for outcome in run_chain_session(40, seed=7):
            monitor.evaluate(outcome)

        self.assertEqual(monitor.metrics.turns, 40)
        self.assertEqual(monitor.metrics.retained_context, 8)
        self.assertEqual(monitor.metrics.checkpoints, [5, 10, 15, 20, 25, 30, 35, 40])
        self.assertEqual(monitor.metrics.governance_flags, 4)
        self.assertEqual(monitor.metrics.security_flags, 1)
        self.assertEqual(monitor.metrics.security_intact_but_wrong, 3)


if __name__ == "__main__":
    unittest.main()
