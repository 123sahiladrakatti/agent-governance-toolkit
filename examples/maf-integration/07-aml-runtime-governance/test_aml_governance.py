import unittest

from agent_sim import generate_alerts, investigate, run_session
from domain import recompute_structuring
from governance import EnhancedGovernance, RegularGovernance, StatefulGovernanceMonitor

class AmlGovernanceTests(unittest.TestCase):
    def test_star_case_is_record_relative_and_contrasting(self):
        alerts = generate_alerts(40, seed=7)
        action = investigate(alerts[6], 6, 40)
        regular = RegularGovernance().check(action, alerts[6])
        enhanced = EnhancedGovernance().check(action, alerts[6])
        self.assertTrue(recompute_structuring(alerts[6]).is_structuring)
        self.assertTrue(regular.passed)
        self.assertFalse(enhanced.passed)
        self.assertEqual(enhanced.category, "semantic-drift")

    def test_planted_faults_are_caught_by_enhanced_lane(self):
        planted = {}
        for alert, action in run_session(40, seed=7):
            regular = RegularGovernance().check(action, alert)
            enhanced = EnhancedGovernance().check(action, alert)
            if action.fault_label:
                planted[action.fault_label] = (regular, enhanced)
        self.assertTrue(planted["star"][0].passed)
        self.assertFalse(planted["star"][1].passed)
        self.assertEqual(planted["wrong-target"][1].category, "wrong-target")
        self.assertFalse(planted["mis-delegation"][0].passed)
        self.assertEqual(planted["mis-delegation"][1].category, "mis-delegation")
        self.assertEqual(planted["false-completion"][1].category, "false-completion")

    def test_no_hidden_expected_disposition_labels(self):
        alert = generate_alerts(1, seed=7)[0]
        self.assertFalse(hasattr(alert, "expected_disposition"))

    def test_stateful_monitor_retains_bounded_context_and_checkpoints(self):
        monitor = StatefulGovernanceMonitor(context_window=8, checkpoint_interval=5)
        for alert, action in run_session(40, seed=7):
            monitor.evaluate(action, alert)
        self.assertEqual(monitor.metrics.turns, 40)
        self.assertEqual(monitor.metrics.retained_context, 8)
        self.assertEqual(monitor.metrics.checkpoints, [5, 10, 15, 20, 25, 30, 35, 40])
        self.assertEqual(monitor.metrics.regular_missed, 3)
        self.assertEqual(monitor.metrics.drift_score, 6)

if __name__ == "__main__":
    unittest.main()
