"""Offline tests for the advisory GACO long-running projection."""

import unittest

from gaco_longrun import fit_linear_growth, project_long_running_context


class IntegerStateAdapter:
    """Token-exact synthetic state for testing the projection algorithm itself."""

    def empty(self):
        return 0

    def token_count(self, state):
        return int(state)

    def advance(self, state, case, trace, verdict, run_index):
        return int(state) + int(case["increment"])

    def synthesize(self, target_tokens, exemplar):
        return int(target_tokens)


def overflow_oracle(case, window, accumulated_state):
    occupied = int(case["base_tokens"]) + int(accumulated_state)
    passed = occupied <= window
    record = {
        "agent": "decision",
        "model_call": True,
        "assembled_tokens": occupied,
        "processed_tokens": min(occupied, window),
        "truncated": not passed,
        "overflow_error": False,
        "required_evidence": ["source"],
        "evidence_present_in_prompt": {"source": True},
        "evidence_visible": {"source": False if not passed else True},
        "model_visibility_confirmed": True,
        "evidence_used": {"source": passed},
        "decision": "CONTINUE" if passed else "CLEAR_WITHOUT_SOURCE",
    }
    verdict = {
        "passed": passed,
        "category": None if passed else "required-evidence-not-carried",
        "evidence_not_carried": {} if passed else {"decision": ["source"]},
    }
    return [record], verdict


def degradation_oracle(case, window, accumulated_state):
    occupied = int(case["base_tokens"]) + int(accumulated_state)
    passed = int(accumulated_state) < int(case["degradation_at"])
    record = {
        "agent": "decision",
        "model_call": True,
        "assembled_tokens": occupied,
        "processed_tokens": occupied,
        "truncated": False,
        "overflow_error": False,
        "required_evidence": ["source"],
        "evidence_present_in_prompt": {"source": True},
        "evidence_visible": {"source": True},
        "model_visibility_confirmed": True,
        "evidence_used": {"source": passed},
        "decision": "CONTINUE" if passed else "IGNORED_VISIBLE_SOURCE",
    }
    verdict = {
        "passed": passed,
        "category": None if passed else "required-evidence-not-carried",
        "evidence_not_carried": {} if passed else {"decision": ["source"]},
    }
    return [record], verdict


class LongRunProjectionTests(unittest.TestCase):
    def test_overflow_projection_and_fast_forward_crossover(self):
        cases = [{"base_tokens": 500, "increment": 100} for _ in range(4)]
        report = project_long_running_context(
            cases,
            950,
            oracle=overflow_oracle,
            state_adapter=IntegerStateAdapter(),
            maximum_occupied_tokens=2_000,
            resolution=1,
            cadence_seconds=86_400,
        )

        self.assertEqual([item.occupied_tokens for item in report.observations], [500, 600, 700, 800])
        self.assertEqual([item.accumulated_increment_tokens for item in report.observations], [100] * 4)
        self.assertTrue(report.growth_fit.adequate)
        self.assertAlmostEqual(report.growth_fit.tokens_per_run, 100.0)
        self.assertAlmostEqual(report.growth_fit.r_squared, 1.0)
        self.assertEqual(report.threshold.occupied_tokens, 951)
        self.assertEqual(report.threshold.just_below_occupied_tokens, 950)
        self.assertEqual(report.threshold.dominant_mechanism, "overflow")
        self.assertLess(report.projection.run_range[0], report.projection.run_range[1])
        self.assertIsNotNone(report.projection.elapsed_range)
        self.assertTrue(report.validation.early_runs_clean)
        self.assertTrue(report.validation.just_below_passed)
        self.assertFalse(report.validation.fast_forward_passed)
        self.assertTrue(report.validation.crossover_confirmed)
        self.assertIn("No loss observed in the first 4 runs", report.headline)
        self.assertIn("via overflow", report.headline)
        self.assertIn("Projection is extrapolated from 4 measured runs", report.caveats[0])
        self.assertTrue(report.validation.accelerated_proxy)

    def test_degradation_is_detected_without_assuming_overflow(self):
        cases = [{"base_tokens": 500, "increment": 100, "degradation_at": 350} for _ in range(3)]
        report = project_long_running_context(
            cases,
            10_000,
            oracle=degradation_oracle,
            state_adapter=IntegerStateAdapter(),
            maximum_occupied_tokens=2_000,
            resolution=1,
        )

        self.assertEqual(report.threshold.accumulated_tokens, 350)
        self.assertEqual(report.threshold.dominant_mechanism, "degradation")
        self.assertFalse(any(probe.truncated for probe in report.threshold.probes))
        self.assertTrue(report.validation.crossover_confirmed)
        self.assertEqual(report.validation.mechanism_observed, "degradation")

    def test_too_few_runs_reports_poor_fit_instead_of_forcing_projection(self):
        cases = [{"base_tokens": 500, "increment": 100} for _ in range(2)]
        report = project_long_running_context(
            cases,
            950,
            oracle=overflow_oracle,
            state_adapter=IntegerStateAdapter(),
            maximum_occupied_tokens=2_000,
            resolution=1,
        )

        self.assertFalse(report.growth_fit.adequate)
        self.assertIn("at least three", report.growth_fit.quality)
        self.assertIsNone(report.projection)
        self.assertIn("no reliable crossover range", report.headline)

    def test_non_linear_noisy_growth_is_labelled_poor(self):
        from gaco_longrun import GrowthObservation

        observations = [
            GrowthObservation(1, 100, 0, 10, True),
            GrowthObservation(2, 900, 10, 10, True),
            GrowthObservation(3, 120, 20, 10, True),
            GrowthObservation(4, 950, 30, 10, True),
        ]
        fit = fit_linear_growth(observations, minimum_r_squared=0.80)
        self.assertFalse(fit.adequate)
        self.assertIn("poor", fit.quality)


if __name__ == "__main__":
    unittest.main()
