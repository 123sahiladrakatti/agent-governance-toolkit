"""Tests for the GACO window optimizer (gaco.py). Offline: fake tokenizer and fake Ollama runtime."""

import copy
import unittest
from unittest import mock

import httpx

import gaco
from instrumentation import HarnessConfigError, _resolve, run_case
from test_instrumentation import FakeOllama, fake_harness, make_case


class GacoTests(unittest.TestCase):
    def setUp(self):
        self.case = make_case()
        self.harness = fake_harness(self.case)
        self.ollama = FakeOllama(self.harness)
        self.patch = mock.patch.object(httpx, "post", self.ollama)
        self.patch.start()
        trace, _ = run_case(self.case, 100_000, harness=self.harness)
        # The fake runtime passes exactly when the prompt is not truncated: assembled <= window.
        self.base = next(r for r in trace if r["agent"] == "care-decision")["assembled_tokens"]
        self.resolution = self.harness.config["sizing"]["resolution"]

    def tearDown(self):
        self.patch.stop()

    def windows(self, result):
        return [window for window, _ in result["probes"]]

    # ------------------------------------------------------------ floor
    def test_floor_is_the_exact_assembled_prompt_plus_output_room(self):
        detail = gaco.measure_floor_detail(self.case, harness=self.harness)
        self.assertEqual(detail["prompt_tokens"], self.base)            # same count as the real run's trace
        pipe = self.harness.pipeline
        config = _resolve(pipe["model_config_factory"])(model=self.harness.model["name"])
        self.assertEqual(detail["output_room_tokens"], config.num_predict)
        self.assertEqual(detail["floor"], self.base + config.num_predict + detail["margin_tokens"])
        self.assertEqual(detail["tag"], "measured")
        self.assertEqual(gaco.measure_floor(self.case, harness=self.harness), detail["floor"])

    def test_floor_makes_no_model_calls(self):
        def no_model(*args, **kwargs):
            raise AssertionError("measure_floor called the model")
        with mock.patch.object(httpx, "post", no_model):
            gaco.measure_floor(self.case, harness=self.harness)

    def test_floor_margin_comes_from_config(self):
        case = make_case(lambda h: h["gaco"].update(floor_margin_tokens=64))
        harness = fake_harness(case)
        self.assertEqual(gaco.measure_floor(case, harness=harness),
                         gaco.measure_floor(self.case, harness=self.harness) + 64)

    # ------------------------------------------------------------ oracle
    def test_probe_is_the_verdict(self):
        self.assertTrue(gaco.probe(self.case, self.base * 2, harness=self.harness))
        self.assertFalse(gaco.probe(self.case, self.base // 2, harness=self.harness))

    # ------------------------------------------------------------ search
    def test_over_provisioned_client_searches_down_to_the_floor(self):
        floor = gaco.measure_floor(self.case, harness=self.harness)
        original = copy.deepcopy(self.case)
        result = gaco.smallest_safe_window(self.case, self.base * 3, harness=self.harness)
        self.assertEqual(result["direction"], "down")
        self.assertTrue(result["found"])
        self.assertEqual(result["suggested_window"], floor)
        self.assertGreaterEqual(min(self.windows(result)), floor)        # never probes below the floor
        self.assertEqual(self.windows(result).count(floor), 2)           # confirmed by a repeat run
        self.assertTrue(result["monotonic"])
        self.assertEqual(self.case, original)                            # read-only

    def test_down_search_finds_the_fit_threshold_when_lo_is_lower(self):
        result = gaco.smallest_safe_window(self.case, self.base * 3, harness=self.harness, lo=self.base // 2)
        self.assertEqual(result["direction"], "down")
        self.assertGreaterEqual(result["suggested_window"], self.base)
        self.assertLessEqual(result["suggested_window"], self.base + self.resolution)

    def test_under_provisioned_client_searches_up(self):
        result = gaco.smallest_safe_window(self.case, self.base // 2, harness=self.harness, lo=self.base // 4)
        self.assertEqual(result["direction"], "up")
        self.assertTrue(result["initial_probed"])
        self.assertFalse(result["initial_passed"])
        self.assertIn("grow", [d["purpose"] for d in result["probe_details"]])
        self.assertGreaterEqual(result["suggested_window"], self.base)
        self.assertLessEqual(result["suggested_window"], self.base + self.resolution)

    def test_initial_below_floor_is_not_probed(self):
        floor = gaco.measure_floor(self.case, harness=self.harness)
        result = gaco.smallest_safe_window(self.case, self.base // 2, harness=self.harness)
        self.assertEqual(result["direction"], "up")
        self.assertFalse(result["initial_probed"])
        self.assertNotIn(self.base // 2, self.windows(result))
        self.assertEqual(result["suggested_window"], floor)

    def test_no_safe_window_stops_without_guessing(self):
        result = gaco.smallest_safe_window(self.case, self.base // 2, harness=self.harness,
                                           lo=self.base // 4, hi=self.base - 5)
        self.assertFalse(result["found"])
        self.assertIsNone(result["suggested_window"])
        self.assertLessEqual(max(self.windows(result)), self.base - 5)

    def test_floor_above_the_ceiling_runs_nothing(self):
        result = gaco.smallest_safe_window(self.case, self.base, harness=self.harness, hi=self.base // 2)
        self.assertFalse(result["found"])
        self.assertEqual(result["probes"], [])
        self.assertIn("above the largest window", result["stop_reason"])

    def test_ceiling_defaults_to_advertised_window_or_smaller_max(self):
        self.assertEqual(gaco._settings(self.harness)["max_window"], self.harness.model["advertised_window"])
        capped = fake_harness(make_case(lambda h: h["gaco"].update(max_window=8192)))
        self.assertEqual(gaco._settings(capped)["max_window"], 8192)
        with self.assertRaises(HarnessConfigError):
            gaco._settings(fake_harness(make_case(lambda h: h["gaco"].update(growth_factor=1))))

    # ------------------------------------------------------------ monotonicity guard
    def test_flip_at_the_candidate_widens_conservatively(self):
        threshold, seen = self.base, {}

        def scripted(case_json, window, harness):
            seen[window] = seen.get(window, 0) + 1
            # Passes at or above the threshold, except the repeat run near it fails (model variance).
            passed = window >= threshold and not (seen[window] > 1 and window < threshold + 3 * self.resolution)
            return {"window": window, "passed": passed, "category": None if passed else "required-evidence-not-carried",
                    "reason": "", "assembled_tokens": self.base, "processed_tokens": self.base, "truncated": False,
                    "decision": None, "evidence_check": "measured-only", "model_calls": 1}

        with mock.patch.object(gaco, "_run_probe", scripted):
            result = gaco.smallest_safe_window(self.case, self.base * 3, harness=self.harness, lo=self.base // 2)
        self.assertFalse(result["monotonic"])
        self.assertTrue(any("both passed and failed" in v for v in result["violations"]))
        self.assertTrue(result["widened"])
        suggested = result["suggested_window"]
        self.assertGreater(suggested, result["search_candidate"])
        runs = [passed for window, passed in result["probes"] if window == suggested]
        self.assertGreaterEqual(len(runs), 2)
        self.assertTrue(all(runs))                                       # never a window that ever failed
        failed = [window for window, passed in result["probes"] if not passed]
        self.assertGreater(suggested, max(failed))

    def test_inversion_is_reported(self):
        issues = gaco._monotonicity({1000: [True], 2000: [False], 3000: [True]})
        self.assertEqual(issues, ["window 2000 failed although the smaller window 1000 passed"])

    # ------------------------------------------------------------ report
    def test_optimize_report_and_text(self):
        floor = gaco.measure_floor(self.case, harness=self.harness)
        initial = self.base * 3
        report = gaco.optimize(self.case, initial, harness=self.harness)
        self.assertTrue(report["advisory"])
        self.assertEqual(report["floor"]["tag"], "measured")
        self.assertEqual(report["suggested_window"], {"tokens": floor, "tag": "found-by-search"})
        self.assertEqual(report["comparison"]["relation"], "smaller")
        self.assertEqual(report["comparison"]["delta_tokens"], initial - floor)
        self.assertEqual(report["model_runs"], len(report["probes"]))
        self.assertEqual(report["evidence_check"], "measured-only")
        text = gaco.format_report(report)
        self.assertTrue(text.endswith(
            f"Suggested context window: {floor}  ({initial - floor} tokens below your initial setting of {initial})."))
        for phrase in ("Advisory", "does not set it", "not a provable optimum", "relative to the tested case",
                       "maximum of the per-case floors", "probe ladder", "Measured-only", "[measured, no model runs]"):
            self.assertIn(phrase, text)

    def test_larger_and_not_found_headlines(self):
        report = gaco.optimize(self.case, self.base // 2, harness=self.harness)
        floor = report["floor"]["floor"]
        self.assertTrue(gaco.format_report(report).endswith(
            f"Suggested context window: {floor}  ({floor - self.base // 2} tokens above your initial setting of "
            f"{self.base // 2})."))
        missing = gaco.optimize(self.case, self.base // 2, harness=self.harness, lo=self.base // 4, hi=self.base - 5)
        text = gaco.format_report(missing)
        self.assertIn(f"No safe context window was found up to {self.base - 5} tokens", text)
        self.assertIn("may not fit this model's window at all", text)
        self.assertNotIn("Suggested context window:", text)


if __name__ == "__main__":
    unittest.main()
