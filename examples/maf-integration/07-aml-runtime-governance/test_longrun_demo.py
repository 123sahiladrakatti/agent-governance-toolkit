"""Tests for the long-running context-loss demo (longrun_demo.py)."""

import copy
import json
import re
import unittest
from unittest import mock

import httpx

import longrun_demo
from instrumentation import run_case
from longrun_demo import (
    FillerGuardError,
    classify_mechanism,
    format_report,
    required_finding,
    run_demo,
    synthesize_accumulated_context,
)
from test_instrumentation import FakeOllama, FakeTokenizer, fake_harness, make_case

HISTORY_LINE = re.compile(r"^\d{4}-\d{2}-\d{2} run \d{3} · HC-PRIOR-\d{4}:", re.M)


class DegradingOllama(FakeOllama):
    """Like FakeOllama, but the 'model' ignores a report buried deep in the middle of the context."""

    def __init__(self, harness, bury_tokens):
        super().__init__(harness)
        self.bury_tokens = bury_tokens

    def __call__(self, url, json=None, timeout=None, **kwargs):
        response = super().__call__(url, json=json, timeout=timeout, **kwargs)
        if json["options"].get("num_predict") == 1:
            return response
        content = json["messages"][-1]["content"]
        line = content.find("[REPORT allergy-safety")
        before = self.harness.count(content[:line], with_bos=False)
        after = self.harness.count(content[line:], with_bos=False)
        if before > self.bury_tokens and after > self.bury_tokens:
            body = response.json()
            reply = __import__("json").loads(body["message"]["content"])
            reply["allergy"] = "not-present"
            if reply["disposition"] == "HOLD_FOR_CLINICAL_REVIEW":
                reply["disposition"] = "APPROVE_TREATMENT"
            body["message"]["content"] = __import__("json").dumps(reply)
            return httpx.Response(200, json=body, request=response.request)
        return response


class LongRunTests(unittest.TestCase):
    def setUp(self):
        self.case = make_case()
        self.harness = fake_harness(self.case)
        self.finding = required_finding(self.case, self.harness)
        self.ollama = FakeOllama(self.harness)
        self.patch = mock.patch.object(httpx, "post", self.ollama)
        self.patch.start()
        trace, _ = run_case(self.case, 100_000, harness=self.harness)
        self.base = next(r for r in trace if r["agent"] == "care-decision")["assembled_tokens"]

    def tearDown(self):
        self.patch.stop()

    def decision_prompt(self):
        return next(r for r in self.ollama.requests if r["options"].get("num_predict") != 1)["messages"][-1]["content"]

    def test_finding_comes_from_the_pipeline(self):
        self.assertEqual(self.finding["id"], "allergy-safety")
        self.assertEqual(self.finding["finding"], "contraindication")
        self.assertTrue(self.finding["line"].startswith("[REPORT allergy-safety v2]"))

    def test_filler_size_and_guard(self):
        for placement in longrun_demo.PLACEMENTS:
            with self.subTest(placement=placement):
                context = synthesize_accumulated_context(400, self.finding, placement, harness=self.harness)
                self.assertGreaterEqual(context["tokens"], 400)
                self.assertLess(context["tokens"], 400 + 40)          # within about one note
                text = context["before_finding"] + context["after_finding"]
                for word in ("allergy", "contraindication", "amoxicillin", "penicillin", "[report"):
                    self.assertNotIn(word, text.lower())
        with mock.patch.object(longrun_demo, "_EVENTS", ("penicillin stock checked",)):
            with self.assertRaises(FillerGuardError):
                synthesize_accumulated_context(50, self.finding, "recent_end", harness=self.harness)

    def test_placements_in_the_agent_prompt(self):
        expectations = {"recent_end": ("all-before",), "buried_mid": ("both",), "overflowed": ("all-after",)}
        for placement in expectations:
            with self.subTest(placement=placement):
                context = synthesize_accumulated_context(300, self.finding, placement, harness=self.harness)
                self.ollama.requests.clear()
                run_case(self.case, 100_000, context, harness=self.harness)
                prompt = self.decision_prompt()
                at = prompt.index(self.finding["line"])
                history = [m.start() for m in HISTORY_LINE.finditer(prompt)]
                before = [p for p in history if p < at]
                after = [p for p in history if p > at]
                if placement == "recent_end":
                    self.assertTrue(before and not after)
                elif placement == "buried_mid":
                    self.assertTrue(before and after)
                else:
                    self.assertTrue(after and not before)
                self.assertIn(self.finding["line"], prompt)             # the finding itself is untouched

    def test_demo_overflow_after_burial_was_insufficient(self):
        window = self.base * 3
        original = copy.deepcopy(self.case["records"])
        report = run_demo(self.case, window, 50, self.base, harness=self.harness)
        c1, c2 = report["case1"], report["case2"]
        self.assertTrue(c1["passed"])
        self.assertTrue(c1["agent_used_finding"])
        self.assertTrue(report["burial_alone_insufficient"])
        self.assertEqual([a["placement"] for a in report["case2_attempts"]], ["buried_mid", "overflowed"])
        self.assertTrue(report["case2_attempts"][0]["passed"])
        self.assertFalse(c2["passed"])
        self.assertEqual(c2["mechanism"], "overflow")
        self.assertFalse(c2["finding_inside_window"])
        self.assertFalse(c2["agent_used_finding"])
        self.assertEqual({c1["window"], c2["window"]}, {window})       # same window: growth is the cause
        self.assertGreater(c2["occupied_tokens"], c1["occupied_tokens"])
        self.assertEqual(self.case["records"], original)
        text = format_report(report)
        self.assertIn("CATCHES LOSS", text)
        self.assertIn("the required finding is lost (overflow)", text)
        self.assertIn("accelerated simulation", text)
        self.assertIn("no loss observed in these runs", text)
        self.assertNotIn("zero loss", text.lower())

    def test_demo_detects_degradation_inside_the_window(self):
        self.patch.stop()
        with mock.patch.object(httpx, "post", DegradingOllama(self.harness, bury_tokens=self.base // 2)):
            report = run_demo(self.case, self.base * 4, 50, self.base * 2, harness=self.harness)
        self.patch.start()
        c2 = report["case2"]
        self.assertTrue(report["case1"]["passed"])
        self.assertFalse(report["burial_alone_insufficient"])
        self.assertEqual(c2["placement"], "buried_mid")
        self.assertFalse(c2["truncated"])
        self.assertTrue(c2["finding_inside_window"])
        self.assertFalse(c2["agent_used_finding"])
        self.assertEqual(c2["mechanism"], "degradation")
        self.assertIn("yes, but buried mid-context", format_report(report))

    def test_case1_failure_is_reported_honestly(self):
        report = run_demo(self.case, self.base // 2, 50, self.base, harness=self.harness)
        self.assertFalse(report["case1"]["passed"])
        self.assertIn("Case 1 already fails", format_report(report))

    def test_read_only_branch_findings(self):
        context = synthesize_accumulated_context(300, self.finding, "buried_mid", harness=self.harness)
        plain, _ = run_case(self.case, 100_000, harness=self.harness)
        with_history, _ = run_case(self.case, 100_000, context, harness=self.harness)
        branch = lambda trace: {r["agent"]: r["decision"] for r in trace if not r["model_call"]}
        self.assertEqual(branch(plain), branch(with_history))


class StrictOllama(FakeOllama):
    """A runtime that truncates before the prompt fills the window (keeps a reserve)."""

    def __init__(self, harness, reserve):
        super().__init__(harness)
        self.reserve = reserve

    def __call__(self, url, json=None, timeout=None, **kwargs):
        json = __import__("copy").deepcopy(json)
        json["options"]["num_ctx"] = json["options"]["num_ctx"] - self.reserve
        return super().__call__(url, json=json, timeout=timeout, **kwargs)


class DegradationDemoTests(unittest.TestCase):
    def setUp(self):
        self.case = make_case()
        self.harness = fake_harness(self.case)
        self.patch = mock.patch.object(httpx, "post", FakeOllama(self.harness))
        self.patch.start()
        trace, _ = run_case(self.case, 100_000, harness=self.harness)
        self.base = next(r for r in trace if r["agent"] == "care-decision")["assembled_tokens"]
        self.settings = {"growth_factor": 2, "max_window": self.base * 32, "margin_tokens": 16}

    def tearDown(self):
        self.patch.stop()

    def run_with(self, runtime, window, aged, **settings):
        self.patch.stop()
        try:
            with mock.patch.object(httpx, "post", runtime):
                return longrun_demo.run_degradation_demo(
                    self.case, window, 50, aged, 0.9, harness=self.harness, **{**self.settings, **settings})
        finally:
            self.patch.start()

    def test_robust_model_is_reported_honestly(self):
        report = self.run_with(FakeOllama(self.harness), self.base * 4, self.base, max_aged_tokens=self.base * 4)
        self.assertEqual(report["outcome"], "robust")
        self.assertTrue(all(a["inside_window_proven"] and a["passed"] for a in report["sweep"]))
        self.assertEqual([a["history_tokens"] // self.base for a in report["sweep"]], [1, 2, 4])
        text = longrun_demo.format_degradation_report(report)
        self.assertIn("No degradation observed up to", text)
        self.assertIn("held the finding throughout", text)
        self.assertNotIn("CATCHES LOSS", text)
        self.assertNotIn("zero loss", text.lower())

    def test_degradation_onset_inside_the_window(self):
        # Degrades once enough history surrounds the finding: holds at 1x and 2x, loses it at 4x.
        degrading = DegradingOllama(self.harness, bury_tokens=self.base // 4)
        report = self.run_with(degrading, self.base * 4, self.base, max_aged_tokens=self.base * 16)
        self.assertEqual(report["outcome"], "degradation")
        c2 = report["case2"]
        self.assertTrue(c2["inside_window_proven"])
        self.assertFalse(c2["truncated"])
        self.assertEqual(c2["occupied_tokens"], c2["processed_tokens"])
        self.assertFalse(c2["agent_used_finding"])
        self.assertEqual(c2["mechanism"], "degradation")
        self.assertGreater(c2["tokens_after_finding"], c2["history_tokens"] // 2)   # buried deep (depth 0.9)
        self.assertTrue(report["sweep"][0]["passed"])                               # escalated to the onset
        self.assertEqual(report["onset"]["history_tokens"], c2["history_tokens"])
        text = longrun_demo.format_degradation_report(report)
        self.assertIn("yes, NOT truncated", text)
        self.assertIn("degradation (detected)", text)
        self.assertIn("a bigger window does not help", text)
        self.assertIn("accelerated simulation", text)

    def test_truncated_attempt_is_invalid_and_window_grows(self):
        strict = StrictOllama(self.harness, reserve=self.base * 2)   # truncates the first fitted window
        report = self.run_with(strict, self.base * 2, self.base, max_aged_tokens=self.base)
        first, last = report["sweep"][0], report["sweep"][-1]
        self.assertTrue(first["truncated"])
        self.assertIn("not a degradation case", first["invalid"])
        self.assertTrue(last["inside_window_proven"])
        self.assertGreater(last["window"], first["window"])
        self.assertNotEqual(report["outcome"], "degradation")      # never reports an overflow as degradation

    def test_window_ceiling_is_reported(self):
        report = self.run_with(FakeOllama(self.harness), self.base * 2, self.base * 8,
                               max_aged_tokens=self.base * 8, max_window=self.base * 3)
        self.assertIn(report["outcome"], ("window-ceiling", "robust"))
        self.assertTrue(all(a["window"] <= self.base * 3 for a in report["sweep"]))
        tiny = self.run_with(FakeOllama(self.harness), self.base * 2, self.base * 8,
                             max_aged_tokens=self.base * 8, max_window=self.base)
        self.assertEqual(tiny["outcome"], "window-ceiling")

    def test_depth_places_most_history_on_top_of_the_finding(self):
        finding = required_finding(self.case, self.harness)
        context = synthesize_accumulated_context(500, finding, "buried_mid", harness=self.harness, depth=0.9)
        self.assertGreater(context["tokens_after_finding"], 0.8 * context["tokens"])
        self.assertTrue(context["before_finding"])
        with self.assertRaises(ValueError):
            synthesize_accumulated_context(500, finding, "buried_mid", harness=self.harness, depth=1.0)


class AccumulatedStateSlotTests(unittest.TestCase):
    """The two-slot history in llm_decision.build_case_file is invisible when unused."""

    def test_prompt_parity(self):
        from healthcare.langgraph_pipeline import REVIEW_BRANCHES, run_care_path_graph
        from healthcare.llm_decision_agent import build_case_file, default_config
        from healthcare.system import decode_healthcare_case
        from test_instrumentation import RECORDS

        case = decode_healthcare_case(RECORDS)
        steps = run_care_path_graph(case)["steps"]
        branches = {agent: steps[agent] for agent in REVIEW_BRANCHES}
        build = lambda state: build_case_file(case, branches, default_config(), state)[0]
        empty = build("")
        self.assertEqual(build(None), empty)
        self.assertEqual(build({}), empty)
        self.assertEqual(build({"before_finding": "", "after_finding": ""}), empty)
        self.assertEqual(build({"after_finding": "note"}), build("note"))
        both = build({"before_finding": "OLDER", "after_finding": "NEWER"})
        self.assertLess(both.index("OLDER"), both.index("[REPORT allergy-safety"))
        self.assertLess(both.index("[REPORT allergy-safety"), both.index("NEWER"))


class ClassifyMechanismTests(unittest.TestCase):
    def record(self, **overrides):
        base = {"agent": "care-decision", "model_call": True, "required_evidence": ["allergy-safety"],
                "assembled_tokens": 1000, "processed_tokens": 1000, "truncated": False,
                "evidence_in_kept_window_estimate": {"allergy-safety": True},
                "evidence_used": {"allergy-safety": True}}
        return [{**base, **overrides}]

    def test_overflow(self):
        trace = self.record(assembled_tokens=3000, processed_tokens=514, truncated=True,
                            evidence_in_kept_window_estimate={"allergy-safety": False},
                            evidence_used={"allergy-safety": False})
        self.assertEqual(classify_mechanism(trace), "overflow")

    def test_degradation(self):
        self.assertEqual(classify_mechanism(self.record(evidence_used={"allergy-safety": False})), "degradation")

    def test_truncated_but_finding_kept_and_unused_is_degradation(self):
        trace = self.record(assembled_tokens=3000, processed_tokens=514, truncated=True,
                            evidence_used={"allergy-safety": False})
        self.assertEqual(classify_mechanism(trace), "degradation")

    def test_indeterminate(self):
        trace = self.record(assembled_tokens=3000, processed_tokens=514, truncated=True,
                            evidence_in_kept_window_estimate={"allergy-safety": None},
                            evidence_used={"allergy-safety": None})
        self.assertEqual(classify_mechanism(trace), "indeterminate")
        self.assertEqual(classify_mechanism([{"agent": "x", "model_call": False}]), "indeterminate")


if __name__ == "__main__":
    unittest.main()
