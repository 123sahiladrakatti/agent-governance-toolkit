"""Tests for the config-driven governance harness (instrumentation.py)."""

import copy
import json
import re
import unittest
from pathlib import Path
from unittest import mock

import httpx

import instrumentation
from instrumentation import HarnessConfigError, case_json_for, load_harness, run_case, smallest_safe_window

HERE = Path(__file__).parent
DEFINITION = HERE / "client_pipelines" / "healthcare_pipeline.json"
RECORDS = json.loads((HERE / "harness_cases" / "healthcare_contraindication.json").read_text(encoding="utf-8"))


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    """Deterministic stand-in: special tokens, words, and punctuation each count as one token."""

    def encode(self, text, add_special_tokens=False):
        return FakeEncoding(re.findall(r"<\|[^|]+\|>|\w+|[^\w\s]", text))


class FakeOllama:
    """Ollama-like /api/chat: keeps the tail on overflow and reports prompt_eval_count.

    The 'model' answers from the reports that fit in its kept window, like the real decision agent.
    """

    def __init__(self, harness, overflow_errors=False):
        self.harness, self.overflow_errors, self.requests = harness, overflow_errors, []

    def __call__(self, url, json=None, timeout=None, **kwargs):
        self.requests.append(copy.deepcopy(json))
        request = httpx.Request("POST", url)
        rendered = self.harness.render(json["messages"])
        assembled = self.harness.count(rendered)
        window = json["options"]["num_ctx"]
        if assembled > window and self.overflow_errors:
            return httpx.Response(400, text="input length exceeds the context length", request=request)
        processed = assembled if assembled <= window else window // 2 + 2
        if json["options"].get("num_predict") == 1:
            return httpx.Response(200, json={"prompt_eval_count": processed, "message": {"content": ""}}, request=request)
        visible = {}
        for match in re.finditer(r"^\[REPORT (\S+) v\d+\] [^:]+: ([A-Z_-]+)\.", json["messages"][-1]["content"], re.M):
            offset = rendered.find(match.group(0))
            if self.harness.count(rendered[offset:]) <= processed:
                visible[match.group(1)] = match.group(2)
        allergy = visible.get("allergy-safety", "not-present")
        coverage = visible.get("coverage-review", "not-present")
        disposition = ("HOLD_FOR_CLINICAL_REVIEW" if allergy == "CONTRAINDICATION" else
                       "HOLD_FOR_AUTHORIZATION" if coverage == "NOT-AUTHORIZED" else "APPROVE_TREATMENT")
        reply = {"allergy": allergy, "medication": visible.get("medication-review", "not-present"),
                 "coverage": coverage, "disposition": disposition}
        return httpx.Response(200, json={"prompt_eval_count": processed,
                                         "message": {"content": __import__("json").dumps(reply)}}, request=request)


def make_case(config_edit=None):
    case = case_json_for(str(DEFINITION), RECORDS)
    if config_edit:
        config_edit(case["harness"])
    return case


def fake_harness(case):
    return load_harness(case, tokenizer=FakeTokenizer(), runtime_checks=False)


class ConfigTests(unittest.TestCase):
    def test_missing_section_fails_loudly(self):
        case = make_case(lambda h: h.pop("observability"))
        with self.assertRaisesRegex(HarnessConfigError, "missing observability"):
            fake_harness(case)

    def test_missing_model_field_fails_loudly(self):
        case = make_case(lambda h: h["model"].pop("window_param"))
        with self.assertRaisesRegex(HarnessConfigError, "harness.model is missing window_param"):
            fake_harness(case)

    def test_unknown_tokenizer_source_fails_loudly(self):
        case = make_case(lambda h: h["model"]["tokenizer"].update(source="guess"))
        with self.assertRaisesRegex(HarnessConfigError, "unsupported tokenizer source"):
            load_harness(case, runtime_checks=False)

    def test_runtime_without_processed_counts_is_rejected(self):
        case = make_case(lambda h: h["observability"].update(exposes_processed_token_count=False))
        with self.assertRaisesRegex(HarnessConfigError, "processed tokens"):
            fake_harness(case)

    def test_every_pipeline_node_must_be_declared(self):
        case = make_case(lambda h: h.update(agents=[a for a in h["agents"] if a["name"] != "coverage-review"]))
        with self.assertRaisesRegex(HarnessConfigError, "coverage-review"):
            run_case(case, 1024, harness=fake_harness(case))

    def test_template_renders_like_the_runtime_template(self):
        harness = fake_harness(make_case())
        self.assertEqual(
            harness.render([{"role": "user", "content": "Hi"}]),
            "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        )
        with self.assertRaises(HarnessConfigError):
            harness.render([{"role": "user", "content": "Hi"}], tools=[{"name": "x"}])


class RunCaseTests(unittest.TestCase):
    def setUp(self):
        self.case = make_case()
        self.harness = fake_harness(self.case)
        self.ollama = FakeOllama(self.harness)
        self.patch = mock.patch.object(httpx, "post", self.ollama)
        self.patch.start()
        trace, _ = run_case(self.case, 100_000, harness=self.harness)
        self.assembled = self.decision(trace)["assembled_tokens"]

    def tearDown(self):
        self.patch.stop()

    @staticmethod
    def decision(trace):
        return next(r for r in trace if r["agent"] == "care-decision")

    def test_one_record_per_step(self):
        trace, _ = run_case(self.case, 100_000, harness=self.harness)
        self.assertEqual(sorted(r["agent"] for r in trace), sorted(self.harness.agents))
        for record in trace:
            for key in ("agent", "window", "assembled_tokens", "processed_tokens", "truncated",
                        "required_evidence", "evidence_visible", "decision"):
                self.assertIn(key, record)

    def test_small_window_is_truncation_at_the_source_and_measured_only(self):
        window = self.assembled // 2
        trace, verdict = run_case(self.case, window, harness=self.harness)
        record = self.decision(trace)
        self.assertEqual(record["window"], window)                 # Hook B: from run state
        self.assertEqual(record["window_sent"], window)
        self.assertTrue(record["window_matches_state"])
        self.assertEqual(record["assembled_tokens"], self.assembled)
        self.assertEqual(record["processed_tokens"], window // 2 + 2)
        self.assertTrue(record["truncated"])                       # Hook C
        self.assertEqual(record["evidence_visible"], {e: None for e in record["required_evidence"]})
        self.assertFalse(record["model_visibility_confirmed"])
        self.assertEqual(record["confirmation"], "measured-only")
        self.assertTrue(all(record["evidence_present_in_prompt"].values()))
        self.assertEqual(record["evidence_in_kept_window_estimate"],
                         {"allergy-safety": False, "medication-review": True, "coverage-review": True})
        self.assertEqual(record["decision"], "APPROVE_TREATMENT")
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["category"], "unreconciled-conflict")
        self.assertEqual(verdict["evidence_check"], "measured-only")

    def test_large_window_is_verified_and_safe(self):
        trace, verdict = run_case(self.case, self.assembled + 50, harness=self.harness)
        record = self.decision(trace)
        self.assertFalse(record["truncated"])
        self.assertTrue(record["token_count_verified"])
        self.assertTrue(all(record["evidence_in_kept_window_estimate"].values()))
        self.assertEqual(record["decision"], "HOLD_FOR_CLINICAL_REVIEW")
        self.assertTrue(verdict["passed"])

    def test_model_visible_runtime_sets_evidence_visible(self):
        case = make_case(lambda h: h["observability"].update(can_capture_model_visible_prompt=True))
        harness = fake_harness(case)
        small = self.decision(run_case(case, self.assembled // 2, harness=harness)[0])
        large = self.decision(run_case(case, self.assembled + 50, harness=harness)[0])
        self.assertEqual(small["evidence_visible"], {e: False for e in small["required_evidence"]})
        self.assertEqual(large["evidence_visible"], {e: True for e in large["required_evidence"]})
        self.assertEqual(large["confirmation"], "measured-and-verified")

    def test_instrumentation_is_read_only(self):
        from healthcare.langgraph_pipeline import run_care_path_graph
        from healthcare.llm_decision_agent import default_config
        from healthcare.system import decode_healthcare_case

        records = decode_healthcare_case(RECORDS)
        for window in (self.assembled // 2, self.assembled + 50):
            with self.subTest(window=window):
                config = default_config(num_ctx=window, model=self.harness.model["name"])
                self.ollama.requests.clear()
                plain = run_care_path_graph(records, decision_mode="llm-context-loss", llm_config=config)
                plain_requests = list(self.ollama.requests)
                self.ollama.requests.clear()
                graph = instrumentation._instrumented_graph(self.harness, __import__("healthcare.langgraph_pipeline",
                                                                                    fromlist=["x"]))
                with mock.patch("healthcare.langgraph_pipeline.build_care_path_graph", lambda: graph):
                    observed = run_care_path_graph(records, decision_mode="llm-context-loss", llm_config=config)
                self.assertEqual(observed["steps"], plain["steps"])          # same decisions and evidence
                self.assertEqual(self.ollama.requests, plain_requests)        # same prompts and windows sent
                self.assertNotIn("trace", plain)

    def test_overflow_error_is_recorded_not_raised(self):
        self.patch.stop()
        with mock.patch.object(httpx, "post", FakeOllama(self.harness, overflow_errors=True)):
            trace, verdict = run_case(self.case, self.assembled // 2, harness=self.harness)
        self.patch.start()
        self.assertEqual(len(trace), 1)
        self.assertTrue(trace[0]["truncated"])
        self.assertTrue(trace[0]["overflow_error"])
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["category"], "runtime-overflow")

    def test_losing_every_finding_is_not_safe(self):
        # At a tiny window the agent sees no report and records consuming nothing. The existing
        # integrity check skips such a consumer; the harness must still call the run unsafe.
        trace, verdict = run_case(self.case, 64, harness=self.harness)
        self.assertTrue(verdict["integrity_passed"])
        self.assertFalse(verdict["required_evidence_carried"])
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["category"], "required-evidence-not-carried")
        self.assertEqual(sorted(verdict["evidence_not_carried"]["care-decision"]),
                         ["allergy-safety", "coverage-review", "medication-review"])

    def test_smallest_safe_window(self):
        probes = []
        best = smallest_safe_window(self.case, 64, self.assembled * 4, resolution=8, probes=probes, harness=self.harness)
        self.assertIsNotNone(best)
        self.assertGreaterEqual(best, self.assembled)                # the whole prompt must fit
        self.assertLess(best - self.assembled, 8)                    # within the resolution
        self.assertTrue(any(not p["passed"] for p in probes))
        self.assertIsNone(smallest_safe_window(self.case, 64, self.assembled // 2, harness=self.harness))

    def test_structured_evidence_match_rejects_wrong_version(self):
        from healthcare.langgraph_pipeline import run_care_path_graph
        from healthcare.system import decode_healthcare_case

        state = run_care_path_graph(decode_healthcare_case(RECORDS))
        steps = dict(state["steps"])
        steps["allergy-safety"] = steps["allergy-safety"].__class__(
            **{**steps["allergy-safety"].__dict__, "finding_version": 9})
        prompt = "[REPORT allergy-safety v2] Allergy safety: CONTRAINDICATION. x.\n" \
                 "[REPORT medication-review v1] Medication review: MEDICATION-ORDER-VALIDATED. y.\n" \
                 "allergy-safety contraindication mentioned in prose only"
        messages = [{"role": "user", "content": prompt}]
        rendered = self.harness.render(messages)
        evidence = instrumentation._evidence(self.harness, self.harness.agent("care-decision"), {"steps": steps},
                                             messages, rendered, None, False)
        self.assertEqual(evidence["present"], {"allergy-safety": False, "medication-review": True,
                                               "coverage-review": False})
        self.assertEqual(evidence["mismatch"]["allergy-safety"]["emitted"], [9, "contraindication"])


def _ollama_has(model):
    try:
        return any(m.get("name") == model for m in httpx.get("http://localhost:11434/api/tags", timeout=3).json()["models"])
    except Exception:
        return False


@unittest.skipUnless(_ollama_has("llama3.1:8b"), "local Ollama with llama3.1:8b is not available")
class LiveHarnessTests(unittest.TestCase):
    def test_real_tokenizer_and_template_match_the_runtime(self):
        harness = load_harness(make_case())
        self.assertEqual(harness.calibration["tokens"], harness.calibration["runtime_tokens"])

    def test_real_run_detects_truncation_and_verifies_counts(self):
        case = make_case()
        small = next(r for r in run_case(case, 1024)[0] if r["agent"] == "care-decision")
        self.assertTrue(small["truncated"])
        self.assertEqual(small["processed_tokens"], 514)
        self.assertFalse(small["evidence_in_kept_window_estimate"]["allergy-safety"])
        trace, verdict = run_case(case, 4096)
        large = next(r for r in trace if r["agent"] == "care-decision")
        self.assertFalse(large["truncated"])
        self.assertTrue(large["token_count_verified"])             # our count == runtime count, live
        self.assertEqual(large["assembled_tokens"], small["assembled_tokens"])
        self.assertTrue(verdict["passed"], verdict)


if __name__ == "__main__":
    unittest.main()
