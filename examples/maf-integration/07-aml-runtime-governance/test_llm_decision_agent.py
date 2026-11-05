"""Tests for the real-LLM decision agent and the context-loss run mode."""

import json
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import httpx

import llm_decision as llm_agent
from agt_resolver import GraphGovernor
from aml.client_payload import AML_RECORD_CODECS, simulate_aml_case, simulate_aml_case_llm
from aml.langgraph_pipeline import build_langgraph_diamond
from aml.llm_decision_agent import LLMDecisionConfig, LLMDecisionError, parse_decision_output
from aml.system import DISPOSITION_AGENT, SANCTIONS_AGENT
from client_pipeline import compose_client_payload, ingest_client_pipeline, load_pipeline_definition
from domain import Alert, CustomerProfile, Transaction
from graph_integrity import check_graph_integrity

PIPELINE_FILE = Path(__file__).parent / "client_pipelines" / "aml_pipeline.json"


def make_alert(name):
    account = "ACCT-DEMO-001"
    return Alert(
        "AML-INTERACTIVE", account,
        (
            Transaction("T-001", account, 2_500.0, "cash_deposit", datetime(2026, 1, 5), "BR-01"),
            Transaction("T-002", account, 1_500.0, "cash_deposit", datetime(2026, 1, 6), "BR-01"),
        ),
        CustomerProfile(account, name, "business owner", 8_000.0),
        "customer and transaction activity review",
    )


def no_content_check(_output):
    return {"passed": True, "rail": "unavailable", "reason": "disabled in tests"}


def govern(run, records, content_checker=no_content_check):
    definition = load_pipeline_definition(PIPELINE_FILE)
    ingested = ingest_client_pipeline(json.dumps(compose_client_payload(definition, run, records)), AML_RECORD_CODECS)
    review = GraphGovernor(
        content_checker=content_checker,
        system_description=ingested.description,
        graph_policy=ingested.graph_policy,
        action_policy=ingested.action_policy,
    ).review(ingested.outcome)
    return review, check_graph_integrity(ingested.outcome)


class ParseDecisionOutputTests(unittest.TestCase):
    def test_clean_json(self):
        disposition, statuses = parse_decision_output(
            '{"screening": "not-present", "transactions": "NOT-STRUCTURING", "kyc": "VERIFIED", "disposition": "CLEAR"}'
        )
        self.assertEqual(disposition, "CLEAR")
        self.assertEqual(statuses, {"screening": "not-present", "transactions": "not-structuring", "kyc": "verified"})

    def test_json_in_code_fence_with_prose_and_odd_casing(self):
        disposition, statuses = parse_decision_output(
            'Sure! Here is my decision:\n```json\n{"Screening": "HIT", "Transactions": "not structuring", '
            '"KYC": "verified", "Disposition": "file sar"}\n```\nLet me know.'
        )
        self.assertEqual(disposition, "FILE_SAR")
        self.assertEqual(statuses["screening"], "hit")
        self.assertEqual(statuses["transactions"], "not-structuring")

    def test_answer_split_across_json_objects(self):
        disposition, statuses = parse_decision_output(
            '{"screening": "HIT", "transactions": "NOT-STRUCTURING", "kyc": "VERIFIED"} {"disposition": "FILE_SAR"}'
        )
        self.assertEqual(disposition, "FILE_SAR")
        self.assertEqual(statuses, {"screening": "hit", "transactions": "not-structuring", "kyc": "verified"})

    def test_labelled_lines_without_json(self):
        disposition, statuses = parse_decision_output(
            "screening: not-present\ntransactions: not-structuring\nkyc: verified\nDisposition: ESCALATE"
        )
        self.assertEqual(disposition, "ESCALATE")
        self.assertEqual(statuses["screening"], "not-present")

    def test_single_bare_keyword(self):
        self.assertEqual(parse_decision_output("I would CLEAR this case.")[0], "CLEAR")

    def test_ambiguous_or_missing_disposition_raises(self):
        for text in ("Either FILE_SAR or CLEAR depending on screening.", "No idea.", '{"disposition": "maybe"}'):
            with self.subTest(text=text), self.assertRaises(LLMDecisionError) as ctx:
                parse_decision_output(text)
            self.assertEqual(ctx.exception.raw_output, text)


def fake_ollama(reply_text, kept=514):
    """Ollama-shaped fake: report lines near the end fit the window, the early one does not."""

    def chat(config, prompt, num_predict):
        if num_predict == 1:  # window measurement call
            return {"prompt_eval_count": kept if len(prompt) > 1500 else 100}
        return {"message": {"content": reply_text}, "prompt_eval_count": kept}

    return mock.patch.object(llm_agent, "_chat", side_effect=chat)


LOST_SCREENING_REPLY = (
    '{"screening": "not-present", "transactions": "not-structuring", "kyc": "verified", "disposition": "CLEAR"}'
)


class ContextLossModeWiringTests(unittest.TestCase):
    """Pipeline + governance wiring with a fake LLM; the live test below uses a real one."""

    def test_trace_records_only_what_the_agent_consumed(self):
        with fake_ollama(LOST_SCREENING_REPLY):
            run, records, decision = simulate_aml_case_llm(make_alert("Org 1"))
        decision_step = next(step for step in run["steps"] if step["agent"] == DISPOSITION_AGENT)
        screening_step = next(step for step in run["steps"] if step["agent"] == SANCTIONS_AGENT)
        self.assertEqual(screening_step["output"]["finding"], "hit")      # the branch emitted the hit
        self.assertNotIn(SANCTIONS_AGENT, decision_step["received"])       # the agent never consumed it
        self.assertEqual(decision_step["output"]["disposition"], "CLEAR")
        self.assertEqual(decision.not_consumed, (SANCTIONS_AGENT,))
        self.assertEqual(decision.in_window[SANCTIONS_AGENT], False)

    def test_governance_flags_lost_finding_and_routes_to_review(self):
        with fake_ollama(LOST_SCREENING_REPLY):
            run, records, _ = simulate_aml_case_llm(make_alert("Org 1"))
        review, integrity = govern(run, records)
        self.assertFalse(integrity.passed)
        self.assertIn(SANCTIONS_AGENT, integrity.altered_parents)
        self.assertEqual(review.attribution.origin_agent, DISPOSITION_AGENT)
        self.assertEqual(review.attribution.expected_disposition, "FILE_SAR")
        self.assertEqual(review.policy_decision["action"], "require_approval")

    def test_agent_that_consumes_everything_passes(self):
        reply = '{"screening": "no-hit", "transactions": "not-structuring", "kyc": "verified", "disposition": "CLEAR"}'
        with fake_ollama(reply, kept=900):
            run, records, decision = simulate_aml_case_llm(make_alert("Jane Smith"))
        review, integrity = govern(run, records)
        self.assertEqual(decision.not_consumed, ())
        self.assertTrue(integrity.passed)
        self.assertEqual(review.policy_decision["action"], "allow")

    def test_unparseable_output_surfaces_instead_of_defaulting(self):
        with fake_ollama("I cannot decide this case."), self.assertRaises(LLMDecisionError):
            simulate_aml_case_llm(make_alert("Org 1"))

    def test_unreachable_ollama_surfaces_clearly(self):
        config = LLMDecisionConfig(base_url="http://127.0.0.1:9", timeout_s=2)
        with self.assertRaises(LLMDecisionError) as ctx:
            simulate_aml_case_llm(make_alert("Org 1"), config)
        self.assertIn("Ollama call failed", str(ctx.exception))

    def test_deterministic_mode_is_unchanged(self):
        with mock.patch.object(llm_agent, "_chat", side_effect=AssertionError("LLM must not be called")):
            run, records = simulate_aml_case(make_alert("Org 1"), stale_screening_view=True)
            steps = build_langgraph_diamond(make_alert("Jane Smith"))
        review, _ = govern(run, records)
        self.assertEqual(review.policy_decision["action"], "require_approval")
        self.assertEqual(steps[-1].disposition, "CLEAR")


def _ollama_has(model):
    try:
        tags = httpx.get(f"{LLMDecisionConfig().base_url}/api/tags", timeout=3).json()
        return any(m.get("name") == model for m in tags.get("models", []))
    except Exception:
        return False


@unittest.skipUnless(_ollama_has(LLMDecisionConfig().model), "local Ollama with the decision model is not available")
class LiveContextLossTests(unittest.TestCase):
    """Real Ollama calls with the validated settings (qwen2.5:3b, num_ctx=1024)."""

    def test_real_llm_loses_screening_hit_and_governance_catches_it(self):
        run, records, decision = simulate_aml_case_llm(make_alert("Org 1"))
        self.assertTrue(decision.prompt_truncated)
        self.assertEqual(decision.in_window[SANCTIONS_AGENT], False)
        self.assertIn(SANCTIONS_AGENT, decision.not_consumed)
        self.assertNotEqual(decision.disposition, "FILE_SAR")
        review, integrity = govern(run, records)
        self.assertIn(SANCTIONS_AGENT, integrity.altered_parents)
        self.assertEqual(review.attribution.origin_agent, DISPOSITION_AGENT)
        self.assertIn(review.policy_decision["action"], ("require_approval", "deny"))

    def test_large_window_control_sees_the_hit(self):
        config = LLMDecisionConfig(num_ctx=8192, measure_window=False)
        _, _, decision = simulate_aml_case_llm(make_alert("Org 1"), config)
        self.assertEqual(decision.consumed.get(SANCTIONS_AGENT), "hit")
        self.assertEqual(decision.disposition, "FILE_SAR")


if __name__ == "__main__":
    unittest.main()
