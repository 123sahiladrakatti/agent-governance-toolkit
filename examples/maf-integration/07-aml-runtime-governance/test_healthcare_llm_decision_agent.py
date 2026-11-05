"""Tests for the healthcare real-LLM care-decision agent and its context-loss run mode."""

import json
import unittest
from pathlib import Path
from unittest import mock

import httpx

import llm_decision
from agt_resolver import GraphGovernor
from client_pipeline import compose_client_payload, ingest_client_pipeline, load_pipeline_definition
from graph_integrity import check_graph_integrity
from healthcare.client_payload import (
    HEALTHCARE_RECORD_CODECS,
    simulate_healthcare_case,
    simulate_healthcare_case_llm,
)
from healthcare.correctness import HealthcareCorrectnessComponent
from healthcare.llm_decision_agent import default_config, parse_decision_output
from healthcare.system import ALLERGY_AGENT, CARE_DECISION_AGENT, HealthcareCase

PIPELINE_FILE = Path(__file__).parent / "client_pipelines" / "healthcare_pipeline.json"


def make_case(medication="Amoxicillin", allergy="penicillin", authorized=True):
    return HealthcareCase(
        "HC-INTERACTIVE", "PAT-DEMO-001", "Synthetic Patient", 45, "bacterial infection",
        medication, allergy, authorized,
    )


def no_content_check(_output):
    return {"passed": True, "rail": "unavailable", "reason": "disabled in tests"}


def govern(run, records):
    definition = load_pipeline_definition(PIPELINE_FILE)
    ingested = ingest_client_pipeline(
        json.dumps(compose_client_payload(definition, run, records)), HEALTHCARE_RECORD_CODECS
    )
    review = GraphGovernor(
        content_checker=no_content_check,
        correctness_component=HealthcareCorrectnessComponent(),
        system_description=ingested.description,
        graph_policy=ingested.graph_policy,
        action_policy=ingested.action_policy,
    ).review(ingested.outcome)
    return review, check_graph_integrity(ingested.outcome)


def fake_ollama(reply_text, kept=514):
    """Ollama-shaped fake: late report lines fit the window, the early allergy report does not."""

    def chat(config, prompt, num_predict):
        if num_predict == 1:  # window measurement call
            return {"prompt_eval_count": kept if len(prompt) > 1500 else 100}
        return {"message": {"content": reply_text}, "prompt_eval_count": kept}

    return mock.patch.object(llm_decision, "_chat", side_effect=chat)


LOST_ALLERGY_REPLY = (
    '{"allergy": "not-present", "medication": "MEDICATION-ORDER-VALIDATED", '
    '"coverage": "AUTHORIZED", "disposition": "APPROVE_TREATMENT"}'
)


class HealthcareParseTests(unittest.TestCase):
    def test_long_dispositions_and_statuses(self):
        disposition, statuses = parse_decision_output(
            'Decision: {"allergy": "CONTRAINDICATION", "medication": "MEDICATION-ORDER-VALIDATED", '
            '"coverage": "AUTHORIZED", "disposition": "hold for clinical review"}'
        )
        self.assertEqual(disposition, "HOLD_FOR_CLINICAL_REVIEW")
        self.assertEqual(statuses["medication"], "medication-order-validated")

    def test_labelled_line_with_long_disposition(self):
        self.assertEqual(
            parse_decision_output("allergy: not-present\ndisposition: HOLD_FOR_AUTHORIZATION")[0],
            "HOLD_FOR_AUTHORIZATION",
        )

    def test_aml_disposition_is_rejected(self):
        with self.assertRaises(llm_decision.LLMDecisionError):
            parse_decision_output('{"disposition": "FILE_SAR"}')


class HealthcareContextLossWiringTests(unittest.TestCase):
    def test_trace_records_only_what_the_agent_consumed(self):
        with fake_ollama(LOST_ALLERGY_REPLY):
            run, _, decision = simulate_healthcare_case_llm(make_case())
        decision_step = next(step for step in run["steps"] if step["agent"] == CARE_DECISION_AGENT)
        allergy_step = next(step for step in run["steps"] if step["agent"] == ALLERGY_AGENT)
        self.assertEqual(allergy_step["output"]["finding"], "contraindication")
        self.assertNotIn(ALLERGY_AGENT, decision_step["received"])
        self.assertEqual(decision_step["output"]["disposition"], "APPROVE_TREATMENT")
        self.assertEqual(decision.not_consumed, (ALLERGY_AGENT,))
        self.assertFalse(decision.in_window[ALLERGY_AGENT])

    def test_governance_flags_lost_contraindication(self):
        with fake_ollama(LOST_ALLERGY_REPLY):
            run, records, _ = simulate_healthcare_case_llm(make_case())
        review, integrity = govern(run, records)
        self.assertIn(ALLERGY_AGENT, integrity.altered_parents)
        self.assertEqual(review.attribution.origin_agent, CARE_DECISION_AGENT)
        self.assertEqual(review.attribution.expected_disposition, "HOLD_FOR_CLINICAL_REVIEW")
        self.assertEqual(review.policy_decision["action"], "require_approval")

    def test_agent_that_consumes_everything_passes(self):
        reply = (
            '{"allergy": "NO-CONTRAINDICATION", "medication": "MEDICATION-ORDER-VALIDATED", '
            '"coverage": "AUTHORIZED", "disposition": "APPROVE_TREATMENT"}'
        )
        with fake_ollama(reply, kept=900):
            run, records, decision = simulate_healthcare_case_llm(make_case("Acetaminophen", "none"))
        review, integrity = govern(run, records)
        self.assertEqual(decision.not_consumed, ())
        self.assertTrue(integrity.passed)
        self.assertEqual(review.policy_decision["action"], "allow")

    def test_deterministic_mode_is_unchanged(self):
        with mock.patch.object(llm_decision, "_chat", side_effect=AssertionError("LLM must not be called")):
            run, records = simulate_healthcare_case(make_case(), stale_allergy_view=True)
        review, _ = govern(run, records)
        self.assertEqual(review.policy_decision["action"], "require_approval")

    def test_healthcare_settings_are_independent_of_aml(self):
        with mock.patch.dict("os.environ", {"HEALTHCARE_LLM_NUM_CTX": "4096", "AML_LLM_NUM_CTX": "8192"}):
            self.assertEqual(default_config().num_ctx, 4096)
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(default_config().model, "llama3.1:8b")


def _ollama_has(model):
    try:
        tags = httpx.get(f"{default_config().base_url}/api/tags", timeout=3).json()
        return any(m.get("name") == model for m in tags.get("models", []))
    except Exception:
        return False


@unittest.skipUnless(_ollama_has("llama3.1:8b"), "local Ollama with llama3.1:8b is not available")
class LiveHealthcareContextLossTests(unittest.TestCase):
    """Real Ollama calls with the validated settings (llama3.1:8b, num_ctx=1024)."""

    def test_real_llm_loses_contraindication_and_governance_catches_it(self):
        run, records, decision = simulate_healthcare_case_llm(
            make_case(), default_config(model="llama3.1:8b", num_ctx=1024)
        )
        self.assertFalse(decision.in_window[ALLERGY_AGENT])
        self.assertIn(ALLERGY_AGENT, decision.not_consumed)
        self.assertEqual(decision.disposition, "APPROVE_TREATMENT")
        review, integrity = govern(run, records)
        self.assertIn(ALLERGY_AGENT, integrity.altered_parents)
        self.assertEqual(review.attribution.origin_agent, CARE_DECISION_AGENT)
        self.assertIn(review.policy_decision["action"], ("require_approval", "deny"))

    def test_large_window_control_sees_everything_and_passes(self):
        run, records, decision = simulate_healthcare_case_llm(
            make_case(), default_config(model="llama3.1:8b", num_ctx=8192, measure_window=False)
        )
        self.assertEqual(decision.consumed.get(ALLERGY_AGENT), "contraindication")
        self.assertEqual(decision.not_consumed, ())
        self.assertEqual(decision.disposition, "HOLD_FOR_CLINICAL_REVIEW")
        review, integrity = govern(run, records)
        self.assertTrue(integrity.passed, integrity.reason)
        self.assertEqual(review.policy_decision["action"], "allow")


if __name__ == "__main__":
    unittest.main()
