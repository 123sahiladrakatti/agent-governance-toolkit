"""Deterministic tests for the isolated real-LLM probe's experimental controls."""

import unittest

from experiments.context_loss_probe import (
    CallResult,
    SCREENING_MARKER,
    classify_visibility,
    decision_prompt,
    parse_json_result,
    recall_prompt,
)


class ContextLossProbeTests(unittest.TestCase):
    def test_all_constructed_conditions_preserve_authoritative_screening_input(self):
        short = decision_prompt(long=False, input_length=0, placement="early")
        long = decision_prompt(long=True, input_length=4000, placement="early")
        self.assertIn(SCREENING_MARKER, short)
        self.assertIn(SCREENING_MARKER, long)
        self.assertGreater(len(long), len(short))

    def test_recall_probe_is_separate_from_decision_task(self):
        prompt = recall_prompt(long=True, input_length=4000, placement="early")
        self.assertIn(SCREENING_MARKER, prompt)
        self.assertIn("Do not make a disposition", prompt)

    def test_structured_response_parses_without_defaulting(self):
        call = CallResult(
            execution_status="SUCCESS",
            raw_response=(
                '{"disposition":"ESCALATE","rationale":"positive screening",'
                '"evidence_cited":["screening"]}'
            ),
        )
        parsed = parse_json_result(
            call, ("disposition", "rationale", "evidence_cited")
        )
        self.assertEqual(parsed.parse_status, "PARSED")
        self.assertEqual(parsed.parsed["disposition"], "ESCALATE")

    def test_invalid_response_remains_invalid(self):
        call = CallResult(execution_status="SUCCESS", raw_response="CLEAR")
        parsed = parse_json_result(
            call, ("disposition", "rationale", "evidence_cited")
        )
        self.assertEqual(parsed.parse_status, "INVALID_JSON")
        self.assertIsNone(parsed.parsed)

    def test_visibility_is_inferred_not_claimed_as_direct_observation(self):
        recall = CallResult(
            execution_status="SUCCESS",
            parsed={"screening_status": "UNKNOWN", "evidence_quote": "not retained"},
            parse_status="PARSED",
        )
        self.assertEqual(
            classify_visibility(
                prompt_has_finding=True,
                recall=recall,
                constrained_token_reduction=True,
            ),
            "INFERRED_NOT_RETAINED",
        )


if __name__ == "__main__":
    unittest.main()
