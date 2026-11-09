"""Tests for the per-case governance PDF report."""

import json
import unittest
from pathlib import Path
from unittest import mock

import llm_decision
from client_pipeline import load_pipeline_definition
from governance_view import build_governance_view
from governed_run import govern_client_run
from healthcare.batch import load_healthcare_cases, run_healthcare_batch
from healthcare.client_payload import HEALTHCARE_RECORD_CODECS, simulate_healthcare_case
from healthcare.correctness import HealthcareCorrectnessComponent
from report_pdf import build_case_report, mask_identifier, mask_name
from test_healthcare_batch import context_loss_fake, no_pii_checker

PIPELINE_FILE = Path(__file__).parent / "client_pipelines" / "healthcare_pipeline.json"
RAW_IDENTIFIERS = (b"Pat Example", b"PAT-DEMO-001", b"MBR-8847-2291-05")
SECTIONS = (b"FINAL VERDICT", b"Governance decision", b"Event breakdown", b"WHERE & WHY")
STEP_TOKENS = (b"Client agents run case", b"JSON ingestion", b"Authorization", b"Generic trace integrity",
               b"NeMo content", b"Correctness plug-in", b"Action Governance gate")


def leaky_allergy_chat(config, prompt, num_predict):
    return {"message": {"content": json.dumps({
        "status": "CONTRAINDICATION",
        "note": "Pat Example (PAT-DEMO-001, member MBR-8847-2291-05) is allergic to penicillin.",
    })}}


class MaskingTests(unittest.TestCase):
    def test_masks(self):
        self.assertEqual(mask_name("Pat Example"), "P** E******")
        self.assertEqual(mask_identifier("PAT-DEMO-001"), "PAT-****-001")
        self.assertEqual(mask_identifier("MBR-8847-2291-05", keep=2), "MBR-****-****-05")
        self.assertEqual(mask_identifier(""), "none")


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        definition = load_pipeline_definition(PIPELINE_FILE)
        first = load_healthcare_cases()[:1]
        with context_loss_fake():
            cls.context_loss = run_healthcare_batch(first, "llm-context-loss", definition,
                                                    content_checker=no_pii_checker)[0]
        with mock.patch.object(llm_decision, "_chat", side_effect=leaky_allergy_chat):
            cls.pii = run_healthcare_batch(first, "llm-pii-leak", definition, content_checker=no_pii_checker)[0]
        run, records = simulate_healthcare_case(first[0].case, stale_allergy_view=True)  # the demo's conflict
        cls.deterministic = govern_client_run(definition, run, records, HEALTHCARE_RECORD_CODECS,
                                              HealthcareCorrectnessComponent(), content_checker=no_pii_checker)
        cls.case = first[0].case

    def build(self, result, mode):
        # The report is built from the stored result; any LLM call here would be a re-run.
        with mock.patch.object(llm_decision, "_chat", side_effect=AssertionError("must not re-run the case")):
            return build_case_report(result.case, mode, result.governed, result.llm, row=result.row)

    def assert_full_detail(self, data, verdict):
        self.assertTrue(data.startswith(b"%PDF-"))
        for token in (*SECTIONS, *STEP_TOKENS, verdict):
            self.assertIn(token, data, token)

    def assert_masked(self, data):
        for raw in RAW_IDENTIFIERS:
            self.assertNotIn(raw, data, raw)
        self.assertIn(b"P** E******", data)
        self.assertIn(b"PAT-****-001", data)

    def test_context_loss_report(self):
        data = self.build(self.context_loss, "llm-context-loss")
        self.assert_full_detail(data, b"HUMAN REVIEW")
        self.assert_masked(data)
        for token in (b"LLM detail: real context loss", b"Emitted", b"Consumed", b"outside", b"num_ctx=1024",
                      b"require-review-for-unreconciled-conflict", b"unreconciled-conflict"):
            self.assertIn(token, data, token)

    def test_pii_report_masks_the_leaked_note(self):
        data = self.build(self.pii, "llm-pii-leak")
        self.assert_full_detail(data, b"HUMAN REVIEW")
        self.assert_masked(data)
        for token in (b"LLM detail: real content failure", b"[REDACTED NAME]", b"[REDACTED ID]",
                      b"[REDACTED MEMBER ID]", b"BLOCKED", b"require-review-for-nemo-flag"):
            self.assertIn(token, data, token)

    def test_report_lists_every_governance_event(self):
        view = build_governance_view("Healthcare", self.pii.governed.review, self.pii.governed.integrity,
                                     self.pii.governed.final_agent, "langgraph", False, None, self.pii.llm)
        data = self.build(self.pii, "llm-pii-leak")
        nemo_rows = [e for e in view.events if e[1].startswith("NeMo content")]
        # Event rows render as "NeMo content · <agent>"; the middle dot is octal · in the PDF stream.
        self.assertEqual(data.count(b"NeMo content \\267"), len(nemo_rows))
        self.assertEqual(len(nemo_rows), 5)
        self.assertEqual(data.count(b"AUTHORIZED"), sum(1 for e in view.events if e[2] == "AUTHORIZED"))

    def phases(self, result, mode):
        llm_decision = result.llm if mode == "llm-context-loss" else None
        llm_allergy = result.llm if mode == "llm-pii-leak" else None
        governed = result.governed if hasattr(result, "governed") else result
        view = build_governance_view("Healthcare", governed.review, governed.integrity, governed.final_agent,
                                     "langgraph", False, llm_decision, llm_allergy)
        return [event[1] for event in view.events]

    def test_events_follow_execution_order_then_governance(self):
        expected_tail_start = "JSON ingestion"
        for result, mode, agent_event in ((self.context_loss, "llm-context-loss", "Decision agent (real LLM)"),
                                          (self.pii, "llm-pii-leak", "Allergy agent (real LLM)")):
            with self.subTest(mode=mode):
                phases = self.phases(result, mode)
                # client agents ran -> the real-LLM agent ran -> then the framework evaluates
                self.assertEqual(phases[:3], ["Client agents run case (Healthcare)", agent_event, expected_tail_start])
                self.assertTrue(phases[3].startswith("Authorization"))
                self.assertEqual(phases[-1], "Action Governance gate")
                # the PDF lists them in the same order
                data = self.build(result, mode)
                agent_token = agent_event.split(" (")[0].encode()  # the PDF escapes "(" as "\("
                table = data.index(b"Event breakdown")  # earlier sections also mention the gate
                positions = [data.index(token, table) for token in (b"Client agents run case", agent_token,
                                                             b"JSON ingestion", b"Authorization", b"Generic trace",
                                                             b"Correctness plug-in", b"Action Governance gate")]
                self.assertEqual(positions, sorted(positions))
        deterministic = self.phases(self.deterministic, "deterministic")
        self.assertEqual(deterministic[:2], ["Client agents run case (Healthcare)", expected_tail_start])

    def test_deterministic_run_report(self):
        data = build_case_report(self.case, "deterministic", self.deterministic, None)
        self.assert_full_detail(data, b"HUMAN REVIEW")
        self.assertNotIn(b"LLM detail", data)
        self.assert_masked(data)


if __name__ == "__main__":
    unittest.main()
