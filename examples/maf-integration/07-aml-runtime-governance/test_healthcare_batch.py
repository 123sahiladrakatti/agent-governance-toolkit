"""Tests for the backend healthcare Excel batch."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from openpyxl import Workbook

import llm_decision
from client_pipeline import load_pipeline_definition
from healthcare.batch import (
    BATCH_COLUMNS,
    BATCH_LIMIT,
    DEFAULT_BATCH_FILE,
    BatchCase,
    BatchFileError,
    load_healthcare_cases,
    run_healthcare_batch,
    summary_line,
)
from healthcare.system import HealthcareCase

sys.path.insert(0, str(Path(__file__).parent / "batch_cases"))
import make_healthcare_cases  # noqa: E402

PIPELINE_FILE = Path(__file__).parent / "client_pipelines" / "healthcare_pipeline.json"
GOOD_ROW = ("Pat Example", "PAT-DEMO-001", "MBR-8847-2291-05", 45, "bacterial infection",
            "Amoxicillin", "penicillin", True)


def write_xlsx(path, header, rows):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(header)
    for row in rows:
        sheet.append(row)
    workbook.save(path)


def no_pii_checker(output):
    leaked = "Pat Example" in output or "PAT-DEMO-001" in output
    return {"passed": not leaked, "rail": "self_check_output", "reason": "blocked" if leaked else "allowed"}


class SampleFileTests(unittest.TestCase):
    def test_generator_reproduces_the_committed_file_byte_for_byte(self):
        first = make_healthcare_cases.build_workbook_bytes()
        self.assertEqual(first, make_healthcare_cases.build_workbook_bytes())
        self.assertEqual(first, DEFAULT_BATCH_FILE.read_bytes())

    def test_sample_file_parses_into_a_mix_of_cases(self):
        cases = load_healthcare_cases()
        self.assertEqual([c.row for c in cases], [2, 3, 4, 5, 6])
        self.assertEqual(cases[0].case.member_id, "MBR-8847-2291-05")
        pairs = {(c.case.medication, c.case.known_allergy) for c in cases}
        self.assertIn(("Amoxicillin", "penicillin"), pairs)      # a contraindication
        self.assertIn(("Acetaminophen", "none"), pairs)          # a clean case
        self.assertFalse(cases[-1].case.coverage_authorized)


class MalformedFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "cases.xlsx"

    def tearDown(self):
        self.dir.cleanup()

    def assert_rejected(self, fragment, header=BATCH_COLUMNS, rows=(GOOD_ROW,)):
        write_xlsx(self.path, header, rows)
        with self.assertRaises(BatchFileError) as ctx:
            load_healthcare_cases(self.path)
        self.assertIn(fragment, str(ctx.exception))
        return str(ctx.exception)

    def test_missing_and_unexpected_columns(self):
        header = [c if c != "known_allergy" else "allergy" for c in BATCH_COLUMNS]
        message = self.assert_rejected("header row", header=header)
        self.assertIn("missing column(s): known_allergy", message)
        self.assertIn("unexpected column(s): allergy", message)

    def test_bad_values_name_row_and_column(self):
        cases = {
            "Row 2, column 'age'": GOOD_ROW[:3] + ("forty",) + GOOD_ROW[4:],
            "Row 2, column 'medication'": GOOD_ROW[:5] + ("Aspirin",) + GOOD_ROW[6:],
            "Row 2, column 'known_allergy'": GOOD_ROW[:6] + ("latex",) + GOOD_ROW[7:],
            "Row 2, column 'coverage_authorized'": GOOD_ROW[:7] + ("maybe",),
            "Row 2, column 'patient_name'": ("",) + GOOD_ROW[1:],
            "Row 2, column 'member_id'": GOOD_ROW[:2] + ("123-45-6789",) + GOOD_ROW[3:],  # SSN-shaped
        }
        for fragment, row in cases.items():
            with self.subTest(fragment=fragment):
                self.assert_rejected(fragment, rows=(row,))

    def test_error_names_the_right_row(self):
        bad = GOOD_ROW[:3] + (150,) + GOOD_ROW[4:]
        self.assert_rejected("Row 3, column 'age'", rows=(GOOD_ROW, bad))

    def test_empty_missing_and_non_excel_files(self):
        self.assert_rejected("no cases found", rows=())
        with self.assertRaises(BatchFileError):
            load_healthcare_cases(Path(self.dir.name) / "absent.xlsx")
        not_excel = Path(self.dir.name) / "notes.xlsx"
        not_excel.write_text("not a workbook", encoding="utf-8")
        with self.assertRaises(BatchFileError) as ctx:
            load_healthcare_cases(not_excel)
        self.assertIn("could not be read as an Excel workbook", str(ctx.exception))

    def test_blank_member_id_and_flexible_booleans_are_accepted(self):
        write_xlsx(self.path, BATCH_COLUMNS, [GOOD_ROW[:2] + (None,) + GOOD_ROW[3:7] + ("no",), (None,) * 8])
        (only,) = load_healthcare_cases(self.path)
        self.assertEqual(only.case.member_id, "")
        self.assertFalse(only.case.coverage_authorized)


def context_loss_fake():
    reply = ('{"allergy": "not-present", "medication": "MEDICATION-ORDER-VALIDATED", '
             '"coverage": "AUTHORIZED", "disposition": "APPROVE_TREATMENT"}')

    def chat(config, prompt, num_predict):
        if num_predict == 1:
            return {"prompt_eval_count": 514 if len(prompt) > 1500 else 100}
        return {"message": {"content": reply}, "prompt_eval_count": 514}

    return mock.patch.object(llm_decision, "_chat", side_effect=chat)


class BatchRunTests(unittest.TestCase):
    def setUp(self):
        self.definition = load_pipeline_definition(PIPELINE_FILE)
        self.cases = load_healthcare_cases()

    def test_context_loss_batch_is_caught_by_the_trace_check(self):
        progress = []
        with context_loss_fake():
            results = run_healthcare_batch(
                self.cases, "llm-context-loss", self.definition, content_checker=no_pii_checker,
                on_case=lambda position, total, item: progress.append((position, total)),
            )
        self.assertEqual(progress, [(i, 5) for i in range(1, 6)])
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertEqual(result.trace_check, "FLAG")
            self.assertEqual(result.verdict, "HUMAN REVIEW")
            self.assertIn("allergy-safety", result.caught_by)
        self.assertTrue(summary_line(results).startswith("5 cases run · 5 flagged · 0 clean"))

    def test_pii_batch_is_caught_by_the_content_check(self):
        def chat(config, prompt, num_predict):
            leaky = "Pat Example" in prompt
            note = "Pat Example (PAT-DEMO-001) is allergic." if leaky else "The patient's record was checked."
            status = "CONTRAINDICATION" if any(
                pair in prompt for pair in ('"medication_order": "Amoxicillin", "known_allergy": "penicillin"',
                                            '"medication_order": "Ibuprofen", "known_allergy": "NSAID"')
            ) else "NO-CONTRAINDICATION"
            return {"message": {"content": json.dumps({"status": status, "note": note})}}

        with mock.patch.object(llm_decision, "_chat", side_effect=chat):
            results = run_healthcare_batch(self.cases, "llm-pii-leak", self.definition, content_checker=no_pii_checker)
        leaked, *clean = results
        self.assertEqual((leaked.trace_check, leaked.nemo, leaked.verdict), ("PASS", "FLAG", "HUMAN REVIEW"))
        self.assertIn("NeMo", leaked.caught_by)
        self.assertTrue(all(r.nemo == "PASS" and r.trace_check == "PASS" for r in clean))
        self.assertTrue(all(r.verdict == "CONTINUE" for r in clean))

    def test_a_failing_case_becomes_an_error_row_and_the_batch_continues(self):
        calls = {"n": 0}

        def chat(config, prompt, num_predict):
            calls["n"] += 1
            if calls["n"] == 2:
                raise llm_decision.LLMDecisionError("Ollama call failed (simulated)")
            return {"message": {"content": '{"status": "NO-CONTRAINDICATION", "note": "Checked."}'}}

        with mock.patch.object(llm_decision, "_chat", side_effect=chat):
            results = run_healthcare_batch(self.cases, "llm-pii-leak", self.definition, content_checker=no_pii_checker)
        self.assertEqual(len(results), 5)
        self.assertIn("Ollama call failed", results[1].error)
        self.assertIsNone(results[1].verdict)
        self.assertTrue(all(r.error is None for r in results[:1] + results[2:]))
        self.assertIn("1 errored", summary_line(results))

    def test_batch_is_capped(self):
        many = [BatchCase(row, HealthcareCase(f"HC-{row}", f"PAT-{row}", "Synthetic", 40, "dx",
                                               "Acetaminophen", "none", True)) for row in range(2, 15)]
        with mock.patch.object(llm_decision, "_chat", side_effect=lambda c, p, n: {
                "message": {"content": '{"status": "NO-CONTRAINDICATION", "note": "Checked."}'}}):
            results = run_healthcare_batch(many, "llm-pii-leak", self.definition, content_checker=no_pii_checker)
        self.assertEqual(len(results), BATCH_LIMIT)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            run_healthcare_batch(self.cases, "deterministic", self.definition)


if __name__ == "__main__":
    unittest.main()
