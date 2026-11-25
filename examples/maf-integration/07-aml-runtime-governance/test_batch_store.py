"""Tests for the persistent backend-to-frontend batch result contract."""

import json
import tempfile
import unittest
from pathlib import Path

from batch_store import BatchStore, healthcare_result_document, source_sha256
from batch_worker import process_healthcare_batch
from client_pipeline import load_pipeline_definition
from healthcare.batch import DEFAULT_BATCH_FILE, load_healthcare_cases, run_healthcare_batch

PIPELINE_FILE = Path(__file__).parent / "client_pipelines" / "healthcare_pipeline.json"


def no_content_check(_output):
    return {"passed": True, "rail": "test", "reason": "allowed"}


class BatchStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "results.sqlite3"

    def tearDown(self):
        self.temp.cleanup()

    def test_worker_materializes_full_results_for_fast_indexed_reads(self):
        batch_id = process_healthcare_batch(
            source=DEFAULT_BATCH_FILE,
            database=self.database,
            mode="deterministic",
            content_checker=no_content_check,
        )
        store = BatchStore(self.database)
        batch = store.latest_batch("Healthcare")
        self.assertEqual(batch.batch_id, batch_id)
        self.assertEqual(batch.status, "COMPLETE")
        self.assertEqual(batch.processed_cases, 5)
        rows = store.case_summaries(batch_id)
        self.assertEqual(len(rows), 5)
        document = store.case_document(batch_id, 1)
        self.assertEqual(document["schema"], "governance-batch-result/v1")
        self.assertIn("normalized_payload", document["governance"])
        self.assertIn("review", document["governance"])
        self.assertIn("integrity", document["governance"])
        self.assertEqual(len(source_sha256(DEFAULT_BATCH_FILE)), 64)

    def test_processing_batch_exposes_each_committed_case(self):
        definition = load_pipeline_definition(PIPELINE_FILE)
        cases = load_healthcare_cases()[:2]
        store = BatchStore(self.database)
        batch_id = store.begin_batch(
            domain="Healthcare", mode="deterministic", source_name="cases.xlsx",
            source_hash="abc", total_cases=2,
        )

        def persist(position, result):
            store.save_case(batch_id, position, result, healthcare_result_document(result, "deterministic"))

        run_healthcare_batch(
            cases, "deterministic", definition, content_checker=no_content_check, on_result=persist
        )
        current = store.latest_batch("Healthcare")
        self.assertEqual(current.status, "PROCESSING")
        self.assertEqual(current.processed_cases, 2)
        self.assertEqual(len(store.case_summaries(batch_id)), 2)


if __name__ == "__main__":
    unittest.main()
