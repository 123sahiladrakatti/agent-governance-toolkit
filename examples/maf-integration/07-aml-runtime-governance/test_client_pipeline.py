"""Tests for client-pipeline JSON ingestion and AML round-trip parity."""

import copy
import json
import unittest
from datetime import datetime
from pathlib import Path

from agent_sim import build_diamond
from agt_resolver import GraphGovernor
from aml.client_payload import AML_RECORD_CODECS, build_aml_client_payload, simulate_aml_case
from client_pipeline import (
    ClientPipelineError,
    compose_client_payload,
    ingest_client_json,
    ingest_client_pipeline,
    load_pipeline_definition,
)
from domain import Alert, CustomerProfile, GraphOutcome, SANCTIONS_WATCHLIST, Transaction


def no_content_check(_output):
    return {"passed": True, "rail": "unavailable", "reason": "content checks disabled in tests"}


def make_alert(name):
    account = "ACCT-DEMO-001"
    return Alert(
        "AML-INTERACTIVE",
        account,
        (
            Transaction("T-001", account, 2_500.0, "cash_deposit", datetime(2026, 1, 5), "BR-01"),
            Transaction("T-002", account, 1_500.0, "cash_deposit", datetime(2026, 1, 6), "BR-01"),
        ),
        CustomerProfile(account, name, "business owner", 8_000.0),
        "customer and transaction activity review",
    )


def govern_ingested(alert, stale):
    payload = json.dumps(build_aml_client_payload(alert, stale_screening_view=stale))
    ingested = ingest_client_pipeline(payload, AML_RECORD_CODECS)
    governor = GraphGovernor(content_checker=no_content_check, system_description=ingested.description)
    return governor.review(ingested.outcome)


def govern_internal(alert, stale):
    steps = build_diamond(alert, stale_screening_view=stale)
    outcome = GraphOutcome(alert=alert, steps=steps, resource_id=f"aml-alert:{alert.alert_id}")
    return GraphGovernor(content_checker=no_content_check).review(outcome)


class ClientPipelineParityTests(unittest.TestCase):
    def test_round_trip_preserves_trace(self):
        alert = make_alert("Org 1")
        payload = build_aml_client_payload(alert, stale_screening_view=True)
        outcome = ingest_client_json(json.dumps(payload), AML_RECORD_CODECS)
        self.assertEqual(outcome.steps, build_diamond(alert, stale_screening_view=True))
        self.assertEqual(outcome.alert, alert)

    def test_verdict_parity(self):
        for name, stale, expected in (("Jane Smith", False, "allow"), ("Org 1", True, "require_approval")):
            with self.subTest(name=name):
                ingested = govern_ingested(make_alert(name), stale)
                internal = govern_internal(make_alert(name), stale)
                self.assertEqual(ingested.policy_decision["action"], expected)
                self.assertEqual(ingested.policy_decision["action"], internal.policy_decision["action"])
                self.assertEqual(ingested.attribution.category, internal.attribution.category)
                self.assertEqual(ingested.status, internal.status)

    def test_watchlist_conflict_detected_from_payload_alone(self):
        review = govern_ingested(make_alert("Org 1"), stale=True)
        self.assertEqual(review.attribution.category, "unreconciled-conflict")
        self.assertFalse(review.outcome.is_conflict_case)  # no demo label carried in the payload


class ClientPipelineValidationTests(unittest.TestCase):
    def setUp(self):
        self.payload = build_aml_client_payload(make_alert("Jane Smith"))

    def assert_rejected(self, payload, fragment):
        with self.assertRaises(ClientPipelineError) as ctx:
            ingest_client_json(payload, AML_RECORD_CODECS)
        self.assertIn(fragment, str(ctx.exception))

    def test_rejects_wrong_format(self):
        payload = copy.deepcopy(self.payload)
        payload["format"] = "something-else"
        self.assert_rejected(payload, "payload.format")

    def test_rejects_invalid_json(self):
        self.assert_rejected("{not json", "not valid JSON")

    def test_rejects_missing_permissions_for_agent(self):
        payload = copy.deepcopy(self.payload)
        del payload["system"]["permissions"]["decision"]
        self.assert_rejected(payload, "system.permissions")

    def test_rejects_permissions_declared_in_run(self):
        payload = copy.deepcopy(self.payload)
        payload["run"]["steps"][0]["allowed_actions"] = ["anything"]
        self.assert_rejected(payload, "may not appear in a run step")

    def test_rejects_handoff_from_undeclared_parent(self):
        payload = copy.deepcopy(self.payload)
        payload["run"]["steps"][-1]["received"]["intake"] = {"finding": "x"}
        self.assert_rejected(payload, "no declared handoff")

    def test_rejects_undeclared_agent(self):
        payload = copy.deepcopy(self.payload)
        payload["run"]["steps"][0]["agent"] = "rogue-agent"
        self.assert_rejected(payload, "not declared")

    def test_rejects_wrong_field_type(self):
        payload = copy.deepcopy(self.payload)
        payload["run"]["steps"][0]["output"]["version"] = "2"
        self.assert_rejected(payload, "output.version")

    def test_permissions_come_from_declared_system(self):
        payload = copy.deepcopy(self.payload)
        payload["system"]["permissions"]["decision"] = ["something_else"]
        ingested = ingest_client_pipeline(payload, AML_RECORD_CODECS)
        review = GraphGovernor(
            content_checker=no_content_check, system_description=ingested.description
        ).review(ingested.outcome)
        self.assertEqual(review.policy_decision["action"], "deny")


PIPELINE_FILE = Path(__file__).parent / "client_pipelines" / "aml_pipeline.json"


def govern_live_case(definition, name):
    alert = make_alert(name)
    run, records = simulate_aml_case(alert, stale_screening_view=name in SANCTIONS_WATCHLIST)
    payload = json.dumps(compose_client_payload(definition, run, records))
    ingested = ingest_client_pipeline(payload, AML_RECORD_CODECS)
    return GraphGovernor(
        content_checker=no_content_check,
        system_description=ingested.description,
        graph_policy=ingested.graph_policy,
        action_policy=ingested.action_policy,
    ).review(ingested.outcome)


class PipelineDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.definition = load_pipeline_definition(PIPELINE_FILE)

    def test_definition_file_is_static(self):
        self.assertNotIn("run", self.definition)
        self.assertNotIn("records", self.definition)
        self.assertEqual(
            [agent["id"] for agent in self.definition["system"]["agents"]],
            ["intake", "customer-screening", "transaction-analysis", "kyc-entity", "decision"],
        )

    def test_live_cases_through_loaded_pipeline(self):
        clean = govern_live_case(self.definition, "Jane Smith")
        conflict = govern_live_case(self.definition, "Org 1")
        self.assertEqual(clean.policy_decision["action"], "allow")
        self.assertEqual(conflict.policy_decision["action"], "require_approval")
        self.assertEqual(conflict.attribution.category, "unreconciled-conflict")

    def test_live_case_matches_internal_path(self):
        for name, stale in (("Jane Smith", False), ("Org 1", True)):
            with self.subTest(name=name):
                live = govern_live_case(self.definition, name)
                internal = govern_internal(make_alert(name), stale)
                self.assertEqual(live.outcome.steps, internal.outcome.steps)
                self.assertEqual(live.policy_decision["action"], internal.policy_decision["action"])

    def test_policies_from_definition_are_enforced(self):
        definition = copy.deepcopy(self.definition)
        for rule in definition["policy"]["action"]["rules"]:
            if rule["name"] == "require-review-for-unreconciled-conflict":
                rule["action"] = "deny"
        review = govern_live_case(definition, "Org 1")
        self.assertEqual(review.policy_decision["action"], "deny")
        self.assertEqual(review.policy_decision["matched_rule"], "require-review-for-unreconciled-conflict")

    def test_permissions_from_definition_are_enforced(self):
        definition = copy.deepcopy(self.definition)
        definition["system"]["permissions"]["decision"] = ["read_only"]
        self.assertEqual(govern_live_case(definition, "Jane Smith").policy_decision["action"], "deny")

    def test_definition_rejects_embedded_run(self):
        definition = copy.deepcopy(self.definition)
        definition["run"] = {"steps": []}
        with self.assertRaises(ClientPipelineError) as ctx:
            load_pipeline_definition(definition)
        self.assertIn("runs are supplied per case", str(ctx.exception))

    def test_definition_requires_policy(self):
        definition = copy.deepcopy(self.definition)
        del definition["policy"]["action"]
        with self.assertRaises(ClientPipelineError):
            load_pipeline_definition(definition)


if __name__ == "__main__":
    unittest.main()
