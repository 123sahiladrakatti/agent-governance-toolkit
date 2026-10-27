"""Tests for customer bundle ingestion used by the simulator."""

import unittest

from context_simulation import build_context_bundle
from graph_integrity import check_graph_integrity
from pipeline_ingestion import ingest_bundle


def bundle(action="release", consumed_version=3):
    schema = "inventory/v1"
    graph_policy = {
        "apiVersion": "governance.toolkit/v1",
        "version": "1.0",
        "name": "inventory-graph-policy",
        "agents": ["*"],
        "scope": "global",
        "default_action": "allow",
        "rules": [],
    }
    action_policy = {
        "apiVersion": "governance.toolkit/v1",
        "version": "1.0",
        "name": "inventory-action-policy",
        "agents": ["*"],
        "scope": "global",
        "default_action": "deny",
        "rules": [],
    }
    return {
        "system_description": {
            "name": "inventory",
            "schema": schema,
            "agents": [
                {"id": "scanner", "node_id": "S", "allowed_actions": ["scan"]},
                {"id": "releaser", "node_id": "R", "allowed_actions": ["release"]},
            ],
            "topology": {"scanner": [], "releaser": ["scanner"]},
            "final_agent": "releaser",
        },
        "trace": {
            "resource_id": "inventory:42",
            "records": {"item": 42},
            "steps": [
                {"agent": "scanner", "action_type": "scan", "finding": "ready", "finding_version": 3},
                {
                    "agent": "releaser",
                    "action_type": action,
                    "finding": "released",
                    "disposition": "CONTINUE",
                    "received_findings_by_parent": {"scanner": "ready"},
                    "received_versions_by_parent": {"scanner": consumed_version},
                },
            ],
        },
        "policies": {"graph": graph_policy, "action": action_policy},
    }


class PipelineIngestionTests(unittest.TestCase):
    def test_customer_bundle_is_normalized(self):
        ingested = ingest_bundle(bundle(), "inventory-backend.json")

        self.assertEqual(ingested.description.name, "inventory")
        self.assertEqual(ingested.outcome.resource_id, "inventory:42")
        self.assertEqual(ingested.outcome.step("releaser").received_from, ("scanner",))
        self.assertEqual(ingested.source_name, "inventory-backend.json")

    def test_observed_action_does_not_change_declared_permissions(self):
        ingested = ingest_bundle(bundle(action="delete_inventory"))

        step = ingested.outcome.step("releaser")
        self.assertEqual(step.action_type, "delete_inventory")
        self.assertEqual(step.allowed_actions, frozenset({"release"}))

    def test_unknown_agent_is_rejected_during_backend_ingestion(self):
        value = bundle()
        value["trace"]["steps"][0]["agent"] = "undeclared-agent"

        with self.assertRaises(KeyError):
            ingest_bundle(value)

    def test_guided_simulation_attributes_context_loss_to_receiver(self):
        value = build_context_bundle(
            "Prepare a recommendation using the complete customer context. " * 12,
            ["Manager approval is mandatory"],
            agent_count=4,
            handoff_capacity=150,
        )
        ingested = ingest_bundle(value)

        result = check_graph_integrity(ingested.outcome)

        self.assertFalse(result.passed)
        self.assertEqual(result.origin_agent, "analysis-agent")
        self.assertEqual(result.altered_parents, ("intake-agent",))
        self.assertTrue(ingested.outcome.alert["handoff_losses"])

    def test_guided_simulation_passes_when_context_fits(self):
        value = build_context_bundle(
            "Short request",
            ["Preserve this"],
            agent_count=3,
            handoff_capacity=500,
        )
        ingested = ingest_bundle(value)

        result = check_graph_integrity(ingested.outcome)

        self.assertTrue(result.passed)
        self.assertEqual(ingested.outcome.alert["handoff_losses"], [])


if __name__ == "__main__":
    unittest.main()
