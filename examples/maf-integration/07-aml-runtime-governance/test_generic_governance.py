"""Tests for governance checks that do not depend on AML records."""

import json
import unittest
from types import SimpleNamespace

from agt_resolver import (
    ACTION_GOVERNANCE_POLICY,
    BehavioralAuthorityResolver,
    GraphGovernor,
    graph_context,
)
from domain import GraphOutcome, GraphStep
from graph_integrity import check_graph_integrity
from system_description import (
    AgentDescription,
    PipelineRunTrace,
    SystemDescription,
    TraceStep,
    adapt_trace,
)


def step(agent, node, parents=(), finding="ok", version=1, received=None, received_versions=None):
    return GraphStep(
        agent=agent,
        node_id=node,
        action_type="process",
        allowed_actions=frozenset({"process"}),
        received_from=tuple(parents),
        finding=finding,
        flagged=False,
        received_findings_by_parent=received or {},
        finding_version=version,
        received_versions_by_parent=received_versions or {},
    )


class GenericGraphIntegrityTests(unittest.TestCase):
    def test_accepts_current_unchanged_parent_findings(self):
        producer = step("producer", "A", finding="ready", version=2)
        consumer = step(
            "consumer", "B", parents=("producer",), finding="complete",
            received={"producer": "ready"}, received_versions={"producer": 2},
        )

        result = check_graph_integrity(GraphOutcome(alert=object(), steps=(producer, consumer)))

        self.assertTrue(result.passed)

    def test_detects_stale_parent_without_domain_records(self):
        producer = step("producer", "A", finding="updated", version=2)
        consumer = step(
            "consumer", "B", parents=("producer",), finding="complete",
            received={"producer": "old"}, received_versions={"producer": 1},
        )

        result = check_graph_integrity(GraphOutcome(alert=object(), steps=(producer, consumer)))

        self.assertFalse(result.passed)
        self.assertEqual(result.category, "unreconciled-conflict")
        self.assertEqual(result.origin_agent, "consumer")
        self.assertEqual(result.stale_parents, ("producer",))
        self.assertIn("stale case state", result.reason)


class GenericIngestionTests(unittest.TestCase):
    def setUp(self):
        self.description = SystemDescription(
            name="document-approval",
            schema="document-approval/v1",
            agents=(
                AgentDescription("author", "A", frozenset({"draft"})),
                AgentDescription("approver", "B", frozenset({"approve"})),
            ),
            topology={"author": (), "approver": ("author",)},
            final_agent="approver",
            final_action="publish_document",
            resource_prefix="document",
        )
        self.trace = PipelineRunTrace(
            records={"document_id": "DOC-7"},
            resource_id="document:DOC-7",
            steps=(
                TraceStep("author", "draft", "ready", finding_version=2),
                TraceStep(
                    "approver",
                    "approve",
                    "approved",
                    disposition="PUBLISH",
                    received_findings_by_parent={"author": "ready"},
                    received_versions_by_parent={"author": 2},
                ),
            ),
        )

    def _graph_policy(self):
        return json.dumps({
            "apiVersion": "governance.toolkit/v1",
            "version": "1.0",
            "name": "document-graph-governance",
            "agents": ["*"],
            "scope": "global",
            "default_action": "allow",
            "rules": [
                {"name": "allow-authorized-handoff", "stage": "pre_tool",
                 "condition": "handoff.status == 'authorized'", "action": "allow", "priority": 100},
                {"name": "deny-unauthorized-handoff", "stage": "pre_tool",
                 "condition": "handoff.status != 'authorized'", "action": "deny", "priority": 100},
                {"name": "deny-without-document-graph", "stage": "post_tool",
                 "condition": "governance_graph.schema != 'document-approval/v1'",
                 "action": "deny", "priority": 100},
            ],
        })

    def test_adapter_attaches_declared_permissions_and_topology(self):
        outcome = adapt_trace(self.description, self.trace)

        self.assertEqual(outcome.step("approver").received_from, ("author",))
        self.assertEqual(outcome.step("approver").allowed_actions, frozenset({"approve"}))
        self.assertEqual(outcome.resource_id, "document:DOC-7")

    def test_non_aml_pipeline_runs_without_correctness_component(self):
        outcome = adapt_trace(self.description, self.trace)
        allow_content = lambda _: {
            "passed": True,
            "rail": "self_check_output",
            "reason": "test allow",
        }

        review = GraphGovernor(
            content_checker=allow_content,
            correctness_component=None,
            graph_policy=self._graph_policy(),
            action_policy=ACTION_GOVERNANCE_POLICY,
            system_description=self.description,
        ).review(outcome)

        self.assertTrue(review.attribution.passed)
        self.assertEqual(review.policy_decision["action"], "allow")
        self.assertEqual(review.handoffs[0].sender, "author")

    def test_generic_context_round_trip_uses_customer_codec_and_schema(self):
        outcome = adapt_trace(self.description, self.trace)
        context = graph_context(outcome, self.description)
        resolver = BehavioralAuthorityResolver(None, self.description)

        decision = resolver.resolve(SimpleNamespace(
            context={"governance_graph": context},
            delegation=None,
        ))

        self.assertEqual(decision.decision, "allow")
        self.assertEqual(resolver.last_attribution.merged_disposition, "PUBLISH")

    def test_observed_action_does_not_expand_declared_permissions(self):
        altered = PipelineRunTrace(
            records=self.trace.records,
            resource_id=self.trace.resource_id,
            steps=(
                self.trace.steps[0],
                TraceStep(
                    "approver",
                    "delete_document",
                    "approved",
                    disposition="PUBLISH",
                    received_findings_by_parent={"author": "ready"},
                    received_versions_by_parent={"author": 2},
                ),
            ),
        )

        outcome = adapt_trace(self.description, altered)

        self.assertNotIn("delete_document", outcome.step("approver").allowed_actions)


if __name__ == "__main__":
    unittest.main()
