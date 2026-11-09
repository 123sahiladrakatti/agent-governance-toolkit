"""Govern one client run: compose the payload, ingest it, and run the governor.

The single path shared by the interactive form and the backend batch, so both
ingest and govern a case exactly the same way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from agt_resolver import GraphGovernor, GraphReview
from client_pipeline import compose_client_payload, ingest_client_pipeline
from domain import GraphOutcome
from graph_integrity import GraphIntegrityResult, check_graph_integrity


@dataclass(frozen=True)
class GovernedRun:
    payload_json: str
    outcome: GraphOutcome
    review: GraphReview
    integrity: GraphIntegrityResult
    agt_live: bool
    final_agent: str


def govern_client_run(
    definition: Mapping[str, Any],
    case_run: Mapping[str, Any],
    case_records: Any,
    codecs: Mapping[str, Any],
    correctness_component: object | None,
    on_stage: Callable[[int, str], None] | None = None,
    content_checker: Callable[[str], Any] | None = None,
) -> GovernedRun:
    """Pipeline definition + one case's run -> ingested payload -> governed outcome.

    ``on_stage(percent, message)`` reports progress between stages (optional).
    ``content_checker`` replaces the default NeMo rail (tests only; the app uses the default).
    Raises ``ClientPipelineError`` when the payload is rejected at ingestion.
    """

    stage = on_stage or (lambda percent, message: None)
    payload_json = json.dumps(compose_client_payload(definition, case_run, case_records), indent=2)
    stage(30, "Framework: ingesting the client-pipeline JSON")
    ingested = ingest_client_pipeline(payload_json, codecs)
    policy_overrides = {
        key: value for key, value in (
            ("graph_policy", ingested.graph_policy), ("action_policy", ingested.action_policy)
        ) if value is not None
    }
    governor = GraphGovernor(
        system_description=ingested.description,
        correctness_component=correctness_component,
        **policy_overrides,
        **({"content_checker": content_checker} if content_checker is not None else {}),
    )
    stage(45, "Framework: running authorization, trace, NeMo, and correctness checks")
    review = governor.review(ingested.outcome)
    stage(85, "Framework: Action Governance is combining signals")
    return GovernedRun(
        payload_json=payload_json,
        outcome=ingested.outcome,
        review=review,
        integrity=check_graph_integrity(ingested.outcome),
        agt_live=governor.agt_live,
        final_agent=ingested.description.final_agent,
    )


__all__ = ["GovernedRun", "govern_client_run"]
