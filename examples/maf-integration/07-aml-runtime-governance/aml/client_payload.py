"""AML example pipeline: simulated client agents that run live cases.

This plays the client's side. ``simulate_aml_case`` runs the AML agents on one
live case and returns only that case's observed run and source records; the
client's declared system and policies come from its pipeline definition file
(``client_pipelines/aml_pipeline.json``).
"""

from __future__ import annotations

from typing import Any

from aml.system import (
    AML_SYSTEM_DESCRIPTION,
    DISPOSITION_AGENT,
    KYC_AGENT,
    SANCTIONS_AGENT,
    TRANSACTION_AGENT,
    TRIAGE_AGENT,
    decode_alert,
    encode_alert,
)
from client_pipeline import RecordCodec, run_to_json, to_client_json
from domain import Alert

try:
    # The AML agents run as a LangGraph StateGraph (deterministic nodes, no LLM).
    from aml.langgraph_pipeline import build_langgraph_diamond as _run_pipeline

    AML_RUN_ENGINE = "langgraph"
except ImportError:  # langgraph not installed: fall back to the hand-rolled reference diamond
    from agent_sim import build_diamond as _run_pipeline

    AML_RUN_ENGINE = "build_diamond"

AML_ROLES = {
    TRIAGE_AGENT: "Routes the alert to the screening, transaction, and KYC branches",
    SANCTIONS_AGENT: "Screens the customer against the watchlist",
    TRANSACTION_AGENT: "Analyzes transactions for structuring",
    KYC_AGENT: "Checks activity against the customer's KYC profile",
    DISPOSITION_AGENT: "Merges branch findings into the case disposition",
}

# Lets the AML correctness plug-in receive typed records after ingestion.
AML_RECORD_CODECS: dict[str, RecordCodec] = {
    AML_SYSTEM_DESCRIPTION.schema: (encode_alert, decode_alert),
}


def simulate_aml_case(alert: Alert, stale_screening_view: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the simulated AML agents on one live case; return its ``(run, records)``."""

    steps = _run_pipeline(alert, stale_screening_view=stale_screening_view)
    run = run_to_json(steps, run_id=alert.alert_id, resource_id=f"aml-alert:{alert.alert_id}")
    return run, encode_alert(alert)


def build_aml_client_payload(alert: Alert, stale_screening_view: bool = False) -> dict[str, Any]:
    """Run the AML example agents and return the client-pipeline JSON they would send."""

    steps = _run_pipeline(alert, stale_screening_view=stale_screening_view)
    return to_client_json(
        AML_SYSTEM_DESCRIPTION,
        steps,
        alert,
        run_id=alert.alert_id,
        resource_id=f"aml-alert:{alert.alert_id}",
        roles=AML_ROLES,
    )


__all__ = ["AML_RECORD_CODECS", "AML_ROLES", "AML_RUN_ENGINE", "build_aml_client_payload", "simulate_aml_case"]
