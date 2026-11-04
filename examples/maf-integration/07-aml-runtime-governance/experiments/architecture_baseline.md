# Real-LLM context-loss experiment: architecture baseline

This note records the implementation inspected before the isolated experiment was added.
No pipeline integration is implied by this document.

## Actual execution path

The project does **not** currently contain `aml/langgraph_pipeline.py`, and `langgraph`
is not an installed dependency. The active AML reference path is:

1. `app.py` creates an `Alert` from synthetic user input.
2. `aml.client_payload.simulate_aml_case` calls `agent_sim.build_diamond`.
3. `build_diamond` deterministically computes customer-screening, transaction, and
   KYC findings, then deterministically calls `merge_disposition` for the decision.
4. `client_pipeline.run_to_json` serializes observed steps without deriving permissions.
5. `compose_client_payload` combines the run and records with the selected static
   pipeline declaration and policies.
6. `ingest_client_pipeline` validates the declaration/run boundary and calls
   `system_description.adapt_trace` to create `GraphOutcome`.
7. `GraphGovernor.review` checks handoff authorization, invokes the configured NeMo
   output checker for authorized outputs, runs generic graph integrity plus the optional
   AML correctness component through the AGT resolver, and sends normalized signals to
   the existing Action Governance policy.

## Branch and decision behavior

- Screening is deterministic: exact membership in `SANCTIONS_WATCHLIST` produces
  `hit`, `flagged=True`.
- Transaction analysis is deterministic: record-relative structuring recomputation.
- KYC is deterministic: required profile fields and positive expected cash volume.
- The current decision is deterministic: sanctions or transaction risk produces
  `FILE_SAR`; incomplete KYC produces `ESCALATE`; otherwise `CLEAR`.
- The existing stale-screening demo deliberately substitutes `no-hit` v1 in the
  decision's consumed findings while preserving the authoritative v2 branch finding.

The isolated experiment defines a different, explicit synthetic decision policy:
a confirmed screening hit requires `ESCALATE`, and `FILE_SAR` requires a separate
condition. That policy must not be confused with the existing AML merge rule if later
integration is considered.

## Governance visibility and capability gaps

- Generic integrity sees emitted and consumed findings/versions in `GraphOutcome`.
- AML correctness independently recomputes branch truth from the original `Alert` and
  compares the final disposition with record-derived expectations. It can detect a
  final `CLEAR` that contradicts authoritative risk even when provenance versions match.
- The current trace schema has no field for an LLM rationale, model metadata, prompt,
  evidence visibility, parsing status, or execution status.
- The current NeMo rail is a prompt-injection output check. It receives a formatted
  agent finding/disposition string, not authoritative case evidence, and therefore is
  **not** an independent grounding/correctness evaluator.
- Action Governance already maps generic conflict to human review, correctness failure
  to deny, NeMo flag to human review, and clean signals to allow.
- The UI has no decision-agent execution-mode selector.

## Local experimental runtime observed

- Ollama service version: `0.40.14`
- Installed model: `llama3.1:latest`
- Model identifier: `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`
- Model metadata: Llama 8B, Q4_K_M, declared context length 131072
- Python `ollama` package: not installed; the isolated probe uses Ollama's local HTTP API
  through the Python standard library.

These observations establish the stop gate: only an isolated probe may run next. No
LangGraph, governance, ingestion, or UI integration should occur unless the probe
demonstrates a reproducible real-LLM failure mechanism.
