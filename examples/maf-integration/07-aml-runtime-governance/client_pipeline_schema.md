# Client-pipeline JSON (`client-pipeline/v1`)

A client sends one JSON object per pipeline run. The governance framework
ingests it with `client_pipeline.ingest_client_json(payload)`, which validates
the payload and returns the normalized `GraphOutcome` that every governance
stage runs on. The format is domain-agnostic. The AML demo is one instance; see
`aml/client_payload.py`.

```jsonc
{
  "format": "client-pipeline/v1",

  // Declared system: trusted and authoritative. Permissions and topology come only from here.
  "system": {
    "name": "aml-diamond",
    "schema": "aml-diamond/v1",           // identifies the records layout and the policy family
    "agents": [
      {"id": "intake", "node_id": "A", "role": "Routes the alert to the branches"}
    ],
    "topology":    {"intake": [], "decision": ["customer-screening", "..."]},  // agent -> parents
    "permissions": {"intake": ["fan_out_alert"], "...": ["..."]},           // agent -> allowed actions
    "final_agent": "decision",
    "final_action": "commit_disposition",     // optional, default "commit_result"
    "handoff_action": "aml_graph_handoff",    // optional, default "graph_handoff"
    "resolver_action": "resolve_behavioral_correctness",  // optional
    "resource_prefix": "aml-alert",           // optional, default "pipeline-run"
    "escalation_outcomes": ["ESCALATE"]       // optional
  },

  // Observed run: what each agent did and what it received.
  "run": {
    "run_id": "AML-INTERACTIVE",
    "resource_id": "aml-alert:AML-INTERACTIVE",   // optional, default "<resource_prefix>:<run_id>"
    "steps": [
      {
        "agent": "decision",
        "action": "merge_disposition",
        "output": {
          "finding": "CLEAR (3/3 branches clean)",
          "flagged": false,
          "version": 3,
          "disposition": "CLEAR",              // optional domain outcome
          "target": "ACCT-DEMO-001",           // optional: the resource the agent acted on
          "evidence": {"values": [], "aggregate": 0.0}   // optional numeric evidence
        },
        "received": {                          // per-parent handoff, as the agent consumed it
          "customer-screening": {"finding": "no-hit", "version": 1, "values": []}
        }
      }
    ]
  },

  // Source records for the run. Opaque to the framework; used by an optional correctness plug-in.
  "records": {},

  // Optional. Deterministic AGT policy documents. Omitted means the framework default policy.
  "policy": {"graph": {}, "action": {}}
}
```

## Pipeline definition (`client-pipeline-definition/v1`)

A client provides its pipeline once, as a definition file such as
`client_pipelines/aml_pipeline.json`. It holds only the static parts: `system`
and `policy`, with the same shapes as above (`policy` is required here). It holds
no `run` and no `records`, and a definition that includes either is rejected.

The run and records are supplied per case at runtime. In the demo, each case
entered in the form is executed by the simulated client agents
(`aml/client_payload.simulate_aml_case`). `compose_client_payload(definition, run, records)`
then combines the case with the loaded definition into one `client-pipeline/v1`
payload, which is ingested and governed as described below. The enforced
permissions and policies are therefore the ones in the client's definition file.

```jsonc
{
  "format": "client-pipeline-definition/v1",
  "description": "free text",
  "system": { /* agents, topology, permissions, ... */ },
  "policy": {"graph": { /* AGT policy */ }, "action": { /* AGT policy */ }}
}
```

Load it with `client_pipeline.load_pipeline_definition(path)`, which validates
`system` and `policy` before any case runs.

## Ingestion rules

- `format` must be `client-pipeline/v1`.
- Every agent must appear exactly once in `system.agents`, `system.topology`, and `system.permissions`.
  Topology may only reference declared agents, and `final_agent` must be declared.
- Permissions are never inferred from the run. A run step that carries `permissions`, `allowed_actions`,
  or `received_from` is rejected.
- Each run step names a declared agent, at most once. The run must include `final_agent`.
- `received` may only list parents that `system.topology` declares for that agent.
- `records` is passed through unchanged, unless the caller registers a record codec for `system.schema`
  (the AML demo registers one so its correctness plug-in gets typed records).
- A malformed payload raises `ClientPipelineError` with the path of the offending field, and nothing
  is governed.

## What governance reads from it

| Stage | Reads |
|---|---|
| Authorization | `system.topology`, `system.permissions`, each step's `action` |
| Trace integrity | each step's `output.finding` / `output.version` vs. downstream `received` |
| NeMo content | each step's `output` |
| Optional correctness | `records` plus step outputs (domain plug-in) |
| Action Governance | the signals above, evaluated with `policy` (or the default policy) |
