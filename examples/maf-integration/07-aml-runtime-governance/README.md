# Generic Runtime Governance Framework

This project has a domain-agnostic governance core with AML retained as one reference implementation. A customer supplies a declared system description, a per-run trace, AGT policy JSON, NeMo settings, and optionally a correctness component. The framework always runs deterministic authorization, generic provenance/version checks, NeMo content monitoring, and final AGT Action Governance. Domain correctness is optional.

The two ingestion inputs are deliberately separate:

- `SystemDescription` declares agents, topology, and permissions. Permissions are never inferred from observed run data.
- `PipelineRunTrace` carries records and observed step results. `adapt_trace` combines it with the declaration to produce the normalized `GraphOutcome`.

The optional customer contract is `CorrectnessComponent.check(trace, records) -> CorrectnessResult`. When it is absent, generic provenance and conflict checks still run and the result explicitly says domain correctness was not configured. The AML implementation is `aml.correctness.AmlCorrectnessComponent`.

The Streamlit simulator is not tied to AML. Its frontend accepts only a user's business context, case information, and critical requirements. Agent count, topology, permissions, context capacity, policies, normalization, and model configuration remain backend concerns. One submission creates the multi-agent run, normalizes its trace, executes governance, and returns Continue, Human Review, or Block.

Context loss occurs naturally when the submitted envelope exceeds the backend handoff capacity; governance then identifies the first receiving agent with altered context, displays what was omitted, and routes the unresolved handoff according to policy. Short inputs that survive every handoff continue normally. Existing companies integrate their own pipeline/log adapter with `pipeline_ingestion.ingest_bundle` behind the frontend rather than asking end users to configure pipelines.

The runtime sequence is:

1. A customer pipeline or log adapter emits the declared system description and per-run trace.
2. `adapt_trace` creates the normalized `GraphOutcome`.
3. AGT checks declared permissions and customer policy.
4. Generic integrity checks inspect provenance, consumed findings, versions, conflicts, and handoffs.
5. NeMo checks authorized output content with the customer configuration.
6. The optional correctness component checks customer records and expected domain results.
7. The normalized signals go to AGT Action Governance.
8. The framework returns Continue, Human Review, or Block with attribution, visualization, and audit evidence.

AML-specific records, terminology, calculations, and correctness attribution remain isolated under `aml/` as a reference implementation; they are not assumptions made by the workbench.

## Run

```bash
cd examples/maf-integration/07-aml-runtime-governance
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Run tests with:

```bash
python3 -m unittest -v test_aml_governance.py
python3 -m unittest -v test_nemo_integration.py test_graph_governance.py
```

NeMo uses `nemo_spike/config` and Ollama at `http://localhost:11434/v1`. If NeMo or Ollama is unavailable, the demo continues with an explicit `unavailable` signal in the UI; that signal is not presented as a NeMo pass. NeMo-only flags require human approval, while failed deterministic authorization or record-relative correctness is denied by AGT Action Governance.

With Ollama intentionally running, check the tuned rail against a benign greeting, a representative AML output, and an obvious prompt injection:

```bash
python -m nemo_spike.smoke_test
```

This live smoke check is separate from the mocked unit tests and may fail if the local model over-blocks or misses the injection scenario.

## Files

- `system_description.py`: generic declared pipeline, per-run trace, ingestion adapter, and graph context codec.
- `correctness.py`: optional customer correctness-component protocol and result contract.
- `graph_integrity.py`: always-on, domain-independent provenance/version/conflict checks.
- `pipeline_ingestion.py`: backend validation and normalization for portable customer governance bundles.
- `context_simulation.py`: domain-neutral generated multi-agent chain with capacity-based context handoffs.
- `aml/`: AML record calculations, correctness component, declared diamond, and record codec.
- `domain.py`: normalized `GraphStep`/`GraphOutcome` plus legacy AML/chain compatibility models.
- `agent_sim.py`: AML reference trace generator and planted faults, normalized through `adapt_trace`.
- `governance.py`: architecture-aligned deterministic top gate, NeMo middle-layer monitor, record-relative correctness checker, final AGT Action Governance, and bounded session metrics.
- `nemo_guardrails.py`: lazy NeMo output-rail adapter with injectable config/model/prompts and verdict aggregation.
- `agt_resolver.py`: generic governor/resolver orchestration with injected policies, system description, and optional correctness.
- `app.py`: domain-agnostic ingestion and governance simulator with an event stream, final decision, attribution, and audit evidence.
- `test_aml_governance.py`: deterministic governance behavior tests.
- `test_nemo_integration.py`: isolated adapter and Action Governance tests using mocked NeMo verdicts.
- `test_generic_governance.py`: non-AML ingestion, permission, provenance, and governor tests.
- `test_pipeline_ingestion.py`: customer bundle normalization and permission-boundary tests.
