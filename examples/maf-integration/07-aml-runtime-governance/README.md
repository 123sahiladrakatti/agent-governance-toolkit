# Generic Runtime Governance Framework

This project has a domain-agnostic governance core with AML and healthcare as two reference implementations. A customer supplies a declared system description, a per-run trace, AGT policy JSON, NeMo settings, and optionally a correctness component. The framework always runs deterministic authorization, generic provenance/version checks, NeMo content monitoring, and final AGT Action Governance. Domain correctness is optional.

The two ingestion inputs are deliberately separate:

- `SystemDescription` declares agents, topology, and permissions. Permissions are never inferred from observed run data.
- `PipelineRunTrace` carries records and observed step results. `adapt_trace` combines it with the declaration to produce the normalized `GraphOutcome`.

The optional customer contract is `CorrectnessComponent.check(trace, records) -> CorrectnessResult`. When it is absent, generic provenance and conflict checks still run and the result explicitly says domain correctness was not configured. The reference implementations are `aml.correctness.AmlCorrectnessComponent` and `healthcare.correctness.HealthcareCorrectnessComponent`.

The Streamlit demo lets a user choose an AML company pipeline or a healthcare company pipeline. That choice loads a different backend JSON definition, so the agents, topology, permissions, policies, records, simulator, and correctness plug-in change. A live case form supplies only domain case data. The generated run is attached to the chosen static definition, ingested through the same `client-pipeline/v1` contract, normalized, and governed. The frontend returns Continue, Human Review, or Block with the actual trace, attribution, and audit proof.

The healthcare reference shows cross-agent context loss: the allergy-safety agent can emit a current v2 contraindication while the care-decision agent consumes an older v1 no-contraindication result. The generic provenance/version checker detects the inconsistency and Action Governance requires human review. A safe medication/allergy combination passes. This is synthetic demonstration logic; real healthcare integrations would use the company's pipeline/log adapter, EHR/formulary records, clinical policies, model/prompts, and validated correctness component.

The runtime sequence is:

1. A customer pipeline or log adapter emits the declared system description and per-run trace.
2. `adapt_trace` creates the normalized `GraphOutcome`.
3. AGT checks declared permissions and customer policy.
4. Generic integrity checks inspect provenance, consumed findings, versions, conflicts, and handoffs.
5. NeMo checks authorized output content with the customer configuration.
6. The optional correctness component checks customer records and expected domain results.
7. The normalized signals go to AGT Action Governance.
8. The framework returns Continue, Human Review, or Block with attribution, visualization, and audit evidence.

Domain-specific records, terminology, simulations, and correctness attribution remain isolated under `aml/` and `healthcare/`; they are not assumptions made by the framework.

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
- `healthcare/`: healthcare case records, simulated care pipeline, record codec, and correctness component.
- `client_pipelines/`: backend client-supplied static definitions for the AML and healthcare examples.
- `domain.py`: normalized `GraphStep`/`GraphOutcome` plus legacy AML/chain compatibility models.
- `agent_sim.py`: AML reference trace generator and planted faults, normalized through `adapt_trace`.
- `governance.py`: architecture-aligned deterministic top gate, NeMo middle-layer monitor, record-relative correctness checker, final AGT Action Governance, and bounded session metrics.
- `nemo_guardrails.py`: lazy NeMo output-rail adapter with injectable config/model/prompts and verdict aggregation.
- `agt_resolver.py`: generic governor/resolver orchestration with injected policies, system description, and optional correctness.
- `app.py`: AML/healthcare reference selector and live case forms over the shared ingestion/governance simulator.
- `test_aml_governance.py`: deterministic governance behavior tests.
- `test_nemo_integration.py`: isolated adapter and Action Governance tests using mocked NeMo verdicts.
- `test_generic_governance.py`: non-AML ingestion, permission, provenance, and governor tests.
- `test_pipeline_ingestion.py`: customer bundle normalization and permission-boundary tests.
- `test_healthcare_pipeline.py`: healthcare definition, clean-run, stale-context, and correctness tests.
