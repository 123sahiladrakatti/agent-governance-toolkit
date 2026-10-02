# AML Runtime Governance Demo

A self-contained Streamlit demo of runtime governance for a long-running AML investigation agent. It separates deterministic authorization, NeMo Guardrails content monitoring during agent work, and final AGT Action Governance. The NeMo content verdict is consumed by AGT's Action Governance decision; NeMo is not an AGT AuthorityResolver and does not recompute AML correctness.

The simulation uses synthetic transactions and no real SAR filing. When Ollama is available, NeMo checks each authorized agent output with the `self_check_output` rail for explicit prompt-injection content. The default 40-alert run includes a deterministic star case where the agent misreads a deposit amount and incorrectly returns `CLEAR` on a genuine structuring pattern. NeMo checks content only; the existing behavioral governance recomputes correctness from the records.

The governance monitor is stateful across the run. It retains only the latest eight alert/account references, creates a checkpoint every five turns, and accumulates a weighted drift score from observed governance failures. The UI reports current turn, context pressure (`turns / 8`), retained context slots, checkpoints, and authorized-but-wrong actions. This demonstrates bounded working state without storing the full transcript.

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

- `domain.py`: transactions, customer profiles, alerts, agent actions, and record-relative structuring calculation.
- `agent_sim.py`: reproducible long-running triage/investigation simulation and planted faults.
- `governance.py`: regular authorization checks, enhanced behavioral checks, and the stateful session monitor.
- `nemo_guardrails.py`: lazy NeMo output-rail adapter and per-session verdict aggregation.
- `agt_resolver.py`: diamond-path behavioral correctness resolver followed by the separate final Action Governance policy.
- `app.py`: manager-facing Streamlit scoreboard, three-stage verdict rows, and long-running session health metrics.
- `test_aml_governance.py`: deterministic governance behavior tests.
- `test_nemo_integration.py`: isolated adapter and Action Governance tests using mocked NeMo verdicts.
