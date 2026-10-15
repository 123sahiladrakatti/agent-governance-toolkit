# AML Runtime Governance Demo

A self-contained Streamlit demo of runtime governance for a long-running AML investigation agent. It separates deterministic authorization, NeMo Guardrails content monitoring during agent work, and final AGT Action Governance. The NeMo content verdict is consumed by AGT's Action Governance decision; NeMo is not an AGT AuthorityResolver and does not recompute AML correctness.

The simulation uses synthetic transactions and no real SAR filing. When Ollama is available, NeMo checks each authorized agent output with the `self_check_output` rail for explicit prompt-injection content. The session includes two independent governance cases: a transaction-analysis agent misreads deposits and returns an incorrect `CLEAR`, and a screening agent flags a probable watchlist match while disposition consumes an older `no-hit` case-state version and clears the customer without reconciliation. Use the **Case to inspect** selector to switch between them. NeMo checks content only; the behavioral resolver checks record correctness and cross-agent consistency.

In this demo, **potential structuring** means at least three cash deposits, each below $10,000, whose total is at least $30,000 and whose qualifying deposits fall within a five-day window. **Not structuring** means only that this synthetic rule was not met; it is not proof that the activity is legitimate and does not suppress other AML signals. A flag is a policy input, not an accusation or legal conclusion. `CLEAR`, `ESCALATE`, and `FILE_SAR` are simulated dispositions, and `FILE_SAR` is a referral label rather than a real filing.

Flagging is deliberately separated by control: AGT denies undeclared handoffs or actions before execution; NeMo checks the content of authorized outputs; the behavioral resolver recomputes branch findings from source records and detects stale or inconsistent fan-in state; and the final AGT Action Governance policy applies priority rules. Unauthorized actions and ordinary correctness failures are denied, an unreconciled conflict or NeMo-only content flag requires human review, and a clean authorized action is allowed. If NeMo is unavailable, that status remains visible; this demo allows the action only when deterministic authorization and correctness still pass.

For the stale-screening case, state versions make the race visible (`hit`, v2 versus consumed `no-hit`, v1). The resolver marks it `unreconciled-conflict`, and AGT Action Governance routes it to human review rather than committing `CLEAR`. The impact is a prevented false clearance and an explicit reconciliation obligation; the example does not claim an actual sanctions match or make a real compliance determination.

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
- `governance.py`: architecture-aligned deterministic top gate, NeMo middle-layer monitor, record-relative correctness checker, final AGT Action Governance, and bounded session metrics.
- `nemo_guardrails.py`: lazy NeMo output-rail adapter and per-session verdict aggregation.
- `agt_resolver.py`: diamond-path behavioral correctness resolver followed by the separate final Action Governance policy.
- `app.py`: manager-facing Streamlit scoreboard, three-stage verdict rows, and long-running session health metrics.
- `test_aml_governance.py`: deterministic governance behavior tests.
- `test_nemo_integration.py`: isolated adapter and Action Governance tests using mocked NeMo verdicts.
