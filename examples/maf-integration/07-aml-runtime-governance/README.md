# AML Runtime Governance Demo

A self-contained Streamlit demo of runtime governance for a long-running AML investigation agent. It compares regular permission checks with enhanced, deterministic behavioral checks that recompute structuring evidence from the alert records.

The simulation uses synthetic transactions and no LLM calls, network calls, database, authentication, or real SAR filing. The default 40-alert run includes a deterministic star case where the agent misreads a deposit amount and incorrectly returns `CLEAR` on a genuine structuring pattern.

## Run

```bash
cd examples/maf-integration/07-aml-runtime-governance
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Run tests with:

```bash
python3 -m unittest -v test_aml_governance.py
```

## Files

- `domain.py`: transactions, customer profiles, alerts, agent actions, and record-relative structuring calculation.
- `agent_sim.py`: reproducible long-running triage/investigation simulation and planted faults.
- `governance.py`: regular authorization checks and enhanced behavioral checks.
- `app.py`: manager-facing Streamlit scoreboard and side-by-side verdict rows.
- `test_aml_governance.py`: deterministic governance behavior tests.
