# Healthcare native governance demo

This Python example uses `MAFKernel` with a native ACS runtime. It runs a
safe request and a blocked request without requiring an LLM credential.

## Step 1: multi-agent skeleton

`multi_agent_demo.py` is intentionally separate from the governance demo. It
contains four local agents with no ACS or policy checks:

- `intake-agent` owns the shared patient-intake task
- `records-agent` reviews the patient record
- `triage-agent` checks for clinical escalation signals
- `scheduling-agent` identifies the next operational step

The `--turns` option makes the same workflow run briefly or for a longer
period, which provides the foundation for adding drift detection later.

```bash
python multi_agent_demo.py --turns 3
python multi_agent_demo.py --turns 30
```

This step does not implement governance, crash recovery, persistence, or
agent-to-agent security. Those concerns are deliberately added in later
steps after the multi-agent interaction is working.

## Integrated governance workflow

`governed_multi_agent_demo.py` connects the four-agent workflow to the native
ACS `AgentControl` runtime. ACS runs before every specialist handoff and the
custom healthcare dispatcher checks:

- sender identity (`intake-agent`)
- allowed recipient (`records-agent`, `triage-agent`, or `scheduling-agent`)
- recipient-specific task shape
- MRN/SSN-like PHI patterns
- turn and handoff budgets
- alignment with the original workflow goal

Run it with:

```bash
python governed_multi_agent_demo.py --turns 10 --drift-turn 7
```

Checkpoints are saved every five turns and also when a drifted handoff pauses
the workflow. To resume from the latest compact checkpoint, use a later drift
point or remove the drift for the resumed run:

```bash
python governed_multi_agent_demo.py --turns 10 --drift-turn 7 --resume
```

State and audit data are written under `.demo-state/` by default. The demo
also exposes `--max-handoffs-per-turn` as the first scale-control knob. The
remaining production work is distributed concurrency, authenticated agent
identity, durable storage, and model-backed behavior.

## Client comparison: regular vs long-running governance

`client_comparison_demo.py` uses the same four-agent care-team workflow to make
the difference visible:

```bash
python client_comparison_demo.py --turns 10 --drift-turn 7
```

To demonstrate state-driven drift rather than injecting a specific drift turn:

```bash
python client_comparison_demo.py --turns 30 --natural-drift
```

In this mode, the workflow accumulates interaction context in a bounded
window. As pressure grows, the simulated agent first broadens its task and
then proposes an external record export. Long-running governance detects the
goal mismatch and pauses before the specialist handoffs. This is a
deterministic behavioral simulator; genuine emergent LLM drift would require
running a real model with conversation state and measuring its outputs.

The regular model checks each request independently for obvious sensitive
identifiers. It does not remember the original goal, so the simulated export
at the drift turn is allowed by that narrow gate.

The long-running model keeps bounded state across turns. It checks the goal,
enforces a turn budget, records periodic checkpoints, and pauses when the
agent changes from patient-intake summarization to external record export.
It retains a compact state summary instead of resending the full transcript,
which is the initial token-efficiency strategy.

| Capability | Existing `main.py` / regular gate | New long-running prototype |
| --- | --- | --- |
| Agent workflow | No real agent; input policy demo | Four agents with coordinated handoffs |
| Safety check | Input pattern check | Input pattern plus goal check |
| Drift detection | None | Goal-relative drift pause |
| Long sessions | No session state | Bounded state and turn budget |
| Recovery | None | Pause and require review; recovery is next step |
| Token strategy | Not applicable | Compact state instead of full history |
| Scale | Single evaluation | Prototype foundation for many sessions |

This comparison is deliberately a stepping stone. It is not yet production
governance, durable crash recovery, authenticated agent-to-agent messaging,
or a distributed multi-agent runtime.

## Client dashboard

`client_dashboard.py` presents the same comparison as a visual, client-facing
demo. It uses synthetic in-memory events only; it does not connect to an EHR,
model provider, database, or backend service.

```bash
streamlit run client_dashboard.py --server.port 8502
```

The dashboard includes workflow controls, decision trajectory charts, a
turn-by-turn event log, capability comparison, and the staged roadmap from
multi-agent interaction to long-running governance and scale.

## Run

```bash
pip install -r requirements.txt
python main.py
```

The manifest under `policies/manifest.yaml` binds the MAF input
intervention point to Rego. Microsoft Agent Framework middleware can be added
with `kernel.as_runtime_middleware()` in a full agent application.
