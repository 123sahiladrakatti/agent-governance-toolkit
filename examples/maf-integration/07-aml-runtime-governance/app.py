
"""Runtime governance demo with swappable AML and healthcare client pipelines.

Run: streamlit run app.py
"""

from __future__ import annotations

import html
import json
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st

from agt_resolver import GraphGovernor
from aml.client_payload import AML_RECORD_CODECS, AML_RUN_ENGINE, simulate_aml_case
from aml.correctness import AmlCorrectnessComponent
from client_pipeline import (
    ClientPipelineError,
    compose_client_payload,
    ingest_client_pipeline,
    load_pipeline_definition,
)
from domain import (
    Alert,
    CustomerProfile,
    SANCTIONS_WATCHLIST,
    Transaction,
)
from graph_integrity import check_graph_integrity
from healthcare.client_payload import (
    HEALTHCARE_RUN_ENGINE,
    HEALTHCARE_RECORD_CODECS,
    allergy_finding,
    simulate_healthcare_case,
)
from healthcare.correctness import HealthcareCorrectnessComponent
from healthcare.system import HealthcareCase, MEDICATION_ALLERGENS


st.set_page_config(page_title="Runtime Agent Governance", page_icon="◆", layout="wide")

st.markdown("""
<style>
:root { --ink:#17212b; --muted:#667580; --line:#d7e0e5; --paper:#f5f7f6; --green:#176b45; --green-bg:#e7f5ed; --red:#b12632; --red-bg:#fff0f1; --gold:#9a6819; --amber-bg:#fbf3e2; --blue:#1f5c8a; --blue-bg:#eaf2f9; --purple:#67558b; --purple-bg:#f4f0fa; }
html,body,[class*="css"] { font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,sans-serif; }
.stApp { background:var(--paper); color:var(--ink); }
.block-container { max-width:1180px; padding:1.7rem 1.5rem 4rem; }
.hero { border:1px solid var(--line); background:#fff; padding:1.45rem 1.6rem; }
.hero .kicker { color:var(--blue); font:700 .75rem ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.05em; }
.hero h1 { margin:.3rem 0 .45rem; font-size:clamp(1.8rem,3.2vw,2.55rem); letter-spacing:-.035em; }
.hero p { max-width:880px; margin:0; color:var(--muted); line-height:1.55; }
.section-label { color:var(--muted); font:700 .74rem ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.04em; margin:1.8rem 0 .35rem; }
.section-title { margin:.1rem 0 .35rem; font-size:1.35rem; }
.section-help { color:var(--muted); margin:0 0 .9rem; font-size:.9rem; line-height:1.5; }
.watchlist { border:1px solid #ecd9ad; background:var(--amber-bg); padding:.75rem .9rem; font-size:.85rem; margin:.7rem 0; }
.simulator { border:1px solid var(--line); background:#fff; padding:1.2rem; }
.event { display:grid; grid-template-columns:18px 250px 105px 1fr; gap:.65rem; align-items:start; padding:.62rem 0; border-bottom:1px solid var(--line); font-size:.85rem; }
.event:last-child { border-bottom:0; }
.event .dot { width:11px; height:11px; border-radius:50%; margin-top:.24rem; background:var(--muted); }
.event.pass .dot { background:var(--green); }
.event.fail .dot { background:var(--red); }
.event.review .dot { background:var(--gold); }
.event .phase { font-weight:700; }
.event .status { font:700 .72rem ui-monospace,SFMono-Regular,Menlo,monospace; }
.event.pass .status { color:var(--green); }
.event.fail .status { color:var(--red); }
.event.review .status { color:var(--gold); }
.event .detail { color:var(--muted); line-height:1.4; }
.verdict { padding:1.3rem; border:2px solid var(--line); background:#fff; margin:1rem 0; }
.verdict.continue { border-color:#67a77d; background:var(--green-bg); }
.verdict.review { border-color:#c09b50; background:var(--amber-bg); }
.verdict.block { border-color:var(--red); background:var(--red-bg); }
.verdict .label { font:700 .72rem ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted); }
.verdict h2 { margin:.25rem 0 .55rem; font-size:2rem; }
.verdict.continue h2 { color:var(--green); }
.verdict.review h2 { color:var(--gold); }
.verdict.block h2 { color:var(--red); }
.verdict p { margin:.25rem 0; line-height:1.5; }
.why-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.6rem; margin-top:.8rem; }
.why-grid div { border:1px solid rgba(102,117,128,.28); background:rgba(255,255,255,.72); padding:.7rem; }
.why-grid small { display:block; color:var(--muted); font-weight:700; text-transform:uppercase; }
.why-grid strong { display:block; margin-top:.25rem; overflow-wrap:anywhere; }
.state-compare { display:grid; grid-template-columns:1fr 1fr; gap:.65rem; margin:.8rem 0; }
.state-card { border:1px solid #b7dcc6; background:var(--green-bg); padding:.85rem; }
.state-card.stale { border-color:#e4aeb2; background:var(--red-bg); }
.state-card small { display:block; color:var(--muted); text-transform:uppercase; font-weight:700; }
.state-card strong { display:block; margin-top:.3rem; }
.pill { display:inline-block; padding:.2rem .5rem; font:700 .72rem ui-monospace,SFMono-Regular,Menlo,monospace; }
.pill.live { color:var(--green); background:var(--green-bg); border:1px solid #b7dcc6; }
.pill.fallback { color:var(--gold); background:var(--amber-bg); border:1px solid #ecd9ad; }
.pill + .pill { margin-left:.4rem; }
.stages { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:.5rem; }
.stage { border:1px solid #bcd3e4; background:var(--blue-bg); padding:.7rem .75rem; font-size:.82rem; line-height:1.4; }
.stage .num { font:700 .7rem ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--blue); }
.stage strong { display:block; margin:.2rem 0 .25rem; color:var(--ink); }
.stage span { color:var(--muted); }
.stage.last { border-color:var(--blue); }
.plugin-row { display:grid; grid-template-columns:1fr 2fr; gap:.5rem; margin-top:.5rem; }
.plugin-row div { padding:.65rem .75rem; font-size:.82rem; line-height:1.45; }
.plugin-row .plug { border:1px dashed #a897c4; background:var(--purple-bg); }
.plugin-row .note { border:1px solid var(--line); background:#fff; color:var(--muted); }
.tag { display:inline-block; margin-right:.35rem; padding:.05rem .35rem; font:700 .62rem ui-monospace,SFMono-Regular,Menlo,monospace; vertical-align:1px; }
.tag.fw { color:var(--blue); background:var(--blue-bg); border:1px solid #bcd3e4; }
.tag.plug { color:var(--purple); background:var(--purple-bg); border:1px dashed #a897c4; }
.example-frame { border-left:3px dashed #a897c4; padding-left:.9rem; }
@media(max-width:960px) { .stages { grid-template-columns:repeat(3,minmax(0,1fr)); } }
@media(max-width:760px) { .stages,.plugin-row { grid-template-columns:1fr; } }
@media(max-width:760px) { .why-grid,.state-compare { grid-template-columns:1fr; } .event { grid-template-columns:18px 1fr; } .event .status,.event .detail { grid-column:2; } }
</style>
""", unsafe_allow_html=True)


def esc(value: object) -> str:
    return html.escape(str(value))


def normalize_date(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    return datetime.fromisoformat(str(value))


st.markdown(
    '<div class="hero"><div class="kicker">GOVERNANCE FRAMEWORK · AML + HEALTHCARE REFERENCE PIPELINES</div>'
    '<h1>Runtime governance for any agentic pipeline.</h1>'
    '<p>This is a domain-agnostic runtime governance framework. It runs the same generic checks on any '
    'multi-agent pipeline: deterministic authorization, trace integrity, content monitoring, an optional '
    'correctness plug-in, and an Action Governance gate that makes the final decision, with attribution and audit. '
    'AML and healthcare are two swappable reference pipelines plugged in here to make it concrete.</p></div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- framework stages
st.markdown('<p class="section-label">HOW THE FRAMEWORK GOVERNS</p>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Six fixed governance stages</h2>', unsafe_allow_html=True)
FRAMEWORK_STAGES = [
    ("Ingestion & normalization", "A client's pipeline JSON is validated into a normalized PipelineTrace."),
    ("Deterministic authorization", "AGT + customer policy authorize every agent handoff."),
    ("Generic trace integrity", "Provenance, version, and conflict checks across the trace."),
    ("NeMo content checks", "Agent outputs checked with the customer's model and prompts."),
    ("Optional correctness", "A domain plug-in recomputes and attributes agent results."),
    ("Action Governance gate", "AGT combines every signal into continue, review, or block."),
]
stage_html = "".join(
    f'<div class="stage{" last" if i == len(FRAMEWORK_STAGES) else ""}"><div class="num">FRAMEWORK · {i}</div>'
    f'<strong>{esc(title)}</strong><span>{esc(detail)}</span></div>'
    for i, (title, detail) in enumerate(FRAMEWORK_STAGES, start=1)
)
st.markdown(
    f'<div class="stages">{stage_html}</div>'
    '<div class="plugin-row">'
    '<div class="plug"><span class="tag plug">DOMAIN PLUG-IN</span><b>Swappable domain pieces:</b> '
    'the client pipeline (sends client-pipeline JSON to stage 1) and its optional correctness component (fills stage 5).</div>'
    '<div class="note">Switch between the AML and healthcare companies below—or supply another domain; '
    'the governance stages stay the same.</div>'
    '</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- the client pipeline (static definition)
REFERENCE_PIPELINES = {
    "Healthcare": {
        "path": Path(__file__).parent / "client_pipelines" / "healthcare_pipeline.json",
        "label": "HEALTHCARE PLUG-IN",
        "description": "patient intake, allergy safety, medication, coverage, and care decision",
    },
    "AML": {
        "path": Path(__file__).parent / "client_pipelines" / "aml_pipeline.json",
        "label": "AML PLUG-IN",
        "description": "alert intake, screening, transactions, KYC, and disposition",
    },
}

st.markdown('<p class="section-label">REFERENCE PIPELINE</p>', unsafe_allow_html=True)
selected_reference = st.selectbox(
    "Choose the company pipeline to ingest in the backend",
    tuple(REFERENCE_PIPELINES),
    help=(
        "The selection changes the client-supplied agents, topology, permissions, policies, "
        "record codec, simulator, and correctness component. The governance framework does not change."
    ),
)
reference = REFERENCE_PIPELINES[selected_reference]
PIPELINE_PATH = reference["path"]
plugin_label = reference["label"]

st.markdown('<p class="section-label">THE CLIENT PIPELINE · STATIC DEFINITION</p>', unsafe_allow_html=True)
if st.button("Load client pipeline", type="primary"):
    st.session_state.loaded_pipeline = selected_reference
if st.session_state.get("loaded_pipeline") != selected_reference:
    st.info("Load the client pipeline to enable live cases.")
    st.stop()
try:
    # Re-read on every run so edits to the client's file take effect immediately.
    pipeline_definition = load_pipeline_definition(PIPELINE_PATH)
except ClientPipelineError as exc:
    st.session_state.pop("loaded_pipeline", None)
    st.error(f"Client pipeline definition rejected: {exc}")
    st.stop()
pipeline_system = pipeline_definition["system"]
pipeline_policy = pipeline_definition["policy"]
st.success("Client pipeline loaded")
st.markdown(
    '<p class="section-help">Provided once by the client: its agents, who hands off to whom, what each agent is '
    'permitted to do, and its own deterministic policies. It holds no case and no run. The permissions and '
    'policies enforced below come from this file.</p>',
    unsafe_allow_html=True,
)
st.dataframe([{
    "node": agent.get("node_id", ""),
    "agent": agent["id"],
    "role": agent.get("role", ""),
    "receives from": ", ".join(pipeline_system["topology"].get(agent["id"], [])) or "root",
    "declared permissions": ", ".join(pipeline_system["permissions"].get(agent["id"], [])),
} for agent in pipeline_system["agents"]], hide_index=True, width="stretch")
with st.expander("Client policies and raw definition", expanded=False):
    for key in ("graph", "action"):
        document = pipeline_policy[key]
        st.markdown(
            f"**{esc(document.get('name', key))}** (default: `{esc(document.get('default_action', '-'))}`)"
        )
        st.dataframe([{
            "rule": rule.get("name"),
            "stage": rule.get("stage"),
            "condition": rule.get("condition"),
            "action": rule.get("action"),
            "priority": rule.get("priority"),
        } for rule in document.get("rules", [])], hide_index=True, width="stretch")
    st.code(PIPELINE_PATH.read_text(encoding="utf-8"), language="json")

# --------------------------------------------------------------------------- live case (dynamic)
st.markdown('<p class="section-label">LIVE CASE · ENTERED NOW</p>', unsafe_allow_html=True)
st.markdown(
    f'<h2 class="section-title">Run a live {esc(selected_reference)} case through the client pipeline</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="section-help example-frame">On submit, the selected company agents ({esc(reference["description"])}) '
    'execute the case. Their run is combined with the backend pipeline definition and ingested into the same '
    'normalized trace. The governance framework itself does not change by domain.</p>',
    unsafe_allow_html=True,
)
# Which engine executes the client agents for this run: LangGraph, or the hand-rolled fallback.
run_engine = AML_RUN_ENGINE if selected_reference == "AML" else HEALTHCARE_RUN_ENGINE
run_engine_fallback = run_engine != "langgraph"
if run_engine_fallback:
    st.warning(
        f"Pipeline engine: {run_engine} (FALLBACK) - LangGraph is not installed, so the hand-rolled "
        f"reference pipeline runs the {selected_reference} agents instead of the LangGraph StateGraph."
    )
else:
    st.caption("Pipeline engine: langgraph (LangGraph StateGraph, deterministic agent nodes)")
if selected_reference == "AML":
    st.markdown(
        '<div class="watchlist"><b>Demo watchlist:</b> '
        + ' &middot; '.join(f'<code>{esc(name)}</code>' for name in sorted(SANCTIONS_WATCHLIST))
        + '<br>Use a watchlist name to see a governance conflict; any other name runs clean. '
        'A watchlist name creates a v2 hit while the final agent consumes stale v1 state.</div>',
        unsafe_allow_html=True,
    )
    empty_transactions = pd.DataFrame({
        "id": pd.Series(dtype="str"),
        "date": pd.Series(dtype="datetime64[ns]"),
        "type": pd.Series(dtype="str"),
        "amount": pd.Series(dtype="float"),
        "branch": pd.Series(dtype="str"),
    })
    with st.form("aml-case-input"):
        c1, c2 = st.columns(2)
        with c1:
            customer_name = st.text_input("Customer name", placeholder="Enter customer name")
            account_id = st.text_input("Account ID", placeholder="Enter account id")
        with c2:
            occupation = st.text_input("Occupation", placeholder="Enter occupation")
            expected_cash = st.number_input(
                "Expected monthly cash volume", min_value=0.0, value=0.0, step=500.0
            )
        monitoring_rule = st.text_input(
            "Monitoring rule", value="customer and transaction activity review"
        )
        st.caption("Transactions: add at least one row.")
        edited_transactions = st.data_editor(
            empty_transactions,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "id": st.column_config.TextColumn("Transaction ID", required=True),
                "date": st.column_config.DateColumn("Date", required=True),
                "type": st.column_config.SelectboxColumn(
                    "Type", options=["cash_deposit", "wire", "card_payment", "cash_withdrawal"], required=True
                ),
                "amount": st.column_config.NumberColumn("Amount", min_value=0.01, format="$%.2f", required=True),
                "branch": st.column_config.TextColumn("Branch", required=True),
            },
        )
        submitted = st.form_submit_button(
            "Run live case through the AML pipeline", type="primary", width="stretch"
        )
else:
    examples = " &middot; ".join(
        f'<code>{esc(medication)} → {esc(allergen)}</code>'
        for medication, allergen in MEDICATION_ALLERGENS.items()
    )
    st.markdown(
        '<div class="watchlist"><b>Healthcare safety examples:</b> ' + examples
        + '<br>Choose a medication with its matching allergy to see a governance conflict; any other '
        'combination runs clean. A match creates a current v2 contraindication that the final care agent '
        'receives as stale v1 context. Use synthetic data only.</div>',
        unsafe_allow_html=True,
    )
    with st.form("healthcare-case-input"):
        c1, c2 = st.columns(2)
        with c1:
            patient_name = st.text_input("Patient name", placeholder="Enter patient name")
            patient_id = st.text_input("Patient ID", value="PAT-DEMO-001")
            patient_age = st.number_input("Age", min_value=0, max_value=120, value=45)
            diagnosis = st.text_input("Diagnosis / care reason", value="bacterial infection")
        with c2:
            medication = st.selectbox(
                "Medication order", ["Acetaminophen", *MEDICATION_ALLERGENS.keys()],
                index=None, placeholder="Choose a medication",
            )
            known_allergy = st.selectbox(
                "Known allergy", ["none", "penicillin", "cephalosporin", "NSAID"],
                index=None, placeholder="Choose a known allergy (or none)",
            )
            coverage_authorized = st.checkbox("Coverage / prior authorization confirmed", value=True)
        submitted = st.form_submit_button(
            "Run live case through the healthcare pipeline", type="primary", width="stretch"
        )

if submitted:
    st.session_state.pop("governance_review", None)
    try:
        if selected_reference == "AML":
            if not customer_name.strip():
                raise ValueError("Enter a customer name")
            if not account_id.strip() or not occupation.strip():
                raise ValueError("Account ID and occupation are required")
            rows = [
                row for row in (
                    edited_transactions.to_dict("records")
                    if hasattr(edited_transactions, "to_dict") else list(edited_transactions)
                )
                if not all(pd.isna(value) or value == "" for value in row.values())
            ]
            if not rows:
                raise ValueError("Add at least one transaction")
            if any(pd.isna(value) or value == "" for row in rows for value in row.values()):
                raise ValueError("Complete every field of each transaction row")
            transactions = tuple(
                Transaction(
                    id=str(row["id"]), account_id=account_id.strip(), amount=float(row["amount"]),
                    type=str(row["type"]), timestamp=normalize_date(row["date"]), branch=str(row["branch"]),
                )
                for row in rows
            )
            customer = CustomerProfile(
                account_id.strip(), customer_name.strip(), occupation.strip(), float(expected_cash)
            )
            domain_records = Alert(
                "AML-INTERACTIVE", account_id.strip(), transactions, customer, monitoring_rule.strip()
            )
            stale_view = customer.name in SANCTIONS_WATCHLIST
            case_run, case_records = simulate_aml_case(domain_records, stale_screening_view=stale_view)
            codecs = AML_RECORD_CODECS
            correctness_component = AmlCorrectnessComponent()
            subject_label, subject_value = "Customer", customer.name
        else:
            if not patient_name.strip():
                raise ValueError("Enter a patient name")
            if medication is None or known_allergy is None:
                raise ValueError("Choose a medication order and a known allergy (or none)")
            if not patient_id.strip() or not diagnosis.strip():
                raise ValueError("Patient ID and diagnosis are required")
            domain_records = HealthcareCase(
                "HC-INTERACTIVE", patient_id.strip(), patient_name.strip(), int(patient_age),
                diagnosis.strip(), medication, known_allergy, bool(coverage_authorized)
            )
            stale_view = allergy_finding(domain_records)[1]
            case_run, case_records = simulate_healthcare_case(
                domain_records, stale_allergy_view=stale_view
            )
            codecs = HEALTHCARE_RECORD_CODECS
            correctness_component = HealthcareCorrectnessComponent()
            subject_label, subject_value = "Patient", domain_records.patient_name

        progress = st.progress(
            10, text=f"Client pipeline: {selected_reference} agents are running the live case"
        )
        payload_json = json.dumps(
            compose_client_payload(pipeline_definition, case_run, case_records), indent=2
        )
        progress.progress(30, text="Framework: ingesting the client-pipeline JSON")
        ingested = ingest_client_pipeline(payload_json, codecs)
        outcome = ingested.outcome
        policy_overrides = {
            key: value for key, value in (
                ("graph_policy", ingested.graph_policy), ("action_policy", ingested.action_policy)
            ) if value is not None
        }
        governor = GraphGovernor(
            system_description=ingested.description,
            correctness_component=correctness_component,
            **policy_overrides,
        )
        progress.progress(45, text="Framework: running authorization, trace, NeMo, and correctness checks")
        review = governor.review(outcome)
        progress.progress(85, text="Framework: Action Governance is combining signals")
        st.session_state.governance_review = {
            "domain": selected_reference,
            "plugin_label": plugin_label,
            "review": review,
            "integrity": check_graph_integrity(outcome),
            "agt_live": governor.agt_live,
            "payload_json": payload_json,
            "subject_label": subject_label,
            "subject_value": subject_value,
            "final_agent": ingested.description.final_agent,
            "run_engine": run_engine,
            "run_engine_fallback": run_engine_fallback,
        }
        progress.progress(100, text="Governance decision complete")
        progress.empty()
    except ClientPipelineError as exc:
        st.error(f"The client-pipeline payload was rejected at ingestion: {exc}")
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"The case could not be processed: {exc}")

result = st.session_state.get("governance_review")
if result is None or result["domain"] != selected_reference:
    st.info("Enter a live case above to run it through the loaded client pipeline.")
    st.stop()

review = result["review"]
integrity = result["integrity"]
steps = {step.agent: step for step in review.outcome.steps}
decision = steps[result["final_agent"]]
consumer = steps.get(integrity.origin_agent, decision)
conflict_parents = (
    *integrity.stale_parents,
    *integrity.altered_parents,
    *integrity.future_version_parents,
    *integrity.missing_parents,
)
conflict_parent = next((parent for parent in conflict_parents if parent in steps), None)
producer = steps.get(conflict_parent) if conflict_parent else None
consumed_finding = (
    consumer.received_findings_by_parent.get(conflict_parent, "missing")
    if conflict_parent else "not applicable"
)
consumed_version = (
    consumer.received_versions_by_parent.get(conflict_parent, "missing")
    if conflict_parent else "not applicable"
)
action = str(review.policy_decision.get("action", "deny"))
action_label = {"allow": "CONTINUE", "require_approval": "HUMAN REVIEW", "deny": "BLOCK"}.get(action, action.upper())
verdict_css = {"allow": "continue", "require_approval": "review", "deny": "block"}.get(action, "block")

# --------------------------------------------------------------------------- ingested client payload
st.markdown('<p class="section-label">GOVERNED PAYLOAD · DEFINITION + LIVE RUN</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="section-help">The framework governed the case for <b>{esc(result["subject_value"])}</b> flowing through '
    f'the client\'s pipeline. <code>system</code> and <code>policy</code> come from '
    f'the client pipeline definition; <code>run</code> and <code>records</code> come from this live '
    'case\'s simulated run. The governance below ran only on this ingested payload.</p>',
    unsafe_allow_html=True,
)
with st.expander(
    "Governed payload: client pipeline definition + this case's run", expanded=False
):
    st.code(result["payload_json"], language="json")

# --------------------------------------------------------------------------- event stream
st.markdown('<p class="section-label">LIVE GOVERNANCE RUN</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-help"><span class="tag fw">FRAMEWORK</span>generic stages, identical for every pipeline '
    f'&nbsp;&nbsp;<span class="tag plug">{esc(result["plugin_label"])}</span>the swappable domain pieces slotted into them.</p>',
    unsafe_allow_html=True,
)
FW, PLUG = "fw", "plug"
events: list[tuple[str, str, str, str, str]] = [
    (PLUG, f"Client agents run case ({result['domain']})", "RAN", "pass",
     f"Simulated {result['domain']} agents executed the live case through the client pipeline's topology "
     f"(engine: {result['run_engine']}{', FALLBACK' if result['run_engine_fallback'] else ''})."),
    (FW, "JSON ingestion", "PASS", "pass",
     f"Definition + live run validated into a PipelineTrace of {len(review.outcome.steps)} agent steps; "
     "permissions and policies taken from the client pipeline definition."),
]
for handoff in review.handoffs:
    events.append((
        FW, f"Authorization · {handoff.sender} → {handoff.recipient}",
        "AUTHORIZED" if handoff.passed else "DENIED",
        "pass" if handoff.passed else "fail",
        handoff.reason or handoff.rule or "Handoff policy evaluated.",
    ))
trace_status = "PASS" if integrity.passed else "STALE / CONFLICT"
events.append((
    FW, "Generic trace integrity", trace_status,
    "pass" if integrity.passed else "review", integrity.reason,
))
for check in review.content_checks:
    if not check["passed"]:
        status, css = "FLAG", "review"
    elif check["rail"] == "unavailable":
        status, css = "UNAVAILABLE", "review"
    else:
        status, css = "PASS", "pass"
    events.append((FW, f"NeMo content · {check['agent']}", status, css, str(check["reason"])))
events.append((
    PLUG, f"Correctness plug-in ({result['domain']})",
    "PASS" if review.attribution.passed else "FLAG",
    "pass" if review.attribution.passed else "review",
    review.attribution.reason,
))
events.append((
    FW, "Action Governance gate", action_label,
    "pass" if action == "allow" else ("review" if action == "require_approval" else "fail"),
    str(review.policy_decision.get("reason", "Final policy evaluated.")),
))

event_html = "".join(
    f'<div class="event {css}"><span class="dot"></span><span class="phase">'
    f'<span class="tag {layer}">{"FRAMEWORK" if layer == FW else esc(result["plugin_label"])}</span>{esc(phase)}</span>'
    f'<span class="status">{esc(status)}</span><span class="detail">{esc(detail)}</span></div>'
    for layer, phase, status, css, detail in events
)
engine_pill = (
    '<span class="pill live">AGT LIVE</span>' if result["agt_live"]
    else '<span class="pill fallback">AGT FALLBACK</span>'
) + (
    f'<span class="pill fallback">PIPELINE ENGINE: {esc(result["run_engine"]).upper()} (FALLBACK)</span>'
    if result["run_engine_fallback"]
    else f'<span class="pill live">PIPELINE ENGINE: {esc(result["run_engine"]).upper()}</span>'
)
st.markdown(
    f'<div class="simulator">{engine_pill}{event_html}</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- state comparison and verdict
st.markdown('<p class="section-label">WHERE & WHY</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-help">The framework makes the decision below from its generic stages; '
    'the AML case is only the input it governed.</p>',
    unsafe_allow_html=True,
)
if integrity.passed:
    st.success("Framework trace integrity: no conflict. The screening and decision agents used consistent state.")
else:
    if producer is not None:
        st.markdown(
            f'<div class="state-compare">'
            f'<div class="state-card"><small>{esc(producer.agent)} emitted</small>'
            f'<strong>{esc(producer.finding)} · version {producer.finding_version}</strong></div>'
            f'<div class="state-card stale"><small>{esc(consumer.agent)} consumed</small>'
            f'<strong>{esc(consumed_finding)} · version {esc(consumed_version)}</strong></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

origin = (
    f"{review.attribution.origin_node or '-'} · {review.attribution.origin_agent}"
    if review.attribution.origin_agent else "No failing agent attributed"
)
st.markdown(
    f'<div class="verdict {verdict_css}"><div class="label">FRAMEWORK GOVERNANCE DECISION · ACTION GOVERNANCE GATE</div>'
    f'<h2>{esc(action_label)}</h2>'
    f'<p><b>Why:</b> {esc(review.attribution.reason)}</p>'
    f'<p><b>Policy result:</b> {esc(str(review.policy_decision.get("reason", "")))}</p>'
    f'<div class="why-grid">'
    f'<div><small>{esc(result["subject_label"])}</small><strong>{esc(result["subject_value"])}</strong></div>'
    f'<div><small>Agent disposition</small><strong>{esc(decision.disposition)}</strong></div>'
    f'<div><small>Attributed to</small><strong>{esc(origin)}</strong></div>'
    f'<div><small>Category</small><strong>{esc(review.attribution.category or "none")}</strong></div>'
    f'</div></div>',
    unsafe_allow_html=True,
)

with st.expander("Audit and proof", expanded=False):
    st.markdown("#### Normalized PipelineTrace")
    st.dataframe([{
        "node": step.node_id,
        "agent": step.agent,
        "action": step.action_type,
        "declared permissions": ", ".join(sorted(step.allowed_actions)),
        "parents": ", ".join(step.received_from) or "root",
        "finding": step.finding,
        "emitted version": step.finding_version,
        "received findings": json.dumps(step.received_findings_by_parent),
        "received versions": json.dumps(step.received_versions_by_parent),
        "disposition": step.disposition or "",
    } for step in review.outcome.steps], hide_index=True, width="stretch")
    st.markdown("#### Branch attribution")
    st.dataframe([{
        "node": check.node_id,
        "agent": check.agent,
        "reported": check.reported,
        "expected": check.recomputed,
        "matches": check.matches,
        "detail": check.detail,
    } for check in review.attribution.branch_checks], hide_index=True, width="stretch")
    st.markdown("#### Resolver AuthorityDecision")
    st.json(review.authority_decision)
    st.markdown("#### Final AGT PolicyDecision")
    st.json(review.policy_decision)
    st.markdown("#### Audit evidence")
    st.dataframe(list(review.audit), hide_index=True, width="stretch")
