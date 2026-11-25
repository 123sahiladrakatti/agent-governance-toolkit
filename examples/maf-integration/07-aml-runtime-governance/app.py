
"""Runtime governance demo with swappable AML and healthcare client pipelines.

Run: streamlit run app.py
"""

from __future__ import annotations

import html
import json
import os
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st

from aml.client_payload import AML_RECORD_CODECS, AML_RUN_ENGINE, simulate_aml_case, simulate_aml_case_llm
from aml.llm_decision_agent import default_config as aml_llm_config
from healthcare.llm_decision_agent import default_config as healthcare_llm_config
from healthcare.llm_allergy_agent import default_config as allergy_llm_config
from healthcare.llm_allergy_agent import redact_note
from healthcare.batch import (
    BATCH_LIMIT,
    BATCH_MODES,
    DEFAULT_BATCH_FILE,
    BatchFileError,
    load_healthcare_cases,
    run_healthcare_batch,
    summary_line,
)
from llm_decision import LLMDecisionError
from aml.correctness import AmlCorrectnessComponent
from client_pipeline import ClientPipelineError, load_pipeline_definition
from domain import (
    Alert,
    CustomerProfile,
    SANCTIONS_WATCHLIST,
    Transaction,
)
from healthcare.client_payload import (
    HEALTHCARE_RUN_ENGINE,
    HEALTHCARE_RECORD_CODECS,
    allergy_finding,
    simulate_healthcare_case,
    simulate_healthcare_case_llm,
    simulate_healthcare_case_llm_pii,
)
from healthcare.correctness import HealthcareCorrectnessComponent
from healthcare.system import DEMO_MEMBER_ID, HealthcareCase, MEDICATION_ALLERGENS
from env_file import load_env_file
from governed_run import govern_client_run
from governance_view import CONTRAST_ROWS, FW, build_governance_view
import gaco
from instrumentation import case_json_for, load_harness
from longrun_demo import report_summary, run_degradation_demo, run_demo
from report_pdf import build_case_report

# Local settings (e.g. AML_LLM_NUM_CTX) from .env, re-read on every run so edits apply without a restart.
ENV_FILE = Path(__file__).parent / ".env"
env_from_file = load_env_file(ENV_FILE)


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
.pill.llm { color:var(--purple); background:var(--purple-bg); border:1px solid #cdbfe3; }
.loss { border:1px solid #cdbfe3; background:var(--purple-bg); padding:1rem 1.1rem; margin:.8rem 0; }
.loss h3 { margin:.1rem 0 .4rem; font-size:1.05rem; }
.loss p { margin:.3rem 0; line-height:1.5; font-size:.9rem; }
.loss table { width:100%; border-collapse:collapse; margin:.6rem 0; font-size:.85rem; }
.loss th, .loss td { text-align:left; padding:.4rem .5rem; border-bottom:1px solid #ddd3ec; }
.loss th { color:var(--muted); font-size:.72rem; text-transform:uppercase; }
.loss .lost { color:var(--red); font-weight:700; }
.loss .kept { color:var(--green); font-weight:700; }
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
DECISION_MODES = {
    "Deterministic": "deterministic",
    "Real LLM (context-loss demo)": "llm-context-loss",
}
# Per-domain wording for the real-LLM decision agent; the machinery is shared (llm_decision.py).
LLM_DEMO = {
    "AML": {
        "config": aml_llm_config,
        "prefix": "AML_LLM",
        "agent": "decision agent",
        "file": "case file",
        "planted": "a watchlist name plants a stale screening view",
        "order": "the screening report comes first, then the case activity log, then the transaction and KYC reports",
        "lost": "screening report",
    },
    "Healthcare": {
        "config": healthcare_llm_config,
        "prefix": "HEALTHCARE_LLM",
        "agent": "care-decision agent",
        "file": "care file",
        "planted": "a medication with its matching allergy plants a stale allergy view",
        "order": "the allergy-safety report comes first, then the care coordination log, then the medication "
                 "and coverage reports",
        "lost": "allergy-safety report",
    },
}
PII_MODE_LABEL = "Real LLM allergy agent (PII-leak demo)"
demo = LLM_DEMO[selected_reference]
llm_config = demo["config"]()
mode_options = {
    **DECISION_MODES,
    **({PII_MODE_LABEL: "llm-pii-leak"} if selected_reference == "Healthcare" else {}),
}
decision_mode = mode_options[st.radio(
    "Agent mode",
    tuple(mode_options),
    horizontal=True,
    help=(
        f"Deterministic: every agent is scripted; {demo['planted']}. "
        f"Real LLM (context-loss demo): only the {demo['agent']} is a real Ollama LLM with a deliberately small "
        "context window; the other agents stay deterministic and nothing is planted."
        + (" Real LLM allergy agent (PII-leak demo): only the allergy-safety agent is a real Ollama LLM, given the "
           "full patient record and a permissive prompt; NeMo checks its handoff note for personal data."
           if selected_reference == "Healthcare" else "")
    ),
)]
redact_control = False
allergy_config = allergy_llm_config()
if decision_mode == "llm-pii-leak":
    st.info(
        f"Allergy-safety agent: real LLM (PII-leak demo). {allergy_config.model} via local Ollama; every other "
        "agent stays deterministic. The agent receives the full patient record and is asked only to check the "
        "medication order and write a short, self-contained handoff note for the receiving nurse. The prompt never "
        "asks for identifiers and has no redaction instruction, so any patient name or ID in the note is the "
        "model's own over-sharing. The NeMo content rail checks that note for personal data. "
        "Each run can take 10-30 seconds."
    )
    redact_control = st.checkbox(
        "Control: add a redaction instruction to the allergy agent's prompt",
        help=("Same prompt plus one sentence telling the agent not to include the patient's name, ID, age, or "
              "diagnosis. With it the agent should not leak, and NeMo should pass the note."),
    )
    allergy_keys = ("HEALTHCARE_ALLERGY_LLM_MODEL", "OLLAMA_URL")
    allergy_sources = [
        f"{key}={os.environ[key]} ({'.env' if key in env_from_file else 'terminal'})"
        for key in allergy_keys if key in os.environ
    ]
    st.caption(
        "LLM settings: " + (", ".join(allergy_sources) if allergy_sources else "built-in defaults")
        + f" (model {allergy_config.model}). Edit {ENV_FILE.name} and run the case again to change them."
    )
elif decision_mode == "llm-context-loss":
    st.info(
        f"{demo['agent'].capitalize()}: real LLM (context-loss demo). {llm_config.model} via local Ollama with "
        f"context window num_ctx={llm_config.num_ctx} (the demo uses 1024, which the {demo['file']} overflows); "
        "the other agents stay deterministic. "
        f"The {demo['file']} reaches the {demo['agent']} in arrival order: {demo['order']}. When the "
        f"{demo['file']} is longer than the window, Ollama keeps only its most recent part and the "
        f"{demo['lost']} genuinely falls out. "
        "Each run makes several local Ollama calls and can take 15-60 seconds."
    )
    llm_keys = (f"{demo['prefix']}_MODEL", f"{demo['prefix']}_NUM_CTX", "OLLAMA_URL")
    sources = [
        f"{key}={os.environ[key]} ({'.env' if key in env_from_file else 'terminal'})"
        for key in llm_keys if key in os.environ
    ]
    st.caption(
        "LLM settings: " + (", ".join(sources) if sources else "built-in defaults")
        + f" (model {llm_config.model}, num_ctx={llm_config.num_ctx}). "
        f"Edit {ENV_FILE.name} and run the case again to change them."
    )
else:
    st.caption("Decision agent: deterministic")
if selected_reference == "AML":
    watchlist_hint = (
        "Use a watchlist name to see a governance conflict; any other name runs clean. "
        "A watchlist name creates a v2 hit while the final agent consumes stale v1 state."
        if decision_mode == "deterministic" else
        "In this mode the screening report falls out of the decision agent's window for every case. "
        "With a watchlist name the agent's decision misses a real sanctions hit; with any other name its "
        "decision happens to be right, but it was still made without the screening result."
    )
    st.markdown(
        '<div class="watchlist"><b>Demo watchlist:</b> '
        + ' &middot; '.join(f'<code>{esc(name)}</code>' for name in sorted(SANCTIONS_WATCHLIST))
        + f'<br>{esc(watchlist_hint)}</div>',
        unsafe_allow_html=True,
    )
    default_transactions = pd.DataFrame({
        "id": pd.Series(["T-001", "T-002"], dtype="str"),
        "date": pd.Series(pd.to_datetime(["2026-01-05", "2026-01-06"]), dtype="datetime64[ns]"),
        "type": pd.Series(["cash_deposit", "cash_deposit"], dtype="str"),
        "amount": pd.Series([2_500.0, 1_500.0], dtype="float"),
        "branch": pd.Series(["BR-01", "BR-01"], dtype="str"),
    })
    with st.form("aml-case-input"):
        c1, c2 = st.columns(2)
        with c1:
            customer_name = st.text_input("Customer name", placeholder="Enter customer name")
            account_id = st.text_input("Account ID", value="ACCT-DEMO-001")
        with c2:
            occupation = st.text_input("Occupation", value="business owner")
            expected_cash = st.number_input(
                "Expected monthly cash volume", min_value=0.0, value=8_000.0, step=500.0
            )
        monitoring_rule = st.text_input(
            "Monitoring rule", value="customer and transaction activity review"
        )
        st.caption("Transactions: at least one row is required.")
        edited_transactions = st.data_editor(
            default_transactions,
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
    safety_hint = (
        "Choose a medication with its matching allergy to see a governance conflict; any other "
        "combination runs clean. A match creates a current v2 contraindication that the final care agent "
        "receives as stale v1 context."
        if decision_mode == "deterministic" else
        "In this mode any medication and allergy combination works: the failure is in the allergy agent's note, "
        "not the decision. For Cephalexin with a penicillin allergy the model tends to report CONTRAINDICATION "
        "(clinical cross-reactivity) although the formulary rule says NO-CONTRAINDICATION, so the deterministic "
        "correctness check also flags that combination."
        if decision_mode == "llm-pii-leak" else
        "In this mode the allergy-safety report falls out of the care-decision agent's window for every case. "
        "With a medication and its matching allergy the agent's decision misses a real contraindication; "
        "otherwise its decision happens to be right, but it was still made without the allergy result."
    )
    st.markdown(
        '<div class="watchlist"><b>Healthcare safety examples:</b> ' + examples
        + f'<br>{esc(safety_hint)} Use synthetic data only.</div>',
        unsafe_allow_html=True,
    )
    with st.form("healthcare-case-input"):
        c1, c2 = st.columns(2)
        with c1:
            patient_name = st.text_input("Patient name", placeholder="Enter patient name")
            patient_id = st.text_input("Patient ID", value="PAT-DEMO-001")
            member_id = st.text_input(
                "Insurance member ID (synthetic)", value=DEMO_MEMBER_ID,
                help="Synthetic, obviously fake member number (MBR-####-####-##); never a real ID or SSN. "
                     "Optional: leave blank for none.",
            )
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
    llm_allergy = None
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
            llm_decision = None
            if decision_mode == "llm-context-loss":
                with st.spinner(
                    f"Decision agent: real LLM ({llm_config.model}, num_ctx={llm_config.num_ctx}) is deciding the case"
                ):
                    case_run, case_records, llm_decision = simulate_aml_case_llm(domain_records, llm_config)
            else:
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
                diagnosis.strip(), medication, known_allergy, bool(coverage_authorized),
                member_id=member_id.strip(),
            )
            llm_decision = None
            if decision_mode == "llm-pii-leak":
                with st.spinner(f"Allergy-safety agent: real LLM ({allergy_config.model}) is writing its handoff note"):
                    case_run, case_records, llm_allergy = simulate_healthcare_case_llm_pii(
                        domain_records, allergy_config, redact_control
                    )
            elif decision_mode == "llm-context-loss":
                with st.spinner(
                    f"Care-decision agent: real LLM ({llm_config.model}, num_ctx={llm_config.num_ctx}) is deciding the case"
                ):
                    case_run, case_records, llm_decision = simulate_healthcare_case_llm(domain_records, llm_config)
            else:
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
        governed = govern_client_run(
            pipeline_definition, case_run, case_records, codecs, correctness_component,
            on_stage=lambda percent, message: progress.progress(percent, text=message),
        )
        st.session_state.governance_review = {
            "domain": selected_reference,
            "plugin_label": plugin_label,
            "review": governed.review,
            "integrity": governed.integrity,
            "agt_live": governed.agt_live,
            "subject_label": subject_label,
            "subject_value": subject_value,
            "final_agent": governed.final_agent,
            "run_engine": run_engine,
            "run_engine_fallback": run_engine_fallback,
            "decision_mode": decision_mode,
            "llm_decision": llm_decision,
            "llm_allergy": llm_allergy,
            "records": domain_records,
        }
        progress.progress(100, text="Governance decision complete")
        progress.empty()
    except LLMDecisionError as exc:
        st.error(f"The real-LLM agent failed, so nothing was governed: {exc}")
        if exc.raw_output:
            with st.expander("Raw decision-agent output", expanded=True):
                st.code(exc.raw_output)
    except ClientPipelineError as exc:
        st.error(f"The client-pipeline payload was rejected at ingestion: {exc}")
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"The case could not be processed: {exc}")

# --------------------------------------------------------------------------- backend batch (healthcare)
if selected_reference == "Healthcare":
    batch_state = st.session_state.get("healthcare_batch")
    with st.expander("BATCH · Govern the sample healthcare cases from the backend Excel file", expanded=bool(batch_state)):
        st.markdown(
            f'<p class="section-help">Reads <code>batch_cases/{esc(DEFAULT_BATCH_FILE.name)}</code> from the repo '
            "(no upload) and runs every case through one real-LLM failure mode, using the same ingestion and "
            f"governance path as the single-case form. Up to {BATCH_LIMIT} cases; each real-LLM case takes 15-60 "
            "seconds.</p>",
            unsafe_allow_html=True,
        )
        batch_mode = {label: key for key, label in BATCH_MODES.items()}[st.radio(
            "Batch LLM failure mode", tuple(BATCH_MODES.values()), horizontal=True, key="batch_mode",
        )]
        batch_decision_config, batch_allergy_config = healthcare_llm_config(), allergy_llm_config()
        st.caption(
            f"Care-decision agent: {batch_decision_config.model}, num_ctx={batch_decision_config.num_ctx}"
            if batch_mode == "llm-context-loss" else
            f"Allergy-safety agent: {batch_allergy_config.model}, permissive prompt (no redaction instruction)"
        )
        if st.button("Load & govern sample cases", key="run_batch"):
            st.session_state.pop("healthcare_batch", None)
            try:
                batch_cases = load_healthcare_cases(DEFAULT_BATCH_FILE)
            except BatchFileError as exc:
                st.error(f"The batch file was rejected: {exc}")
            else:
                if len(batch_cases) > BATCH_LIMIT:
                    st.warning(
                        f"{len(batch_cases)} cases found; only the first {BATCH_LIMIT} run "
                        "(real-LLM runs take 15-60 seconds each)."
                    )
                batch_progress = st.progress(0, text="Starting the batch")

                def report(position, total, item):
                    batch_progress.progress(
                        (position - 1) / total,
                        text=f"Running case {position} of {total}: {item.case.patient_name} (row {item.row})",
                    )

                batch_results = run_healthcare_batch(
                    batch_cases, batch_mode, pipeline_definition,
                    decision_config=batch_decision_config, allergy_config=batch_allergy_config, on_case=report,
                )
                batch_progress.empty()
                st.session_state.healthcare_batch = batch_state = {
                    "mode": batch_mode, "results": batch_results,
                    "skipped": max(0, len(batch_cases) - BATCH_LIMIT),
                }
        if batch_state:
            batch_results = batch_state["results"]
            st.markdown(
                f"**{esc(BATCH_MODES[batch_state['mode']])}:** {esc(summary_line(batch_results))}"
                + (f" ({batch_state['skipped']} case(s) beyond the limit were not run)" if batch_state["skipped"] else "")
            )
            st.dataframe([{
                "row": r.row,
                "patient": r.patient_name,
                "medication / allergy": f"{r.medication} / {r.known_allergy}",
                "agent disposition": r.disposition or "-",
                "trace check": r.trace_check or "-",
                "NeMo": r.nemo or "-",
                "verdict": r.verdict or "ERROR",
                "caught by": r.error or r.caught_by,
            } for r in batch_results], hide_index=True, width="stretch")
            st.caption(
                "These are real LLM runs, so results are probabilistic: the same case can come out differently on "
                "another run, model, or context window. The governance checks themselves are deterministic, apart "
                "from NeMo, which is also an LLM judgement."
            )
            for r in batch_results:
                with st.expander(f"Row {r.row} · {r.patient_name} · {r.verdict or 'ERROR'}"):
                    if r.error:
                        st.error(r.error)
                        continue
                    review_b = r.governed.review
                    st.markdown(f"**Policy result:** {esc(str(review_b.policy_decision.get('reason', '')))}")
                    st.markdown(
                        f"**Records require:** {esc(review_b.attribution.expected_disposition)} · "
                        f"**Agent decided:** {esc(r.disposition or '-')}"
                    )
                    if batch_state["mode"] == "llm-context-loss":
                        st.markdown(
                            f"**Decision agent:** {esc(r.llm.model)}, num_ctx={r.llm.num_ctx}, kept "
                            f"{r.llm.kept_tokens} prompt tokens; not in its context: "
                            f"{esc(', '.join(r.llm.not_consumed) or 'none')}"
                        )
                        st.markdown("**Trace check:** " + esc(r.governed.integrity.reason))
                    else:
                        st.markdown(
                            f"**Allergy agent note** (identifiers masked): "
                            f"<code>{esc(redact_note(r.case, r.llm.note))}</code>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**Identifiers found verbatim:** {esc(', '.join(r.llm.identifiers_in_note) or 'none')}"
                        )
                    nemo_rows = [c for c in review_b.content_checks if c.get("rail") != "skipped"]
                    st.markdown("**NeMo:** " + esc("; ".join(
                        f"{c['agent']} {'FLAG' if not c['passed'] else c['rail']}" for c in nemo_rows
                    )))
                    # The PDF is built from this case's stored governance result (no re-run), once per batch.
                    pdfs = batch_state.setdefault("pdfs", {})
                    if r.row not in pdfs:
                        try:
                            pdfs[r.row] = build_case_report(
                                r.case, batch_state["mode"], r.governed, r.llm,
                                run_engine=HEALTHCARE_RUN_ENGINE,
                                run_engine_fallback=HEALTHCARE_RUN_ENGINE != "langgraph", row=r.row,
                            )
                        except Exception as exc:  # a report failure must not hide the batch results
                            pdfs[r.row] = exc
                    if isinstance(pdfs[r.row], Exception):
                        st.warning(f"The PDF report could not be built: {pdfs[r.row]}")
                    else:
                        st.download_button(
                            "Download PDF report", data=pdfs[r.row], mime="application/pdf",
                            file_name=f"governance-report-{r.case.case_id}.pdf", key=f"batch_pdf_{r.row}",
                            help="Full governance detail for this case (identifiers masked).",
                        )

# --------------------------------------------------------------------------- long-running demo (healthcare)
LONGRUN_CASE_FILE = Path(__file__).parent / "harness_cases" / "healthcare_contraindication.json"
LONGRUN_MODES = {
    "overflow": "Overflow (reliable contrast)",
    "degradation": "Degradation (lost-in-the-middle; may be robust)",
}
LONGRUN_PLACEMENTS = {
    "recent_end": "near end (after the history)",
    "buried_mid": "buried mid-history",
    "overflowed": "overflowed (history after it)",
}


def longrun_span(case: dict) -> str:
    months = case["history_span_days"] / 30.4
    return f"~{months:.0f} months" if months >= 1.5 else f"~{case['history_span_days']} days"


def longrun_yes_no(value: object) -> str:
    return "yes" if value is True else "no" if value is False else "unknown"


def longrun_rows(case: dict, *, kind: str, is_case2: bool, burial_insufficient: bool) -> list[tuple[str, str]]:
    """The per-case rows, worded as in longrun_demo.format_report / format_degradation_report."""
    occupied = (f"{case['occupied_tokens']} tokens ({case['history_tokens']} of history, "
                f"~{case['history_runs']} prior runs over {case['history_span_days']} days)")
    if kind == "degradation":
        if case.get("truncated") is False and case["occupied_tokens"] == case["processed_tokens"]:
            inside = f"yes, NOT truncated ({case['processed_tokens']} of {case['occupied_tokens']} tokens processed)"
        elif case.get("truncated"):
            inside = f"no, truncated ({case['processed_tokens']} of {case['occupied_tokens']} processed)"
        else:
            inside = "unknown"
        placement = (LONGRUN_PLACEMENTS["recent_end"] if case["placement"] == "recent_end" else
                     f"buried deep: {case['depth']:.0%} of the history is newer than it "
                     f"(~{case['tokens_after_finding']} tokens on top)")
    else:
        if case["finding_inside_window"] is True:
            inside = "yes" if case["placement"] != "buried_mid" else "yes, but buried mid-context"
        elif case["finding_inside_window"] is False:
            inside = "no (pushed past the kept window)"
        else:
            inside = "unknown"
        placement = LONGRUN_PLACEMENTS[case["placement"]]
        if is_case2 and burial_insufficient:
            placement = f"buried mid-history -> {placement}; burial alone was insufficient"
    if case["passed"]:
        check = "PASS"
    else:
        check = "CATCHES LOSS" if is_case2 else f"FLAGGED ({case['verdict']['category']})"
    mechanism = case["mechanism"] or "-"
    if kind == "degradation" and is_case2 and case["mechanism"]:
        mechanism = f"{case['mechanism']} (detected)"
    return [
        ("occupied context", occupied),
        ("window", str(case["window"])),
        ("finding placement" if kind == "overflow" else "finding position", placement),
        ("finding inside window?", inside),
        ("agent used finding?", longrun_yes_no(case["agent_used_finding"])),
        ("decision", str(case["decision"])),
        ("governance check", check),
        ("failure mechanism", mechanism),
    ]


def longrun_card(title: str, case: dict, rows: list[tuple[str, str]]) -> str:
    body = "".join(f'<div style="margin-top:.6rem"><small>{esc(k)}</small><strong>{esc(v)}</strong></div>'
                   for k, v in rows)
    return (f'<div class="state-card{"" if case["passed"] else " stale"}">'
            f'<strong style="margin:0;font-size:1.02rem">{esc(title)}</strong>{body}</div>')


def longrun_case2_pill(report: dict) -> str:
    case2, outcome = report["case2"], report.get("outcome")
    if case2 is not None and not case2["passed"] and case2["mechanism"] in ("overflow", "degradation"):
        return f'<span class="pill llm">CASE 2 · CHECK CAUGHT LOSS · {esc(case2["mechanism"].upper())}</span>'
    if outcome == "robust":
        return (f'<span class="pill fallback">CASE 2 · NO LOSS OBSERVED · ROBUST UP TO '
                f'{esc(report["max_history_tested"])} TOKENS</span>')
    if outcome == "window-ceiling":
        return '<span class="pill fallback">CASE 2 · WINDOW CEILING REACHED · NOT TESTED FURTHER</span>'
    if case2 is not None and not case2["passed"]:
        return (f'<span class="pill fallback">CASE 2 · CHECK FLAGGED · '
                f'{esc((case2["mechanism"] or "indeterminate").upper())} (NOT A CONTEXT-LOSS RESULT)</span>')
    return '<span class="pill fallback">CASE 2 · NO LOSS OBSERVED</span>'


def longrun_run_rows(runs: list[dict]) -> list[dict]:
    return [{
        "placement": run["placement"],
        "window": run["window"],
        "history tokens": run["history_tokens"],
        "occupied": run["occupied_tokens"],
        "processed": run["processed_tokens"],
        "truncated": run["truncated"],
        "finding used": longrun_yes_no(run["agent_used_finding"]),
        "check": "PASS" if run["passed"] else "FAIL",
        "mechanism": run["mechanism"] or "-",
        **({"valid": "INVALID (truncated)" if run.get("invalid") else "yes"} if "inside_window_proven" in run else {}),
    } for run in runs]


if selected_reference == "Healthcare":
    st.markdown('<p class="section-label">LONG-RUNNING · EARLY VS. AGED CONTEXT</p>', unsafe_allow_html=True)
    st.markdown(
        '<h2 class="section-title">The same case after simulated months of accumulated history</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-help">Same system, same data, same window: only the amount of accumulated history in '
        "the care-decision agent's context changes. Early runs are clean; after simulated months the required "
        "finding is lost and the governance check catches it. Case 2 is an <b>accelerated simulation</b>: the "
        "history is synthesized prior-run notes (checked never to contain the finding), not months of real "
        f"operation. Sample case: <code>harness_cases/{esc(LONGRUN_CASE_FILE.name)}</code>.</p>",
        unsafe_allow_html=True,
    )
    try:
        longrun_case = case_json_for(
            str(PIPELINE_PATH), json.loads(LONGRUN_CASE_FILE.read_text(encoding="utf-8")),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        longrun_case = None
        st.error(f"The long-running demo case could not be loaded: {exc}")
    if longrun_case is not None:
        longrun_mode = {label: key for key, label in LONGRUN_MODES.items()}[st.radio(
            "Long-running demo mode", tuple(LONGRUN_MODES.values()), horizontal=True, key="longrun_mode",
        )]
        # Defaults come from harness.longrun_demo in the client pipeline JSON (nothing hardcoded here). The
        # harness itself is loaded on click: loading it contacts Ollama to verify the tokenizer and template.
        longrun_cfg = dict(longrun_case["harness"].get("longrun_demo", {}))
        if longrun_mode == "degradation":
            longrun_cfg = {**longrun_cfg, **longrun_cfg.get("degradation", {})}
        cfg_path = "harness.longrun_demo" + (".degradation" if longrun_mode == "degradation" else "")
        needed = {"window": "window", "early": "early_tokens", "aged": "aged_tokens"}
        if longrun_mode == "degradation":
            needed["depth"] = "depth"
        missing_cfg = [key for key in needed.values() if longrun_cfg.get(key) is None]
        if missing_cfg:
            st.warning(f"Set {', '.join(missing_cfg)} in {cfg_path} of the client pipeline JSON to run this mode.")
        else:
            st.caption(
                f"Care-decision agent: {longrun_case['harness']['model']['name']} via Ollama. Defaults from "
                f"{cfg_path} in client_pipelines/{PIPELINE_PATH.name}; each run makes real LLM calls (15-60s+ each"
                + ("; the degradation sweep runs several of them and can take minutes" if longrun_mode == "degradation"
                   else "; Case 2 may run twice") + ")."
            )
            input_cols = st.columns(4 if longrun_mode == "degradation" else 3)
            longrun_inputs = {
                name: input_cols[i].number_input(
                    {"window": "Window (tokens)", "early": "Case 1 history (tokens)",
                     "aged": "Case 2 history (tokens)"}[name],
                    min_value=1, value=int(longrun_cfg[key]), step=1, key=f"longrun_{longrun_mode}_{name}",
                )
                for i, (name, key) in enumerate((n, k) for n, k in needed.items() if n != "depth")
            }
            if longrun_mode == "degradation":
                longrun_inputs["depth"] = input_cols[3].number_input(
                    "Depth (share of history newer than the finding)", min_value=0.05, max_value=0.95,
                    value=float(longrun_cfg["depth"]), step=0.05, key="longrun_degradation_depth",
                )
            longrun_reports = st.session_state.setdefault("longrun_reports", {})
            run_label = "Run overflow demo" if longrun_mode == "overflow" else "Run degradation demo"
            if st.button(run_label, key=f"longrun_run_{longrun_mode}"):
                longrun_reports.pop(longrun_mode, None)
                with st.spinner(f"{run_label}: real Ollama calls in progress"):
                    try:
                        longrun_harness = load_harness(longrun_case)
                        if longrun_mode == "overflow":
                            longrun_report = run_demo(
                                longrun_case, int(longrun_inputs["window"]), int(longrun_inputs["early"]),
                                int(longrun_inputs["aged"]), harness=longrun_harness,
                            )
                        else:
                            longrun_report = run_degradation_demo(
                                longrun_case, int(longrun_inputs["window"]), int(longrun_inputs["early"]),
                                int(longrun_inputs["aged"]), float(longrun_inputs["depth"]), harness=longrun_harness,
                            )
                    except (RuntimeError, ValueError, OSError, KeyError) as exc:
                        longrun_reports[longrun_mode] = {"error": str(exc)}
                    else:
                        longrun_reports[longrun_mode] = {"report": longrun_report}
            stored = longrun_reports.get(longrun_mode)
            if stored and stored.get("error"):
                st.error(f"The long-running demo could not run: {stored['error']}")
            elif stored:
                longrun_report = stored["report"]
                kind = longrun_report.get("kind", "overflow")
                burial = bool(longrun_report.get("burial_alone_insufficient"))
                case1, case2 = longrun_report["case1"], longrun_report["case2"]
                left, right = st.columns(2)
                left.markdown(longrun_card(
                    "Case 1 (early)", case1,
                    longrun_rows(case1, kind=kind, is_case2=False, burial_insufficient=burial),
                ), unsafe_allow_html=True)
                if case2 is None:
                    right.warning("No Case 2 run fit inside the configured ceiling window, so nothing was tested.")
                else:
                    suffix = "buried deep" if kind == "degradation" else "simulated"
                    right.markdown(longrun_card(
                        f"Case 2 ({longrun_span(case2)} of history, {suffix})", case2,
                        longrun_rows(case2, kind=kind, is_case2=True, burial_insufficient=burial),
                    ), unsafe_allow_html=True)
                headline, notes = report_summary(longrun_report)
                case1_pill = ('<span class="pill live">CASE 1 · CHECK PASSED</span>' if case1["passed"] else
                              f'<span class="pill fallback">CASE 1 · CHECK FLAGGED · '
                              f'{esc(case1["verdict"]["category"])}</span>')
                st.markdown(
                    f'<div class="loss"><h3>Outcome</h3><p>{esc(headline)}</p>'
                    f'<p>{case1_pill}{longrun_case2_pill(longrun_report)}'
                    '<span class="pill">ACCELERATED SIMULATION</span></p>'
                    f'<ul>{"".join(f"<li>{esc(n)}</li>" for n in notes)}</ul></div>',
                    unsafe_allow_html=True,
                )
                if kind == "degradation":
                    st.markdown("**Escalation sweep** (each row is a real run; only non-truncated runs count as "
                                "degradation evidence)")
                    st.dataframe(longrun_run_rows(longrun_report["sweep"]), hide_index=True, width="stretch")
                elif burial:
                    st.markdown("**Case 2 attempts** (the buried run kept the finding, so the history was grown "
                                "past the window)")
                    st.dataframe(longrun_run_rows(longrun_report["case2_attempts"]), hide_index=True, width="stretch")
                st.download_button(
                    "Download full report (JSON, with traces)",
                    data=json.dumps(longrun_report, indent=2, default=str), mime="application/json",
                    file_name=f"longrun-{kind}-report.json", key=f"longrun_json_{longrun_mode}",
                )

# --------------------------------------------------------------------------- GACO window suggestion (healthcare)
GACO_CASE_FILE = Path(__file__).parent / "harness_cases" / "healthcare_contraindication.json"
GACO_SAVED_REPORT = Path(__file__).parent / "gaco_report.json"
GACO_SAVED_COMMAND = "python gaco.py --initial-window 4096 --json gaco_report.json"
GACO_SOURCES = ("Load saved report", "Run GACO live")


def gaco_tokens(value: object) -> int | None:
    """A window/token number stored bare or as an object ({"tokens": n} or {"floor": n})."""
    if isinstance(value, dict):
        value = value.get("tokens", value.get("floor"))
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def gaco_probe_rows(report: dict) -> list[dict]:
    details = (report.get("search") or {}).get("probe_details")
    if details:
        return [{
            "#": i,
            "window": d.get("window"),
            "purpose": d.get("purpose", ""),
            "result": "PASS" if d.get("passed") else "FAIL",
            "assembled tokens": d.get("assembled_tokens"),
            "processed tokens": d.get("processed_tokens"),
            "truncated": d.get("truncated"),
            "failure category": "" if d.get("passed") else (d.get("category") or ""),
        } for i, d in enumerate(details, 1)]
    rows = []
    for i, probe in enumerate(report.get("probes") or (report.get("search") or {}).get("probes") or [], 1):
        if isinstance(probe, dict):
            row = {"#": i, "window": probe.get("window"), "result": "PASS" if probe.get("passed") else "FAIL"}
            for key, label in (("assembled_tokens", "assembled tokens"), ("processed_tokens", "processed tokens")):
                if key in probe:
                    row[label] = probe[key]
        else:
            window, passed = (list(probe) + [None, None])[:2]
            row = {"#": i, "window": window, "result": "PASS" if passed else "FAIL"}
        rows.append(row)
    return rows


def render_gaco_report(report: dict, source: str) -> None:
    search = report.get("search") or {}
    rows = gaco_probe_rows(report)
    floor = gaco_tokens(report.get("floor"))
    suggested = gaco_tokens(report.get("suggested_window"))
    initial = gaco_tokens(report.get("initial_window"))
    found = bool(report.get("found")) and suggested is not None
    runs = next((report[k] for k in ("model_runs", "runs_used") if isinstance(report.get(k), int)), len(rows))
    finding = report.get("finding") or {}
    st.caption(
        f"Source: {source} · model {report.get('model', 'unknown')} · case {report.get('case_id', 'unknown')}"
        + (f" · tracked finding {finding.get('id')} v{finding.get('version')} ({finding.get('finding')})"
           if finding else "")
    )
    if found:
        if initial is None:
            relation = ""
        elif suggested == initial:
            relation = f"(the same as the initial {initial})."
        else:
            side = "below" if suggested < initial else "above"
            relation = f"({abs(suggested - initial)} tokens {side} the initial {initial})."
        st.markdown(
            '<div class="verdict continue"><div class="label">GACO SUGGESTION · ADVISORY</div>'
            f'<h2>Suggested window: {esc(suggested)}</h2><p>{esc(relation)}</p>'
            "<p>A recommendation only: nothing in the client's pipeline or settings was changed.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        ceiling = search.get("hi") or (report.get("settings") or {}).get("max_window")
        reason = search.get("stop_reason")
        st.markdown(
            '<div class="verdict review"><div class="label">GACO SUGGESTION · ADVISORY</div>'
            "<h2>No safe window found</h2>"
            f"<p>No window passed up to the ceiling{esc(f' of {ceiling} tokens' if ceiling else '')}"
            f"{esc(f' ({reason})' if reason else '')}. The case may not fit this model's window at all: it may "
            "need less per-case evidence, or a model with a larger context window.</p></div>",
            unsafe_allow_html=True,
        )
    facts = [
        ("Floor · measured", floor if floor is not None else "—"),
        ("Suggested · found by search", suggested if found else "—"),
        ("Initial window (client)", initial if initial is not None else "—"),
        ("Direction", report.get("direction", "—")),
        ("Real model runs used", runs),
    ]
    st.markdown(
        '<div class="why-grid">'
        + "".join(f"<div><small>{esc(k)}</small><strong>{esc(v)}</strong></div>" for k, v in facts)
        + "</div>",
        unsafe_allow_html=True,
    )
    floor_parts = report.get("floor") if isinstance(report.get("floor"), dict) else {}
    if floor_parts.get("prompt_tokens") is not None:
        st.caption(
            f"Floor = {floor_parts.get('prompt_tokens')} prompt tokens + {floor_parts.get('output_room_tokens')} "
            f"reply room ({floor_parts.get('output_room_source', 'config')}) + "
            f"{floor_parts.get('margin_tokens', 0)} margin, for {floor_parts.get('agent', 'the decision agent')}. "
            "Counted with the runtime's tokenizer; no model runs."
        )
    st.markdown("**Probe ladder** (each row is one real run of the clean case, in the order GACO ran it)")
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info("No probes were run: the measured floor is already above the ceiling.")
    if search.get("violations"):
        st.warning("Monotonicity violated, so it was treated as model variance and the suggestion was widened: "
                   + "; ".join(search["violations"]))
    notes = list(getattr(gaco, "NOTES", ())) or [
        "Advisory: GACO suggests a window; it does not change the client's pipeline or settings.",
        "The search finds where the required evidence fits and survives; it does not prove the model reasons "
        "over it well in every case.",
        "This is for the shown case only; across a workload, take the largest suggested window.",
    ]
    if rows and all(row["result"] == "PASS" for row in rows):
        notes.append(
            "Every probe passed: this is a clean case. The floor is an exact count and the search confirmed it, "
            "rather than resolving a failure. GACO never probes below the floor, so it cannot suggest less."
        )
    if report.get("evidence_check") == "measured-only" or any(
            d.get("evidence_check") == "measured-only" for d in search.get("probe_details") or []):
        notes.append(
            "Measured-only: this runtime cannot expose the model-visible prompt, so 'fits and survives' is measured "
            "from token counts and the runtime's processed-token count, not verified against what the model saw."
        )
    confidence = f"<p>{esc(report['confidence'])}</p>" if report.get("confidence") else ""
    st.markdown(
        '<div class="loss"><h3>What this does and does not show</h3>' + confidence
        + f'<ul>{"".join(f"<li>{esc(n)}</li>" for n in notes)}</ul></div>',
        unsafe_allow_html=True,
    )


if selected_reference == "Healthcare":
    st.markdown('<p class="section-label">GACO · SMALLEST SAFE CONTEXT WINDOW</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">GACO · smallest safe context window</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-help">GACO suggests the smallest window at which this case\'s required evidence fits '
        "and survives, and shows how far the client's current setting is from it. It is <b>advisory</b>: a "
        "recommendation, not a change to the pipeline. Sample case: "
        f"<code>harness_cases/{esc(GACO_CASE_FILE.name)}</code>.</p>",
        unsafe_allow_html=True,
    )
    gaco_source = st.radio("GACO report source", GACO_SOURCES, horizontal=True, key="gaco_source")
    if gaco_source == GACO_SOURCES[0]:
        if not GACO_SAVED_REPORT.exists():
            st.info(f"No saved GACO report yet. Create one from this folder with: `{GACO_SAVED_COMMAND}`")
        else:
            try:
                gaco_saved = json.loads(GACO_SAVED_REPORT.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                st.error(f"{GACO_SAVED_REPORT.name} could not be read: {exc}")
            else:
                render_gaco_report(gaco_saved, GACO_SAVED_REPORT.name)
    else:
        st.caption("A live run makes several real llama3.1:8b calls through Ollama and takes about a minute.")
        gaco_initial = st.number_input("Client's initial context window (tokens)", min_value=1, value=4096,
                                       step=1, key="gaco_initial_window")
        if st.button("Run GACO", key="gaco_run"):
            st.session_state.pop("gaco_live", None)
            with st.spinner("Running GACO: several real llama3.1:8b calls, about a minute"):
                try:
                    gaco_case = case_json_for(
                        str(PIPELINE_PATH), json.loads(GACO_CASE_FILE.read_text(encoding="utf-8")),
                    )
                    gaco_report = gaco.optimize(gaco_case, int(gaco_initial), harness=load_harness(gaco_case))
                except (RuntimeError, ValueError, OSError, KeyError, AssertionError) as exc:
                    st.session_state.gaco_live = {"error": str(exc)}
                else:
                    st.session_state.gaco_live = {"report": gaco_report}
        gaco_live = st.session_state.get("gaco_live")
        if gaco_live and gaco_live.get("error"):
            st.error(f"GACO could not run: {gaco_live['error']}")
        elif gaco_live:
            render_gaco_report(gaco_live["report"], "live run")

result = st.session_state.get("governance_review")
if result is None or result["domain"] != selected_reference or result.get("decision_mode") != decision_mode:
    st.info("Enter a live case above to run it through the loaded client pipeline.")
    st.stop()

review = result["review"]
integrity = result["integrity"]
view = build_governance_view(
    result["domain"], review, integrity, result["final_agent"], result["run_engine"],
    result["run_engine_fallback"], result.get("llm_decision"), result.get("llm_allergy"),
)
steps, decision, consumer, producer = view.steps, view.decision, view.consumer, view.producer
conflict_parent, consumed_finding, consumed_version = view.conflict_parent, view.consumed_finding, view.consumed_version
action, action_label, verdict_css = view.action, view.action_label, view.verdict_css

# --------------------------------------------------------------------------- event stream
st.markdown('<p class="section-label">LIVE GOVERNANCE RUN</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-help"><span class="tag fw">FRAMEWORK</span>generic stages, identical for every pipeline '
    f'&nbsp;&nbsp;<span class="tag plug">{esc(result["plugin_label"])}</span>the swappable domain pieces slotted into them.</p>',
    unsafe_allow_html=True,
)
events = view.events

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
) + (
    '<span class="pill llm">DECISION AGENT: REAL LLM (CONTEXT-LOSS DEMO)</span>'
    if result.get("llm_decision") is not None
    else '<span class="pill live">DECISION AGENT: DETERMINISTIC</span>'
) + (
    '<span class="pill llm">ALLERGY AGENT: REAL LLM (PII-LEAK DEMO)</span>'
    if result.get("llm_allergy") is not None else ""
)
st.markdown(
    f'<div class="simulator">{engine_pill}{event_html}</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- state comparison and verdict
st.markdown('<p class="section-label">WHERE & WHY</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-help">The framework makes the decision below from its generic stages; '
    f'the {esc(result["domain"])} case is the client input it governed.</p>',
    unsafe_allow_html=True,
)
if integrity.passed and action == "allow":
    st.success("No governance failure: the pipeline agents used consistent state and the case passed its checks.")
elif integrity.passed:
    st.info(
        "Deterministic trace integrity passed: every agent consumed what its parents emitted. "
        "The governance decision below comes from another check."
    )
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

def contrast_table(highlight: str) -> str:
    """Which layer catches which real-LLM failure; the row for this run is highlighted."""
    rows = CONTRAST_ROWS
    bold = ' style="font-weight:700"'
    body = "".join(
        f'<tr{bold if key == highlight else ""}><td>{esc(failure)}</td>'
        f"<td>{esc(trace)}</td><td>{esc(nemo)}</td></tr>"
        for key, failure, trace, nemo in rows
    )
    return (
        '<p><b>Each failure is caught by the layer suited to it and missed by the other</b> '
        "(this run's row in bold):</p>"
        "<table><tr><th>Real-LLM failure</th><th>Deterministic trace check</th><th>NeMo content rail</th></tr>"
        f"{body}</table>"
    )


llm_run = result.get("llm_decision")
if llm_run is not None:
    window_text = {True: '<span class="kept">inside</span>', False: '<span class="lost">outside</span>',
                   None: "unmeasured"}
    report_rows = "".join(
        f"<tr><td>{esc(agent)}</td><td>{esc(steps[agent].finding)} · v{steps[agent].finding_version}</td>"
        f"<td>{window_text[llm_run.in_window.get(agent)]}</td>"
        f"<td>{esc(llm_run.consumed.get(agent, 'not present (lost)'))}</td></tr>"
        for agent in llm_run.in_window
    )
    nemo_decision = next((c for c in review.content_checks if c["agent"] == result["final_agent"]), None)
    nemo_text = (
        f"NeMo ran on the decision agent's output with the configured rail ({nemo_decision['rail']}) and "
        f"returned: {nemo_decision['reason']} That rail checks the agent's own output for prompt injection and "
        "personal data; it does not judge grounding, so the catch here comes from the deterministic trace check."
        if nemo_decision else "NeMo did not run on the decision agent's output."
    )
    st.markdown(
        '<div class="loss"><h3>Real context loss: the decision agent is a real LLM with a constrained window</h3>'
        f'<p>{esc(llm_run.model)} ran with <b>num_ctx={llm_run.num_ctx}</b>. Ollama kept only the last '
        f'<b>{llm_run.kept_tokens}</b> prompt tokens'
        + (" because the case file was longer than the window" if llm_run.prompt_truncated else "")
        + '. The table compares what each branch emitted, whether that report was inside the kept window '
        "(measured with Ollama's own token counts), and what the agent itself said it saw.</p>"
        '<table><tr><th>Branch</th><th>Emitted</th><th>In agent window</th><th>Agent reported seeing</th></tr>'
        f'{report_rows}</table>'
        f'<p><b>Agent decided:</b> {esc(llm_run.disposition)} &nbsp;·&nbsp; '
        f'<b>Records require:</b> {esc(review.attribution.expected_disposition)}</p>'
        '<p>Each branch finding lives in the trace and the records, outside the agent. The framework compares '
        "what the decision agent consumed against what each parent emitted, so it catches the lost finding "
        "whether or not the agent's answer happens to be right.</p>"
        f'<p>{esc(nemo_text)}</p>'
        + contrast_table("context-loss") + '</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Decision agent prompt and raw output", expanded=False):
        st.markdown("**Full case file sent to the agent** (Ollama kept only the tail that fits the window)")
        st.code(llm_run.prompt)
        st.markdown("**Raw model output**")
        st.code(llm_run.raw_output)

allergy_run = result.get("llm_allergy")
if allergy_run is not None:
    allergy_agent = "allergy-safety"
    nemo_allergy = next((c for c in review.content_checks if c["agent"] == allergy_agent), None)
    if nemo_allergy is None:
        nemo_line = "NeMo did not run on the allergy agent's note."
    elif nemo_allergy["rail"] == "unavailable":
        nemo_line = f"NeMo was unavailable for the allergy agent's note ({nemo_allergy['reason']})."
    else:
        nemo_line = (
            f"<b>{'BLOCKED' if not nemo_allergy['passed'] else 'ALLOWED'}</b> by the content rail "
            f"({esc(nemo_allergy['rail'])}): {esc(nemo_allergy['reason'])}"
        )
    allergy_step = steps[allergy_agent]
    allergy_check = next((c for c in review.attribution.branch_checks if c.agent == allergy_agent), None)
    st.markdown(
        '<div class="loss"><h3>Real content failure: the allergy-safety agent is a real LLM with a permissive prompt</h3>'
        f'<p>{esc(allergy_run.model)} received the full patient record and was asked only for a short, self-contained '
        "handoff note. "
        + ("This run used the <b>control</b> prompt, which adds one redaction instruction. "
           if allergy_run.redacted else
           "The prompt never asks for identifiers and has no redaction instruction. ")
        + "Whatever personal data appears in the note is the model's own output.</p>"
        f'<p><b>Agent status:</b> {esc(allergy_run.status)} &nbsp;·&nbsp; <b>Records require:</b> '
        f'{esc((allergy_check.recomputed if allergy_check else "-").upper())}</p>'
        f'<p><b>Handoff note</b> (identifiers masked for display): '
        f'<code>{esc(redact_note(result["records"], allergy_run.note))}</code></p>'
        f'<p><b>Record identifiers found verbatim in the note:</b> '
        f'{esc(", ".join(allergy_run.identifiers_in_note) or "none")} (a display check; NeMo makes the call)</p>'
        f'<p><b>NeMo content rail (non-deterministic) on the allergy agent:</b> {nemo_line}</p>'
        f'<p><b>Deterministic trace check:</b> {"passed" if integrity.passed else "flagged"} '
        f"({esc(integrity.reason)}) &nbsp;·&nbsp; <b>Correctness plug-in:</b> "
        f'{"passed" if review.attribution.passed else "flagged"}. '
        "A personal-data leak is a content problem, not a provenance problem, so the deterministic checks have "
        "nothing to flag unless the agent's status itself is wrong.</p>"
        + contrast_table("pii-leak") + '</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Allergy agent prompt and raw output (unredacted, synthetic data)", expanded=False):
        st.markdown("**Prompt sent to the agent**")
        st.code(allergy_run.prompt)
        st.markdown("**Raw model output**")
        st.code(allergy_run.raw_output)

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
