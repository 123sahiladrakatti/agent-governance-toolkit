
"""Runtime governance demo with swappable AML and healthcare client pipelines.

Run: streamlit run app.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from batch_dashboard import render_batch_dashboard
from client_pipeline import ClientPipelineError, load_pipeline_definition
import gaco
from instrumentation import case_json_for, load_harness
from longrun_demo import report_summary, run_degradation_demo, run_demo


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
try:
    # The backend pipeline definition is loaded automatically; there is no case-entry form.
    pipeline_definition = load_pipeline_definition(PIPELINE_PATH)
except ClientPipelineError as exc:
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

# --------------------------------------------------------------------------- materialized backend batch
render_batch_dashboard(selected_reference)

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
