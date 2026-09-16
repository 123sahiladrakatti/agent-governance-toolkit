"""Client-facing Streamlit dashboard for the healthcare governance demo."""

from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Healthcare Agent Governance",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#142f3a; --muted:#687e87; --line:#d9e6e8; --paper:#f5f8f6; --teal:#159a8a; --blue:#287ca7; --amber:#d78b2d; --red:#c9564b; }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
    .stApp { background:var(--paper); color:var(--ink); }
    .block-container { max-width:1240px; padding:2.3rem 2rem 4rem; }
    h1,h2,h3 { font-family:'Space Grotesk', sans-serif !important; letter-spacing:-.03em; }
    h1 { font-size:clamp(2rem, 4vw, 3.55rem) !important; line-height:1.02 !important; }
    h2 { font-size:1.55rem !important; }
    [data-testid="stSidebar"] { background:#102d37; }
    [data-testid="stSidebar"] * { color:#e9f5f3 !important; }
    .eyebrow { color:var(--teal); font-size:.72rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; }
    .hero-copy { max-width:790px; padding:1rem 0 1.4rem; }
    .hero-copy p { color:var(--muted); font-size:1.08rem; line-height:1.6; max-width:720px; }
    .synthetic { display:inline-block; padding:.38rem .65rem; border:1px solid #bce0d9; border-radius:999px; color:#197467; background:#e8f7f2; font-size:.72rem; font-weight:700; }
    .panel { height:100%; padding:1.15rem 1.2rem; border:1px solid var(--line); border-radius:14px; background:#fff; box-shadow:0 8px 24px rgba(20,47,58,.045); }
    .panel-kicker { color:var(--muted); font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
    .panel-title { margin:.3rem 0 .85rem; font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:700; }
    .metric-card { min-height:120px; padding:1.1rem 1.15rem; border:1px solid var(--line); border-radius:14px; background:#fff; }
    .metric-label { color:var(--muted); font-size:.75rem; font-weight:600; }
    .metric-value { margin:.3rem 0; font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; }
    .metric-note { color:var(--muted); font-size:.72rem; line-height:1.35; }
    .metric-card.teal { border-top:4px solid var(--teal); }.metric-card.blue { border-top:4px solid var(--blue); }.metric-card.amber { border-top:4px solid var(--amber); }.metric-card.red { border-top:4px solid var(--red); }
    .section-gap { margin-top:1.4rem; }
    .flow { display:flex; align-items:stretch; gap:.55rem; margin:.8rem 0 .2rem; }
    .flow-node { flex:1; min-height:112px; padding:.9rem; border:1px solid #cde1e3; border-radius:11px; background:#f4fbfa; }
    .flow-node strong { display:block; margin-bottom:.35rem; font-family:'Space Grotesk',sans-serif; font-size:.9rem; }.flow-node span { color:var(--muted); font-size:.75rem; line-height:1.4; }
    .flow-arrow { display:grid; place-items:center; color:var(--teal); font-size:1.35rem; }
    .insight { padding:.85rem 1rem; margin:.55rem 0; border-left:4px solid var(--teal); background:#eef8f5; border-radius:0 9px 9px 0; }
    .insight.warning { border-color:var(--amber); background:#fff7e9; }.insight strong { display:block; font-family:'Space Grotesk',sans-serif; font-size:.86rem; }.insight span { color:#526a72; font-size:.76rem; line-height:1.45; }
    .legend { display:flex; gap:1rem; flex-wrap:wrap; margin:.45rem 0 .9rem; color:var(--muted); font-size:.73rem; }.legend i { display:inline-block; width:9px; height:9px; margin-right:4px; border-radius:50%; }.allow { background:var(--teal); }.pause { background:var(--amber); }.baseline { background:#8da2aa; }
    .footer-note { margin-top:1.8rem; color:#789097; font-size:.73rem; text-align:center; }
    @media(max-width:760px){ .block-container{padding:1.3rem 1rem 3rem}.flow{flex-direction:column}.flow-arrow{transform:rotate(90deg)} }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, note: str, tone: str) -> None:
    st.markdown(
        f'<div class="metric-card {tone}"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def workflow_rows(turns: int, drift_turn: int) -> list[dict[str, object]]:
    rows = []
    for turn in range(1, turns + 1):
        drifted = turn == drift_turn
        task = "Export complete patient record" if drifted else "Prepare patient-intake summary"
        rows.append(
            {
                "Turn": turn,
                "Task": task,
                "Regular": "Allowed",
                "Long-running": "Paused" if drifted else "Allowed",
                "Event": "Goal drift" if drifted else "Normal handoff",
            }
        )
        if drifted:
            break
    return rows


st.markdown('<div class="eyebrow">Healthcare agent operations · client preview</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-copy"><h1>Governance that notices when a safe workflow changes direction.</h1><p>See the difference between checking each request in isolation and supervising an agent workflow across time. The scenario is synthetic and runs entirely in the browser session.</p><span class="synthetic">SYNTHETIC DATA · NO BACKEND CONNECTION</span></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Scenario controls")
    turns = st.slider("Workflow length", min_value=5, max_value=60, value=30, step=5)
    drift_turn = st.slider("Simulated drift turn", min_value=2, max_value=turns, value=min(25, turns), step=1)
    st.caption("The drift event changes the task from patient-intake summarization to external record export.")
    st.divider()
    st.markdown("**Demo scope**")
    st.caption("This is a visual client demonstration. It does not connect to an EHR, model provider, or production policy service.")

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
controls_left, controls_right = st.columns([2, 1])
with controls_left:
    st.markdown('<div class="panel"><div class="panel-kicker">Live scenario</div><div class="panel-title">Four agents, one shared task, one deliberate drift event</div><div class="flow"><div class="flow-node"><strong>Intake agent</strong><span>Owns the patient-intake goal and coordinates the care-team workflow.</span></div><div class="flow-arrow">→</div><div class="flow-node"><strong>Governance layer</strong><span>Regular gate checks content. Long-running gate checks content, goal, and lifecycle.</span></div><div class="flow-arrow">→</div><div class="flow-node"><strong>Care-team agents</strong><span>Records reviews history. Triage checks escalation. Scheduling identifies the next operational step.</span></div></div></div>', unsafe_allow_html=True)
with controls_right:
    st.markdown('<div class="panel"><div class="panel-kicker">Selected run</div><div class="panel-title">Control settings</div>', unsafe_allow_html=True)
    st.metric("Workflow turns", turns)
    st.metric("Drift injected", f"Turn {drift_turn}")
    st.markdown('</div>', unsafe_allow_html=True)

regular_completed = turns
long_running_completed = drift_turn - 1
checkpoint_count = (long_running_completed // 5)

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
metric_columns = st.columns(4)
with metric_columns[0]:
    metric_card("Regular path", f"{regular_completed}/{turns}", "Turns allowed to complete", "blue")
with metric_columns[1]:
    metric_card("Long-running path", f"{long_running_completed}/{turns}", "Turns completed before pause", "teal")
with metric_columns[2]:
    metric_card("Detection point", f"Turn {drift_turn}", "Goal drift caught before handoff", "amber")
with metric_columns[3]:
    metric_card("Checkpoints", str(checkpoint_count), "Saved before the pause", "red")

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
st.markdown('<div class="panel"><div class="panel-kicker">Executive readout</div><div class="panel-title">What should a client take away?</div>', unsafe_allow_html=True)
st.markdown('<div class="insight"><strong>Regular controls are point-in-time.</strong><span>They can confirm that a single request has no obvious MRN or SSN-like pattern, but they do not know whether the workflow is still pursuing its original goal.</span></div>', unsafe_allow_html=True)
st.markdown('<div class="insight warning"><strong>Long-running controls are stateful.</strong><span>They compare each new task with the original goal and pause the workflow before the drifted handoff reaches the records agent.</span></div>', unsafe_allow_html=True)
st.markdown('<div class="insight"><strong>Token use stays bounded.</strong><span>The prototype carries a goal, recent actions, and checkpoints instead of replaying the full transcript on every turn.</span></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
tab_timeline, tab_comparison, tab_next = st.tabs(["Run timeline", "Capability comparison", "Roadmap"])
with tab_timeline:
    st.markdown('<div class="panel"><div class="panel-kicker">Synthetic event log</div><div class="panel-title">What happens on each turn</div>', unsafe_allow_html=True)
    st.dataframe(workflow_rows(turns, drift_turn), use_container_width=True, hide_index=True)
    st.caption("The regular path would continue after the drift row. The long-running path pauses there, so no later handoff occurs.")
    st.markdown('</div>', unsafe_allow_html=True)
with tab_comparison:
    st.markdown('<div class="panel"><div class="panel-kicker">Current versus proposed</div><div class="panel-title">From a working multi-agent skeleton to resilient governance</div>', unsafe_allow_html=True)
    comparison = [
        {"Capability": "Four-agent handoffs", "Current baseline": "Yes", "Long-running direction": "Yes, monitored"},
        {"Capability": "Sensitive input check", "Current baseline": "Yes, narrow pattern gate", "Long-running direction": "Yes, retained"},
        {"Capability": "Goal-relative drift", "Current baseline": "No", "Long-running direction": "Pause and review"},
        {"Capability": "Checkpointing", "Current baseline": "No", "Long-running direction": "Periodic state snapshots"},
        {"Capability": "Crash recovery", "Current baseline": "No", "Long-running direction": "Next implementation stage"},
        {"Capability": "Scale and backpressure", "Current baseline": "No", "Long-running direction": "Later implementation stage"},
    ]
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
with tab_next:
    st.markdown('<div class="panel"><div class="panel-kicker">Build sequence</div><div class="panel-title">A staged path clients can understand</div>', unsafe_allow_html=True)
    roadmap = [
        ("01", "Multi-agent foundation", "Four agents exchange coordinated handoffs and responses."),
        ("02", "Regular governance", "Put the current content and tool checks around the handoff."),
        ("03", "Long-running governance", "Track goals, checkpoints, drift, and bounded context."),
        ("04", "Recovery", "Pause, restore a checkpoint, and resume safely."),
        ("05", "Scale", "Add identity, concurrency limits, routing, and audit at fleet level."),
    ]
    for number, title, copy in roadmap:
        st.markdown(f'<div class="insight"><strong>{number} · {title}</strong><span>{copy}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer-note">Client demonstration only · All events are generated locally · No patient, EHR, model, or backend data is used</div>', unsafe_allow_html=True)
