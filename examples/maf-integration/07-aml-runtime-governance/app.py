"""Streamlit UI for the AML long-running governance comparison."""

from __future__ import annotations

import time

import streamlit as st

from agent_sim import run_session
from domain import AlertOutcome
from governance import EnhancedGovernance, RegularGovernance

st.set_page_config(page_title="AML Runtime Governance", page_icon="◆", layout="wide")

st.markdown("""
<style>
:root { --ink:#17212b; --muted:#64717d; --line:#d8e0e5; --paper:#f7f8f6; --green:#16734a; --green-bg:#e6f4ec; --red:#b32631; --red-bg:#fff0f0; }
html, body, [class*="css"] { font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,sans-serif; } .stApp { background:var(--paper); color:var(--ink); }
.block-container { max-width:1240px; padding:2.2rem 2rem 4rem; } h1,h2,h3 { letter-spacing:-.035em; }
.hero { border-bottom:1px solid var(--line); padding:0 0 1.5rem; margin-bottom:1.4rem; } .eyebrow { color:var(--red); font:500 .72rem ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.14em; text-transform:uppercase; }
.hero h1 { font-size:clamp(2rem,4vw,3.35rem); margin:.35rem 0 .5rem; } .hero p { color:var(--muted); font-size:1.03rem; max-width:760px; margin:0; }
.pitch { display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--line); border:1px solid var(--line); margin:1.4rem 0; } .pitch div { padding:1rem 1.2rem; background:white; } .pitch b { display:block; margin-bottom:.25rem; } .pitch span { color:var(--muted); font-size:.88rem; }
.score { display:grid; grid-template-columns:repeat(4,1fr); gap:.7rem; margin:1.5rem 0; } .metric { background:white; border:1px solid var(--line); padding:1rem 1.1rem; } .metric strong { display:block; font-size:1.8rem; } .metric small { color:var(--muted); } .metric.hot { border-color:#efb9bc; background:#fff8f8; } .metric.hot strong { color:var(--red); }
.lane-head { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:1rem; } .lane { padding:.85rem 1rem; border-top:4px solid; background:white; } .lane.reg { border-color:#329a68; } .lane.enh { border-color:var(--red); } .lane small { color:var(--muted); }
.row { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin:.65rem 0; } .verdict { background:white; border:1px solid var(--line); padding:.85rem 1rem; } .verdict.pass { border-left:5px solid #329a68; } .verdict.flag { border-left:5px solid var(--red); background:var(--red-bg); } .verdict strong { font:600 .82rem ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.05em; } .verdict p { margin:.35rem 0 0; color:var(--muted); font-size:.84rem; } .tag { float:right; font:500 .68rem ui-monospace,SFMono-Regular,Menlo,monospace; padding:.25rem .4rem; border-radius:3px; } .tag.pass { color:var(--green); background:var(--green-bg); } .tag.flag { color:var(--red); background:#ffdfe1; }
.star-note { color:#825714; font:500 .76rem ui-monospace,SFMono-Regular,Menlo,monospace; text-transform:uppercase; letter-spacing:.08em; }
@media(max-width:700px){ .block-container{padding:1.3rem 1rem 3rem}.score{grid-template-columns:repeat(2,1fr)}.pitch,.lane-head,.row{grid-template-columns:1fr} }
</style>
""", unsafe_allow_html=True)

st.markdown('<section class="hero"><div class="eyebrow">Runtime governance / AML placement stage</div><h1>Can a permitted agent still be wrong?</h1><p>Long-running investigation agents can stay inside their permissions while drifting away from the records. This demo makes that silent failure visible.</p></section>', unsafe_allow_html=True)
st.markdown('<div class="pitch"><div><b>Regular governance</b><span>Was the action permitted? Deterministic access and tool checks.</span></div><div><b>Enhanced governance</b><span>Was the permitted action correct? Record-relative behavioral checks.</span></div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Session controls")
    alert_count = st.slider("Alerts in long run", 8, 80, 40)
    seed = st.number_input("Replay seed", min_value=1, value=7, step=1)
    run = st.button("Run session", type="primary", use_container_width=True)
    st.caption("Synthetic records only. No LLM calls, network calls, or external detector.")

if "outcomes" not in st.session_state:
    st.session_state.outcomes = []
if run:
    progress = st.progress(0, text="Starting long-running session")
    counter = st.empty()
    regular = RegularGovernance()
    enhanced = EnhancedGovernance()
    outcomes: list[AlertOutcome] = []
    for number, (alert, action) in enumerate(run_session(alert_count, int(seed)), start=1):
        outcomes.append(AlertOutcome(alert, action, regular.check(action, alert), enhanced.check(action, alert)))
        progress.progress(number / alert_count, text=f"Processing alert {number} of {alert_count}")
        counter.caption(f"Session is live · completed {number}/{alert_count} alerts · evaluating {alert.alert_id}")
        time.sleep(0.025)
    st.session_state.outcomes = outcomes
    progress.empty(); counter.empty()

outcomes = st.session_state.outcomes
if not outcomes:
    st.info("Set the alert count in the sidebar and run the session. The star case is guaranteed to appear in the default 40-alert run.")
    st.stop()

regular_flags = sum(not outcome.regular.passed for outcome in outcomes)
enhanced_flags = sum(not outcome.enhanced.passed for outcome in outcomes)
missed = sum(outcome.regular.passed and not outcome.enhanced.passed for outcome in outcomes)
st.markdown(f'<div class="score"><div class="metric"><strong>{len(outcomes)}</strong><small>Total alerts</small></div><div class="metric"><strong>{regular_flags}</strong><small>Regular flagged</small></div><div class="metric"><strong>{enhanced_flags}</strong><small>Enhanced flagged</small></div><div class="metric hot"><strong>{missed}</strong><small>Faulty dispositions regular missed</small></div></div>', unsafe_allow_html=True)
st.caption("Green PASS means the governance lane found no violation. Red FLAG means it found a violation. A green regular result beside a red enhanced result is the key contrast.")
st.markdown('<div class="lane-head"><div class="lane reg"><b>REGULAR GOVERNANCE · PERMISSION</b><br><small>Checks authorization and access scope only</small></div><div class="lane enh"><b>ENHANCED GOVERNANCE · BEHAVIOR</b><br><small>Recomputes the structuring pattern from case records</small></div></div>', unsafe_allow_html=True)

for outcome in outcomes:
    action = outcome.action
    label = " ★ STAR CASE" if action.is_star_case else ""
    with st.expander(f"{action.alert_id} · {action.acted_account_id} · {action.disposition}{label}", expanded=action.is_star_case):
        if action.is_star_case:
            st.markdown('<div class="star-note">★ Silent cleared structuring pattern · value misread / threshold drift</div>', unsafe_allow_html=True)
        regular_class = "pass" if outcome.regular.passed else "flag"
        enhanced_class = "pass" if outcome.enhanced.passed else "flag"
        regular_tag = "PASS" if outcome.regular.passed else "FLAGGED"
        enhanced_tag = "PASS" if outcome.enhanced.passed else "FLAGGED"
        st.markdown(f'<div class="row"><div class="verdict {regular_class}"><span class="tag {regular_class}">{regular_tag}</span><strong>REGULAR</strong><p>{outcome.regular.reason}</p></div><div class="verdict {enhanced_class}"><span class="tag {enhanced_class}">{enhanced_tag}</span><strong>ENHANCED</strong><p>{outcome.enhanced.reason}</p></div></div>', unsafe_allow_html=True)
        if not outcome.enhanced.passed:
            st.json({"failure_category": outcome.enhanced.category, "responsible_agent": outcome.enhanced.responsible_agent, "step": outcome.enhanced.step, "recomputed_vs_claimed": outcome.enhanced.details})

st.divider()
st.caption("Prototype boundary: this is a deterministic governance demonstration, not a real AML detector, SAR filing system, or production compliance decision.")
