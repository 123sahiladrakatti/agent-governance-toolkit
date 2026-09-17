"""Manager-facing Streamlit UI for the AML governance comparison."""

from __future__ import annotations

import time

import streamlit as st

from agent_sim import run_session
from domain import AlertOutcome
from governance import StatefulGovernanceMonitor

st.set_page_config(page_title="AML Governance Review", page_icon="◆", layout="wide")

st.markdown("""
<style>
:root { --ink:#17212b; --muted:#667580; --line:#d7e0e5; --paper:#f5f7f6; --green:#176b45; --green-bg:#e7f5ed; --red:#b12632; --red-bg:#fff0f1; --gold:#9a6819; }
html, body, [class*="css"] { font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,sans-serif; }
.stApp { background:var(--paper); color:var(--ink); } .block-container { max-width:1180px; padding:2rem 1.5rem 4rem; }
.hero { padding:1.5rem 1.7rem; border:1px solid var(--line); background:#fff; } .eyebrow { color:var(--red); font:600 .72rem ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.12em; text-transform:uppercase; }
.hero h1 { margin:.35rem 0 .45rem; font-size:clamp(2rem,4vw,3.2rem); letter-spacing:-.04em; } .hero p { max-width:760px; margin:0; color:var(--muted); font-size:1rem; line-height:1.5; }
.explain { margin:1rem 0; padding:1rem 1.2rem; border-left:5px solid var(--red); background:#fff; line-height:1.5; } .explain b { color:var(--red); }
.control-box { padding:1rem 1.2rem; border:1px solid var(--line); background:#fff; } .control-label { margin-bottom:.3rem; font-weight:700; } .control-help { color:var(--muted); font-size:.85rem; }
.score { display:grid; grid-template-columns:repeat(4,1fr); gap:.75rem; margin:1.2rem 0; } .metric { min-height:88px; padding:1rem; border:1px solid var(--line); background:#fff; } .metric strong { display:block; font-size:1.9rem; } .metric small { color:var(--muted); } .metric.hot { border-color:#e4aeb2; background:#fff8f8; } .metric.hot strong { color:var(--red); }
.legend { display:flex; gap:1.2rem; flex-wrap:wrap; margin:1rem 0; color:var(--muted); font-size:.84rem; } .dot { display:inline-block; width:10px; height:10px; margin-right:5px; border-radius:50%; } .dot.green { background:#329a68; } .dot.red { background:var(--red); }
.section-title { margin:1.5rem 0 .5rem; font-size:1.3rem; } .section-help { color:var(--muted); font-size:.9rem; margin-bottom:.8rem; }
.result { padding:1rem 1.1rem; border:1px solid var(--line); background:#fff; } .result.flag { border-color:#e4aeb2; background:var(--red-bg); } .status { display:inline-block; float:right; padding:.25rem .45rem; font:700 .7rem ui-monospace,SFMono-Regular,Menlo,monospace; } .status.pass { color:var(--green); background:var(--green-bg); } .status.flag { color:var(--red); background:#ffdfe1; }
.result h3 { margin:0; font-size:1rem; } .result p { margin:.45rem 0 0; color:var(--muted); line-height:1.45; font-size:.88rem; } .star-label { color:var(--gold); font:700 .73rem ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.08em; text-transform:uppercase; } .evidence { margin-top:.8rem; padding:.8rem; border-top:1px solid #ecd4d5; color:#5c4145; font-size:.86rem; line-height:1.5; }
@media(max-width:750px){ .score{grid-template-columns:repeat(2,1fr)} .block-container{padding:1rem .8rem 3rem} }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="eyebrow">AML placement stage · synthetic demonstration</div><h1>Can a permitted agent still be wrong?</h1><p>This review shows the difference between checking whether an agent was allowed to act and checking whether its permitted decision matched the transaction records.</p></div>', unsafe_allow_html=True)
st.markdown('<div class="explain"><b>How to read the results:</b> Regular governance asks <b>“Was this action permitted?”</b> Enhanced governance asks <b>“Was the permitted action correct?”</b> The important failure is a green Regular PASS beside a red Enhanced FLAG.</div>', unsafe_allow_html=True)

control_left, control_mid, control_right = st.columns([1.2, 1, 1])
with control_left:
    st.markdown('<div class="control-box"><div class="control-label">Session size</div><div class="control-help">More alerts make the run feel long-running.</div></div>', unsafe_allow_html=True)
    alert_count = st.slider("Alerts to process", 8, 80, 40, label_visibility="collapsed")
with control_mid:
    st.markdown('<div class="control-box"><div class="control-label">Replay seed</div><div class="control-help">Same seed gives the same demo.</div></div>', unsafe_allow_html=True)
    seed = st.number_input("Replay seed", min_value=1, value=7, step=1, label_visibility="collapsed")
with control_right:
    st.markdown('<div class="control-box"><div class="control-label">Start review</div><div class="control-help">Synthetic records only. No LLM.</div></div>', unsafe_allow_html=True)
    run = st.button("Run AML session", type="primary", use_container_width=True)

if "outcomes" not in st.session_state:
    st.session_state.outcomes = []
if run:
    progress = st.progress(0, text="Preparing the investigation run")
    monitor = StatefulGovernanceMonitor(context_window=8, checkpoint_interval=5)
    results: list[AlertOutcome] = []
    for number, (alert, action) in enumerate(run_session(alert_count, int(seed)), start=1):
        regular_verdict, enhanced_verdict = monitor.evaluate(action, alert)
        results.append(AlertOutcome(alert, action, regular_verdict, enhanced_verdict))
        progress.progress(number / alert_count, text=f"Reviewing alert {number} of {alert_count}")
        time.sleep(0.025)
    st.session_state.outcomes = results
    st.session_state.session_metrics = monitor.metrics
    progress.empty()

outcomes: list[AlertOutcome] = st.session_state.outcomes
if not outcomes:
    st.info("Choose the session size and click Run AML session. The default run always includes the highlighted star case.")
    st.stop()

metrics = st.session_state.session_metrics

regular_flags = sum(not item.regular.passed for item in outcomes)
enhanced_flags = sum(not item.enhanced.passed for item in outcomes)
missed = sum(item.regular.passed and not item.enhanced.passed for item in outcomes)
st.markdown(f'<div class="score"><div class="metric"><strong>{len(outcomes)}</strong><small>Alerts reviewed</small></div><div class="metric"><strong>{regular_flags}</strong><small>Regular flags</small></div><div class="metric"><strong>{enhanced_flags}</strong><small>Enhanced flags</small></div><div class="metric hot"><strong>{missed}</strong><small>Authorized but wrong</small></div></div>', unsafe_allow_html=True)
st.markdown('<div class="legend"><span><i class="dot green"></i>PASS: no issue found by this lane</span><span><i class="dot red"></i>FLAG: review needed</span><span>★ Star case: the clearest silent failure</span></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Long-running session health</div><div class="section-help">This is the state retained across alerts. The window stays bounded while the session counters continue to accumulate.</div>', unsafe_allow_html=True)
health_left, health_mid, health_right = st.columns(3)
with health_left:
    st.metric("Current turn", f"{metrics.turns} / {len(outcomes)}")
    st.caption(f"Checkpoint turns: {', '.join(map(str, metrics.checkpoints)) or 'none'}")
with health_mid:
    st.metric("Context pressure", f"{metrics.context_pressure:.1f}x")
    st.caption(f"Retained context: {metrics.retained_context} of {metrics.context_window} slots")
with health_right:
    st.metric("Accumulated drift score", metrics.drift_score)
    st.caption(f"Stateful misses: {metrics.regular_missed} authorized-but-wrong actions")

filter_choice = st.radio("Show", ["All alerts", "Enhanced flags", "Regular PASS / Enhanced FLAG", "Star case"], horizontal=True)
visible = []
for item in outcomes:
    if filter_choice == "Enhanced flags" and item.enhanced.passed:
        continue
    if filter_choice == "Regular PASS / Enhanced FLAG" and not (item.regular.passed and not item.enhanced.passed):
        continue
    if filter_choice == "Star case" and not item.action.is_star_case:
        continue
    visible.append(item)

st.markdown('<div class="section-title">Alert review</div><div class="section-help">Open an alert to compare both governance decisions and see the plain-language reason.</div>', unsafe_allow_html=True)
for item in visible:
    action = item.action
    star = action.is_star_case
    title = f"{'★ ' if star else ''}{action.alert_id} · account {action.acted_account_id} · agent disposition: {action.disposition}"
    with st.expander(title, expanded=star):
        if star:
            st.markdown('<div class="star-label">★ Star case · silent cleared structuring pattern</div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            if item.regular.passed:
                st.markdown('<div class="result"><span class="status pass">PASS</span><h3>Regular governance</h3><p>Permission check passed. The agent used an allowed action.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result flag"><span class="status flag">FLAG</span><h3>Regular governance</h3><p>{item.regular.reason}</p></div>', unsafe_allow_html=True)
        with right:
            enhanced_status = "PASS" if item.enhanced.passed else "FLAG"
            enhanced_class = "result" if item.enhanced.passed else "result flag"
            status_class = "pass" if item.enhanced.passed else "flag"
            st.markdown(f'<div class="{enhanced_class}"><span class="status {status_class}">{enhanced_status}</span><h3>Enhanced governance</h3><p>{item.enhanced.reason}</p></div>', unsafe_allow_html=True)
        if not item.enhanced.passed:
            st.markdown(f'<div class="evidence"><b>Why this was flagged:</b> {item.enhanced.reason}</div>', unsafe_allow_html=True)

st.caption("Synthetic AML governance demonstration. It is not a real AML detector, SAR filing system, or compliance decision.")
