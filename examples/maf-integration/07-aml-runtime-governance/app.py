"""Manager-facing Streamlit UI: runtime governance for a 3-agent AML chain.

The subject is agent governance, not money-laundering detection. AML is only the
setting. The demo shows a failure that cannot happen with a single hop: a value
misread by the first agent flows untouched through two trusting downstream
agents. Every agent-to-agent handoff is authorized (security is intact), yet the
end-to-end decision is wrong (governance fails) - and the governance lane names
the agent that originated the error versus those that merely propagated it.
"""

from __future__ import annotations

import time

import streamlit as st

from agent_sim import run_chain_session
from domain import recompute_structuring
from governance import ChainGovernance, ChainMonitor, SecurityLane

st.set_page_config(page_title="Agent Governance Review", page_icon="\u25c6", layout="wide")

st.markdown("""
<style>
:root { --ink:#17212b; --muted:#667580; --line:#d7e0e5; --paper:#f5f7f6; --green:#176b45; --green-bg:#e7f5ed; --red:#b12632; --red-bg:#fff0f1; --gold:#9a6819; --amber-bg:#fbf3e2; --blue:#1f5c8a; }
html, body, [class*="css"] { font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,sans-serif; }
.stApp { background:var(--paper); color:var(--ink); } .block-container { max-width:1180px; padding:2rem 1.5rem 4rem; }
.hero { padding:1.6rem 1.7rem; border:1px solid var(--line); background:#fff; }
.hero .kicker { color:var(--muted); font-size:.85rem; margin:0 0 .5rem; }
.hero h1 { margin:.1rem 0 .5rem; font-size:clamp(1.8rem,3.4vw,2.7rem); letter-spacing:-.03em; line-height:1.08; }
.hero p { max-width:800px; margin:0; color:var(--muted); font-size:1rem; line-height:1.55; }
.explain { margin:1rem 0 .4rem; padding:1rem 1.2rem; border-left:5px solid var(--red); background:#fff; line-height:1.55; } .explain b { color:var(--red); }
.stage-tag { margin:2rem 0 .35rem; color:var(--muted); font-size:.9rem; }
.stage-title { margin:.1rem 0 .3rem; font-size:1.4rem; letter-spacing:-.02em; }
.stage-help { color:var(--muted); font-size:.92rem; margin:0 0 1rem; max-width:780px; line-height:1.5; }
.scene { border:1px solid var(--line); background:#fff; padding:1.2rem 1.3rem; margin-bottom:1rem; }
.scene h4 { margin:0 0 .7rem; font-size:1.02rem; }
.facts { width:100%; border-collapse:collapse; font-size:.9rem; }
.facts th, .facts td { text-align:left; padding:.42rem .6rem; border-bottom:1px solid var(--line); }
.facts th { color:var(--muted); font-weight:600; }
.facts td.num { text-align:right; font-variant-numeric:tabular-nums; }
/* chain hops */
.chain { display:grid; grid-template-columns:1fr auto 1fr auto 1fr; align-items:stretch; gap:0; }
.hop { border:1px solid var(--line); background:#fff; padding:.9rem .95rem; }
.hop .who { font-weight:700; font-size:.95rem; margin:0 0 .35rem; }
.hop .role { color:var(--muted); font-size:.8rem; margin:0 0 .6rem; }
.hop .val { font-variant-numeric:tabular-nums; font-size:.88rem; margin:.15rem 0; }
.hop .sec { margin-top:.6rem; padding-top:.5rem; border-top:1px solid var(--line); font-size:.82rem; font-weight:600; }
.hop .sec.pass { color:var(--green); } .hop .sec.flag { color:var(--red); }
.hop.origin { border-color:var(--red); background:var(--red-bg); }
.hop.prop { border-color:#dfc98a; background:var(--amber-bg); }
.hop .tag { display:inline-block; margin-top:.5rem; padding:.12rem .4rem; font-size:.72rem; font-weight:700; }
.hop .tag.origin { color:var(--red); background:#ffdfe1; } .hop .tag.prop { color:var(--gold); background:#f3e4bf; }
.arrow { display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:1.1rem; padding:0 .5rem; }
.lane { border:1px solid var(--line); background:#fff; padding:1.1rem 1.15rem; height:100%; }
.lane.miss { border-color:#cdd9d0; background:#fbfdfb; } .lane.catch { border-color:#e4aeb2; background:var(--red-bg); }
.lane .q { color:var(--muted); font-size:.86rem; margin:0 0 .5rem; }
.lane .verdict { font-weight:700; font-size:1.05rem; margin:0 0 .5rem; }
.lane.miss .verdict { color:var(--green); } .lane.catch .verdict { color:var(--red); }
.lane .body { color:var(--ink); font-size:.9rem; line-height:1.5; margin:0; }
.takeaway { margin:.4rem 0 0; padding:1rem 1.2rem; background:var(--amber-bg); border:1px solid #ecd9ad; font-size:.96rem; line-height:1.55; }
.control-box { padding:1rem 1.2rem; border:1px solid var(--line); background:#fff; } .control-label { margin-bottom:.3rem; font-weight:700; } .control-help { color:var(--muted); font-size:.85rem; }
.score { display:grid; grid-template-columns:repeat(4,1fr); gap:.75rem; margin:1.2rem 0; } .metric { min-height:88px; padding:1rem; border:1px solid var(--line); background:#fff; } .metric strong { display:block; font-size:1.9rem; } .metric small { color:var(--muted); } .metric.hot { border-color:#e4aeb2; background:#fff8f8; } .metric.hot strong { color:var(--red); }
.legend { display:flex; gap:1.2rem; flex-wrap:wrap; margin:.6rem 0 0; color:var(--muted); font-size:.84rem; }
.result { padding:1rem 1.1rem; border:1px solid var(--line); background:#fff; } .result.flag { border-color:#e4aeb2; background:var(--red-bg); }
.status { display:inline-block; float:right; padding:.25rem .45rem; font:700 .7rem ui-monospace,SFMono-Regular,Menlo,monospace; } .status.pass { color:var(--green); background:var(--green-bg); } .status.flag { color:var(--red); background:#ffdfe1; }
.result h3 { margin:0; font-size:1rem; } .result p { margin:.45rem 0 0; color:var(--muted); line-height:1.45; font-size:.88rem; }
@media(max-width:820px){ .chain{grid-template-columns:1fr} .arrow{transform:rotate(90deg);padding:.3rem 0} .score{grid-template-columns:repeat(2,1fr)} .block-container{padding:1rem .8rem 3rem} }
</style>
""", unsafe_allow_html=True)


def money(v: float) -> str:
    return f"${v:,.0f}"


# --------------------------------------------------------------------------- header
st.markdown(
    '<div class="hero">'
    '<p class="kicker">Runtime agent governance, shown on a 3-agent AML investigation chain</p>'
    '<h1>Every handoff was authorized. The outcome was still wrong.</h1>'
    '<p>Three AI agents pass one alert down a chain: one reads the records, the next builds the case, '
    'the last files the decision. When the first agent misreads a value, that error flows through the '
    'other two untouched \u2014 each trusting the agent before it. No single handoff breaks any permission. '
    'This review shows why a security check clears all three while a governance check catches the failure '
    'and names the agent that caused it.</p>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="explain"><b>Two different questions:</b> the <b>security lane</b> asks '
    '\u201cwas each agent allowed to do its step?\u201d and checks every hop in isolation. The '
    '<b>governance lane</b> asks \u201cis the chain\u2019s final decision correct against the original '
    'records?\u201d and, when it is not, walks the chain back to the agent that introduced the error.</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.markdown('<div class="control-box"><div class="control-label">Session size</div><div class="control-help">More alerts make the run longer.</div></div>', unsafe_allow_html=True)
    alert_count = st.slider("Alerts", 8, 80, 40, label_visibility="collapsed")
with c2:
    st.markdown('<div class="control-box"><div class="control-label">Replay seed</div><div class="control-help">Same seed reproduces the run.</div></div>', unsafe_allow_html=True)
    seed = int(st.number_input("Seed", min_value=1, value=7, step=1, label_visibility="collapsed"))
with c3:
    st.markdown('<div class="control-box"><div class="control-label">Full session</div><div class="control-help">Synthetic records only. No LLM.</div></div>', unsafe_allow_html=True)
    run = st.button("Run the full session", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- stage 1: narrated chain
session = run_chain_session(alert_count, seed)
star = next((o for o in session if o.is_star_case), None)

if star is not None:
    alert = star.alert
    truth = recompute_structuring(alert)
    sec = SecurityLane().check(star.chain)
    gov = ChainGovernance().check(alert, star.chain)
    sec_by_step = {c.step_index: c for c in sec}

    st.markdown('<p class="stage-tag">Walkthrough \u00b7 one alert down the chain</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="stage-title">One case, three agents, start to finish</h2>', unsafe_allow_html=True)
    st.markdown('<p class="stage-help">Read it left to right. The first agent reads the records; the other two trust what they are handed. Watch where the value goes wrong and how far it travels before anything catches it.</p>', unsafe_allow_html=True)

    # Scene 1: the alert + real records
    rows = "".join(
        f"<tr><td>{t.id}</td><td>{t.timestamp:%b %d}</td><td>{t.type.replace('_',' ')}</td>"
        f"<td class='num'>{money(t.amount)}</td></tr>" for t in alert.transactions
    )
    st.markdown(
        f'<div class="scene"><h4>1 &nbsp; The alert and its actual records</h4>'
        f'<table class="facts"><thead><tr><th>Transaction</th><th>Date</th><th>Type</th><th>Amount</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
        f'<p style="margin:.7rem 0 0;font-size:.9rem;color:var(--muted)">Recomputed from these records: '
        f'{truth.qualifying_count} sub-threshold cash deposits totalling <b>{money(truth.aggregate)}</b> '
        f'\u2014 {"a structuring pattern that should be filed (FILE_SAR)." if truth.is_structuring else "not a structuring pattern."}</p></div>',
        unsafe_allow_html=True,
    )

    # Scene 2: the chain, hop by hop, with security verdicts and origin/propagator tags
    def hop_html(stp) -> str:
        role = {1: "reads the records", 2: "trusts step 1, builds case", 3: "trusts step 2, files"}[stp.step_index]
        is_origin = gov.origin_step == stp.step_index and not gov.passed
        is_prop = stp.agent in gov.propagators and not gov.passed
        klass = "hop origin" if is_origin else ("hop prop" if is_prop else "hop")
        sc = sec_by_step[stp.step_index]
        sec_cls = "pass" if sc.passed else "flag"
        recv = "" if stp.received_from is None else f'<div class="val" style="color:var(--muted)">received: {[round(x) for x in (stp.received_amounts or ())]}</div>'
        tag = ""
        if is_origin:
            tag = '<div class="tag origin">error originated here</div>'
        elif is_prop:
            tag = '<div class="tag prop">propagated the error</div>'
        return (
            f'<div class="{klass}"><p class="who">{stp.agent}</p><p class="role">step {stp.step_index} \u00b7 {role}</p>'
            f'<div class="val">used: {[round(x) for x in stp.used_amounts]}</div>'
            f'{recv}'
            f'<div class="val">decision: <b>{stp.disposition}</b></div>'
            f'<div class="sec {sec_cls}">security: {"PASS" if sc.passed else "FLAG"}</div>'
            f'{tag}</div>'
        )

    st.markdown(
        f'<div class="scene"><h4>2 &nbsp; The three agents, and what each security check sees</h4>'
        f'<div class="chain">{hop_html(star.chain[0])}<div class="arrow">\u2192</div>'
        f'{hop_html(star.chain[1])}<div class="arrow">\u2192</div>{hop_html(star.chain[2])}</div>'
        f'<p style="margin:.8rem 0 0;font-size:.88rem;color:var(--muted)">Every hop passes its own security check: each agent used an action it was permitted to use. Security is intact across the whole chain.</p></div>',
        unsafe_allow_html=True,
    )

    # Scene 3: the two lanes
    left, right = st.columns(2)
    with left:
        st.markdown(
            '<div class="lane miss"><p class="q">Security lane \u2014 was each agent allowed to act?</p>'
            '<p class="verdict">PASS \u2014 all three hops authorized</p>'
            '<p class="body">Every agent stayed within its permitted actions. No breach at any handoff. '
            'A permission-only check clears the entire chain.</p></div>',
            unsafe_allow_html=True,
        )
    with right:
        if gov.passed:
            st.markdown(
                f'<div class="lane miss"><p class="q">Governance lane \u2014 is the chain\u2019s decision correct?</p>'
                f'<p class="verdict">PASS \u2014 outcome matches records</p><p class="body">{gov.reason}</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="lane catch"><p class="q">Governance lane \u2014 is the chain\u2019s decision correct?</p>'
                f'<p class="verdict">FLAG \u2014 {gov.category}</p><p class="body">{gov.reason}</p></div>',
                unsafe_allow_html=True,
            )

    if not gov.passed:
        st.markdown(
            f'<div class="takeaway"><b>Why this needs more than two agents:</b> the agent where the error '
            f'<i>surfaces</i> ({star.chain[-1].agent}, which filed the wrong decision) is not the agent that '
            f'<i>caused</i> it ({gov.origin_agent}, step {gov.origin_step}). With a single handoff there is nothing '
            f'to trace; only across a chain can an error distance itself from its origin \u2014 and only a governance '
            f'view that recomputes from the original records and walks the chain back can pin the blame correctly.</div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- stage 2: batch
if "reviews" not in st.session_state:
    st.session_state.reviews = []

if run:
    progress = st.progress(0, text="Starting the session")
    monitor = ChainMonitor(context_window=8, checkpoint_interval=5)
    reviews = []
    for number, outcome in enumerate(session, start=1):
        reviews.append(monitor.evaluate(outcome))
        progress.progress(number / len(session), text=f"Agents working alert {number} of {len(session)}")
        time.sleep(0.02)
    st.session_state.reviews = reviews
    st.session_state.metrics = monitor.metrics
    progress.empty()

reviews = st.session_state.reviews
if not reviews:
    st.markdown('<p class="stage-tag">Full session</p>', unsafe_allow_html=True)
    st.info("Click \u201cRun the full session\u201d to run every alert through the 3-agent chain and see how often the chain is authorized end-to-end but reaches a wrong decision.")
    st.stop()

m = st.session_state.metrics
gov_flags = sum(not r.governance.passed for r in reviews)
sec_flags = sum(not all(c.passed for c in r.security) for r in reviews)
intact_wrong = sum(all(c.passed for c in r.security) and not r.governance.passed for r in reviews)

st.markdown('<p class="stage-tag">Full session \u00b7 the same chain across a long run</p>', unsafe_allow_html=True)
st.markdown('<h2 class="stage-title">Security intact, governance failed</h2>', unsafe_allow_html=True)
st.markdown('<p class="stage-help">Each alert runs through all three agents. The last figure is the one that matters: chains where every security check passed, yet the end-to-end decision was wrong.</p>', unsafe_allow_html=True)
st.markdown(
    f'<div class="score">'
    f'<div class="metric"><strong>{len(reviews)}</strong><small>Chains reviewed</small></div>'
    f'<div class="metric"><strong>{sec_flags}</strong><small>Security lane flags</small></div>'
    f'<div class="metric"><strong>{gov_flags}</strong><small>Governance lane flags</small></div>'
    f'<div class="metric hot"><strong>{intact_wrong}</strong><small>Security intact but wrong</small></div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown('<p class="stage-tag" style="margin-top:1.4rem">Session state \u00b7 carried across the run</p>', unsafe_allow_html=True)
h1, h2, h3 = st.columns(3)
with h1:
    st.metric("Turn", f"{m.turns} / {len(reviews)}")
    st.caption(f"Checkpoints: {', '.join(map(str, m.checkpoints)) or 'none'}")
with h2:
    st.metric("Context pressure", f"{m.context_pressure:.1f}x")
    st.caption(f"Retaining {m.retained_context} of {m.context_window} recent alerts")
with h3:
    st.metric("Security-intact-but-wrong", m.security_intact_but_wrong)
    st.caption("Chains no permission check would have caught")

st.markdown('<p class="stage-tag" style="margin-top:1.4rem">Chain log</p>', unsafe_allow_html=True)
filter_choice = st.radio("Show", ["All chains", "Governance flags", "The narrated case"], horizontal=True, label_visibility="collapsed")
visible = []
for r in reviews:
    if filter_choice == "Governance flags" and r.governance.passed:
        continue
    if filter_choice == "The narrated case" and not r.outcome.is_star_case:
        continue
    visible.append(r)

for r in visible:
    g = r.governance
    starmark = "\u2605 " if r.outcome.is_star_case else ""
    title = f"{starmark}{r.outcome.alert.alert_id} \u00b7 final decision: {g.final_disposition} \u00b7 {'governance FLAG' if not g.passed else 'ok'}"
    with st.expander(title, expanded=r.outcome.is_star_case):
        all_sec = all(c.passed for c in r.security)
        left, right = st.columns(2)
        with left:
            st.markdown(
                f'<div class="result"><span class="status {"pass" if all_sec else "flag"}">{"PASS" if all_sec else "FLAG"}</span>'
                f'<h3>Security lane</h3><p>{"All three hops authorized." if all_sec else "A hop used an unpermitted action."}</p></div>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                f'<div class="result{"" if g.passed else " flag"}"><span class="status {"pass" if g.passed else "flag"}">{"PASS" if g.passed else "FLAG"}</span>'
                f'<h3>Governance lane</h3><p>{g.reason}</p></div>',
                unsafe_allow_html=True,
            )
        if not g.passed:
            st.caption(f"Origin: {g.origin_agent} (step {g.origin_step}) \u00b7 propagators: {', '.join(g.propagators) or 'none'}")

st.caption("Synthetic demonstration of runtime agent governance across a delegation chain. Not a real AML detector, SAR system, or compliance decision.")