"""Streamlit demo: AML pipeline with AGT gates and NeMo content monitoring.

The intake agent fans one alert out to three parallel branches
(customer-screening, transaction-analysis, kyc-entity); the decision agent merges
them. The demo shows both a transaction misread masked by clean siblings and a
screening hit omitted from the disposition agent's stale case-state version. AGT
checks authorization before actions run, NeMo monitors output content during
agent work, the behavioral resolver checks correctness and consistency, and a
separate AGT Action Governance policy produces the final decision.

Run:  streamlit run app.py
"""

from __future__ import annotations

import html
from types import SimpleNamespace

import streamlit as st

from agent_sim import (
    DIAMOND_NODE_ID,
    DISPOSITION_AGENT,
    KYC_AGENT,
    SANCTIONS_AGENT,
    TRANSACTION_AGENT,
    TRIAGE_AGENT,
    run_diamond_session,
)
from agt_resolver import GraphGovernor, POLICY_VERSION, attribute_graph
from domain import (
    CTR_THRESHOLD,
    STRUCTURING_AGGREGATE,
    STRUCTURING_MIN_DEPOSITS,
    STRUCTURING_WINDOW_DAYS,
    recompute_structuring,
)

st.set_page_config(page_title="AGT Graph Governance", page_icon="\u25c6", layout="wide")

st.markdown("""
<style>
:root { --ink:#17212b; --muted:#667580; --line:#d7e0e5; --paper:#f5f7f6; --green:#176b45; --green-bg:#e7f5ed; --red:#b12632; --red-bg:#fff0f1; --gold:#9a6819; --amber-bg:#fbf3e2; --blue:#1f5c8a; --blue-bg:#eaf2f9; }
html, body, [class*="css"] { font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,sans-serif; }
.stApp { background:var(--paper); color:var(--ink); } .block-container { max-width:1180px; padding:2rem 1.5rem 4rem; }
.agt-bar { display:flex; align-items:center; gap:.8rem; flex-wrap:wrap; padding:.6rem .9rem; border:1px solid var(--line); background:#fff; margin-bottom:.8rem; font-size:.86rem; }
.pill { display:inline-block; padding:.22rem .55rem; font:700 .76rem ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.02em; }
.pill.live { color:var(--green); background:var(--green-bg); border:1px solid #b7dcc6; } .pill.fallback { color:var(--gold); background:var(--amber-bg); border:1px solid #ecd9ad; }
.agt-bar code, .mono { font:.8rem ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted); }
.hero { padding:1.6rem 1.7rem; border:1px solid var(--line); background:#fff; }
.hero .kicker { color:var(--muted); font-size:.85rem; margin:0 0 .5rem; }
.hero h1 { margin:.1rem 0 .5rem; font-size:clamp(1.7rem,3.2vw,2.5rem); letter-spacing:-.03em; line-height:1.1; }
.hero p { max-width:820px; margin:0; color:var(--muted); font-size:1rem; line-height:1.55; }
.primer-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.7rem; margin:.8rem 0 1rem; }
.primer-card { border:1px solid var(--line); background:#fff; padding:1rem 1.1rem; }
.primer-card h4 { margin:0 0 .4rem; font-size:.96rem; }
.primer-card p { margin:0; color:var(--muted); font-size:.87rem; line-height:1.5; }
.primer-card code { font-size:.78rem; }
.rule-table td:first-child { width:24%; font-weight:700; }
.stage-tag { margin:2rem 0 .35rem; color:var(--muted); font-size:.9rem; }
.stage-title { margin:.1rem 0 .3rem; font-size:1.4rem; letter-spacing:-.02em; }
.stage-help { color:var(--muted); font-size:.92rem; margin:0 0 1rem; max-width:800px; line-height:1.5; }
.scene { border:1px solid var(--line); background:#fff; padding:1.2rem 1.3rem; margin-bottom:1rem; }
.scene h4 { margin:0 0 .7rem; font-size:1.02rem; }
.facts { width:100%; border-collapse:collapse; font-size:.9rem; }
.facts th, .facts td { text-align:left; padding:.42rem .6rem; border-bottom:1px solid var(--line); vertical-align:top; }
.facts th { color:var(--muted); font-weight:600; } .facts td.num { text-align:right; font-variant-numeric:tabular-nums; }
.facts tr.bad td { background:var(--red-bg); } .facts td.ok { color:var(--green); font-weight:700; } .facts td.no { color:var(--red); font-weight:700; }
/* pipeline */
.pipe { display:grid; grid-template-columns:repeat(4,1fr); gap:.5rem; margin:.2rem 0 0; }
.pipe .st { border:1px solid var(--line); background:#fff; padding:.75rem .8rem; font-size:.84rem; line-height:1.4; position:relative; }
.pipe .st b { display:block; font-size:.88rem; margin-bottom:.2rem; } .pipe .st span { color:var(--muted); }
.pipe .st.ours { border:2px solid var(--blue); background:var(--blue-bg); } .pipe .st.ours b { color:var(--blue); }
.pipe .who { display:inline-block; margin-top:.45rem; padding:.1rem .35rem; font:700 .68rem ui-monospace,monospace; }
.pipe .who.agt { color:var(--muted); background:#eef1f3; } .pipe .who.us { color:#fff; background:var(--blue); }
.flowchart { display:grid; justify-items:center; gap:.24rem; margin:1rem auto 1.5rem; max-width:1020px; }
.flow-node { width:min(72%,720px); padding:.48rem .7rem; border:1px solid var(--line); background:#fff; text-align:center; line-height:1.25; }
.flow-node b { display:block; font-size:.84rem; } .flow-node small { color:var(--muted); font-size:.72rem; }
.flow-node.policy { border:2px solid #67a77d; background:#eef7ef; }
.flow-node.alert { border:2px solid #79a7cf; background:#eef5fb; }
.flow-node.case { border:2px solid #9280b3; background:#f4f0fa; }
.flow-node.shared { border:2px solid #c09b50; background:#fbf5e8; }
.flow-node.signal { border:2px solid #9280b3; background:#f4f0fa; }
.flow-node.gate { border:2px solid #c66b70; background:#fff0f1; width:min(88%,880px); }
.flow-node.conflict { border:2px solid #c09b50; background:#fbf5e8; }
.flow-node.stale { border:2px solid #c66b70; background:#fff0f1; }
.state-conflict { display:grid; grid-template-columns:1fr auto 1fr; gap:.65rem; align-items:center; width:min(88%,880px); }
.state-version { padding:.48rem .65rem; border:1px solid var(--line); background:#fff; text-align:center; line-height:1.3; }
.state-version b { display:block; font-size:.8rem; } .state-version small { color:var(--muted); font-size:.72rem; }
.state-version.current { border:2px solid #c09b50; background:#fbf5e8; }
.state-version.stale { border:2px solid #c66b70; background:#fff0f1; }
.state-arrow { color:var(--red); font-size:.75rem; font-weight:700; text-align:center; }
.flow-arrow { color:var(--muted); font-size:1.05rem; font-weight:600; line-height:1.1; margin:.15rem 0; }
.flow-branches,.flow-outcomes,.flow-downstream { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.5rem; width:min(92%,920px); position:relative; margin-top:34px; overflow:visible; }
/* BUS: one horizontal line spanning from the centre of box 1 to the centre of box 3. */
.flow-outcomes::before,.flow-downstream::before { content:""; position:absolute; top:-20px; left:16.667%; right:16.667%; height:2px; background:var(--muted); z-index:0; }
/* FEEDER: vertical line from the spine arrow DOWN THROUGH the bus (overlaps it, no gap). */
.flow-outcomes::after,.flow-downstream::after { content:""; position:absolute; top:-34px; left:calc(50% - 1px); width:2px; height:16px; background:var(--muted); z-index:0; }
/* DROPS: from ABOVE the bus (-22px) down INTO each box top (+2px) so both ends overlap, no gaps. */
.flow-outcomes > *::before,.flow-downstream > *::before { content:""; position:absolute; top:-22px; left:calc(50% - 1px); width:2px; height:24px; background:var(--muted); z-index:0; }
.flow-outcomes > *,.flow-downstream > * { position:relative; }
.flow-outcomes > * > *,.flow-downstream > * > * { position:relative; z-index:1; }
/* CONTINUE connector: an L-path from under the CONTINUE box (right third) across to the centre, then down into Disposition. */
.continue-connector { position:relative; width:min(92%,920px); height:44px; margin:2px auto 0; }
/* label */
.continue-connector span { position:absolute; top:12px; left:50%; transform:translateX(-50%); color:var(--muted); font-size:.92rem; font-weight:600; white-space:nowrap; background:var(--paper); padding:0 .4rem; z-index:2; }
/* 1: vertical drop straight down from under the CONTINUE box */
.continue-connector::before { content:""; position:absolute; top:0; right:16.667%; width:2px; height:22px; background:var(--muted); }
/* 2: horizontal run leftwards to the centre at y=22, PLUS the centre drop down into Disposition via box-shadow at the left end */
.continue-connector::after { content:""; position:absolute; top:22px; left:50%; width:calc(33.333% + 2px); height:2px; background:var(--muted); }
/* 3: the centre vertical drop into Disposition */
.continue-drop { content:""; position:absolute; top:22px; left:calc(50% - 1px); width:2px; height:24px; background:var(--muted); }
.flow-reasoning { display:grid; grid-template-columns:minmax(0,1fr) 220px; gap:.55rem; width:min(82%,820px); align-items:center; }
.flow-outcomes .block { border-color:#c66b70; background:#fff0f1; }
.flow-outcomes .human { border-color:#c09b50; background:#fbf5e8; }
.flow-outcomes .continue { border-color:#67a77d; background:#eef7ef; }
.flow-outcomes .active { box-shadow:0 0 0 2px currentColor; font-weight:700; }
/* lanes */
.lane { border:1px solid var(--line); background:#fff; padding:1.1rem 1.15rem; height:100%; }
.lane.pass { border-color:#b7dcc6; background:#fbfdfb; } .lane.deny { border:2px solid var(--red); background:var(--red-bg); } .lane.halt { border-color:#e4aeb2; background:#fff8f8; }
.lane.review { border:2px solid #c09b50; background:#fbf5e8; } .lane.review .verdict { color:var(--gold); }
.lane.neutral { border-color:var(--line); background:#f5f7f6; } .lane.neutral .verdict { color:var(--muted); }
.lane .src { font:700 .72rem ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted); margin:0 0 .35rem; letter-spacing:.02em; }
.lane .q { color:var(--muted); font-size:.86rem; margin:0 0 .5rem; }
.lane .verdict { font-weight:800; font-size:1.25rem; margin:0 0 .5rem; letter-spacing:-.01em; }
.lane.pass .verdict { color:var(--green); } .lane.deny .verdict, .lane.halt .verdict { color:var(--red); }
.lane .body { color:var(--ink); font-size:.9rem; line-height:1.5; margin:0; }
.lane ul { margin:.3rem 0 0; padding-left:1.1rem; font-size:.84rem; } .lane li { margin:.1rem 0; }
.takeaway { margin:1rem 0 0; padding:1rem 1.2rem; background:var(--amber-bg); border:1px solid #ecd9ad; font-size:.96rem; line-height:1.55; }
.control-box { padding:1rem 1.2rem; border:1px solid var(--line); background:#fff; } .control-label { margin-bottom:.3rem; font-weight:700; } .control-help { color:var(--muted); font-size:.85rem; }
.score { display:grid; grid-template-columns:repeat(6,1fr); gap:.75rem; margin:1.2rem 0; } .metric { min-height:88px; padding:1rem; border:1px solid var(--line); background:#fff; } .metric strong { display:block; font-size:1.9rem; } .metric small { color:var(--muted); } .metric.hot { border-color:#e4aeb2; background:#fff8f8; } .metric.hot strong { color:var(--red); }
.diamond { width:100%; height:auto; display:block; }
.status.neutral { color:var(--gold); background:var(--amber-bg); }
@media(max-width:820px){ .primer-grid,.pipe{grid-template-columns:1fr} .flow-node,.flow-node.gate,.flow-branches,.flow-outcomes,.flow-downstream,.flow-reasoning,.state-conflict{width:100%} .flow-branches,.flow-outcomes,.flow-downstream,.flow-reasoning,.state-conflict{grid-template-columns:1fr} .score{grid-template-columns:repeat(2,1fr)} .block-container{padding:1rem .8rem 3rem} }
</style>
""", unsafe_allow_html=True)


def money(v: float) -> str:
    return f"${v:,.0f}"


def esc(s: object) -> str:
    return html.escape(str(s))


governor = GraphGovernor()

# --------------------------------------------------------------------------- AGT status bar
if governor.agt_live:
    status_pill = '<span class="pill live">AGT PolicyEngine: LIVE</span>'
    status_text = (f'<code>{esc(governor.engine_class)}</code> &middot; behavioral resolver supplies the correctness signal; '
                   'a separate Action Governance policy produces the final PolicyDecision.')
else:
    status_pill = '<span class="pill fallback">AGT PolicyEngine: fallback</span>'
    status_text = ('AGT policy fallback active; NeMo availability is reported separately on each run. '
                   'Install <code>agent-governance-toolkit-core</code> to execute the policies inside AGT.')
st.markdown(f'<div class="agt-bar">{status_pill}<span>{status_text}</span></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="hero">'
    '<p class="kicker">Runtime agent governance on a fan-out / fan-in AML investigation graph</p>'
    '<h1>Conflicting agent views need reconciliation before clearance.</h1>'
    '<p>One alert fans out to three parallel agents and fans back in to a fourth that merges their '
    'results. In the conflict case, screening flags a probable watchlist match, but disposition reads an '
    'older case-state version and clears the customer without reconciling the flag. AGT authorizes actions; '
    'the behavioral resolver checks consistency and correctness; NeMo checks output content; final Action '
    'Governance blocks or routes the unresolved case to human review.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- primer
st.markdown('<p class="stage-tag">Before you run it</p>', unsafe_allow_html=True)
st.markdown('<h2 class="stage-title">What the AML terms and flags mean in this demo</h2>', unsafe_allow_html=True)
st.markdown(
    f'<div class="primer-grid">'
    f'<div class="primer-card"><h4>Potential structuring</h4><p>Structuring generally means breaking cash activity into smaller '
    f'transactions to avoid reporting or scrutiny. This synthetic rule flags a pattern only when there are at least '
    f'<b>{STRUCTURING_MIN_DEPOSITS} cash deposits</b>, each below <b>{money(CTR_THRESHOLD)}</b>, totalling at least '
    f'<b>{money(STRUCTURING_AGGREGATE)}</b>, with the '
    f'qualifying deposits contained within <b>{STRUCTURING_WINDOW_DAYS} days</b>.</p></div>'
    f'<div class="primer-card"><h4>\u201cNot structuring\u201d</h4><p>This means only that the records did <b>not meet the demo rule</b>. '
    f'It is not proof that activity is legitimate, and it does not override sanctions, KYC, or other monitoring signals.</p></div>'
    f'<div class="primer-card"><h4>Flag versus disposition</h4><p>A <b>flag</b> is a signal that needs policy handling; it is not an '
    f'accusation or final legal conclusion. <code>CLEAR</code>, <code>ESCALATE</code>, and <code>FILE_SAR</code> are simulated '
    f'dispositions. No real SAR is filed by this application.</p></div>'
    f'<div class="primer-card"><h4>Why there are several flag types</h4><p>Authorization, output safety, and AML correctness answer '
    f'different questions. A permitted action can still contain unsafe text or be factually wrong, so the signals stay separate '
    f'until the final AGT policy evaluates them.</p></div>'
    f'</div>',
    unsafe_allow_html=True,
)

with st.expander("How an alert is checked and how the final outcome is chosen", expanded=True):
    st.markdown(
        """
1. **AGT authorization runs first.** Every handoff must follow a declared graph edge, and the receiving agent's action must be in its delegated action set. A failure is denied before that recipient runs.
2. **Authorized agents produce findings.** Screening checks the demo watchlist, transaction analysis applies the synthetic structuring rule, and KYC checks whether the required customer profile fields are present.
3. **NeMo monitors output content.** It runs only on authorized outputs and applies the configured `self_check_output` rail. It can flag unsafe or prompt-injection-like content, but it does not decide whether the AML finding is correct and it never grants authority.
4. **The behavioral resolver recomputes from records.** It compares each reported branch finding with source data, verifies that the decision agent consumed current parent versions, and checks that the merged disposition follows those inputs.
5. **AGT Action Governance makes the final decision.** It consumes authorization, resolver correctness, and NeMo status using the priority rules below. The first applicable high-priority rule determines the result.
        """
    )
    st.markdown(
        """
<table class="facts rule-table">
<thead><tr><th>Signal seen by final policy</th><th>Result</th><th>Why</th></tr></thead>
<tbody>
<tr><td>Unauthorized action</td><td class="no">DENY / BLOCK</td><td>The agent was not permitted to perform the action.</td></tr>
<tr><td>Unreconciled stale or conflicting state</td><td><b style="color:var(--gold)">HUMAN REVIEW</b></td><td>A person must reconcile the competing findings before a clearance can commit.</td></tr>
<tr><td>Other record-relative correctness failure</td><td class="no">DENY / BLOCK</td><td>The reported or merged result conflicts with the source records.</td></tr>
<tr><td>NeMo content flag only</td><td><b style="color:var(--gold)">HUMAN REVIEW</b></td><td>The action was authorized and may be correct, but its content needs review.</td></tr>
<tr><td>Authorization and correctness pass; NeMo passes</td><td class="ok">ALLOW / CONTINUE</td><td>All configured signals permit the action.</td></tr>
<tr><td>Authorization and correctness pass; NeMo unavailable</td><td class="ok">ALLOW / CONTINUE</td><td>This demo allows the action but reports that content monitoring did not run.</td></tr>
</tbody></table>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Demo thresholds and dispositions are illustrative controls, not legal advice or a production SAR determination.")

# --------------------------------------------------------------------------- controls
c1, c2, c3 = st.columns([1.2, 1, 1.2])
with c1:
    st.markdown('<div class="control-box"><div class="control-label">Batch size</div><div class="control-help">Alerts to check (star case at #7).</div></div>', unsafe_allow_html=True)
    count = st.slider("Alerts", 8, 40, 8, label_visibility="collapsed")
with c2:
    st.markdown('<div class="control-box"><div class="control-label">Replay seed</div><div class="control-help">Same seed reproduces the run.</div></div>', unsafe_allow_html=True)
    seed = int(st.number_input("Seed", min_value=1, value=7, step=1, label_visibility="collapsed"))
with c3:
    st.markdown('<div class="control-box"><div class="control-label">Run the pipeline</div><div class="control-help">Calls NeMo for each authorized agent output.</div></div>', unsafe_allow_html=True)
    run_pipeline = st.button("Evaluate graph session", type="primary", use_container_width=True)

review_key = ("nemo-rail-decision-v2", count, seed)
if st.session_state.get("diamond_review_key") != review_key:
    st.session_state.pop("diamond_reviews", None)

if run_pipeline:
    session = run_diamond_session(count, seed)
    progress = st.progress(0, text="Running deterministic gates, NeMo checks, and Action Governance")
    reviews = []
    for index, outcome in enumerate(session, start=1):
        reviews.append(governor.review(outcome))
        progress.progress(index / len(session), text=f"Evaluated alert {index} of {len(session)}")
    st.session_state.diamond_review_key = review_key
    st.session_state.diamond_reviews = reviews
    progress.empty()

reviews = st.session_state.get("diamond_reviews", [])
has_live_review = bool(reviews)
if has_live_review:
    case_options = {
        "Transaction misread - incorrect CLEAR": next(r for r in reviews if r.outcome.is_star_case),
        "Screening conflict - stale case state": next(r for r in reviews if r.outcome.is_conflict_case),
    }
    selected_case = st.radio("Case to inspect", list(case_options), horizontal=True)
    star = case_options[selected_case]
else:
    preview_cases = run_diamond_session(count, seed)
    case_options = {
        "Transaction misread - incorrect CLEAR": next(item for item in preview_cases if item.is_star_case),
        "Screening conflict - stale case state": next(item for item in preview_cases if item.is_conflict_case),
    }
    selected_case = st.radio("Case to inspect", list(case_options), horizontal=True)
    preview_outcome = case_options[selected_case]
    preview_handoffs = tuple(
        SimpleNamespace(
            sender=sender,
            recipient=recipient,
            passed=True,
            decision="PENDING",
            rule="not evaluated",
            reason="Run the pipeline to evaluate this handoff.",
        )
        for sender, recipient in preview_outcome.edges
    )
    star = SimpleNamespace(
        outcome=preview_outcome,
        handoffs=preview_handoffs,
        attribution=attribute_graph(preview_outcome.alert, preview_outcome.steps),
        content_checks=tuple(
            {"agent": step.agent, "passed": True, "rail": "pending", "reason": "NeMo runs when you evaluate the session."}
            for step in preview_outcome.steps
        ),
        nemo_live=False,
        policy_decision={},
        authority_decision={"decision": "pending"},
        status="pending",
    )
    star.is_conflict_case = preview_outcome.is_conflict_case
alert = star.outcome.alert
truth = recompute_structuring(alert)
steps = {s.agent: s for s in star.outcome.steps}
checks = {c.agent: c for c in star.attribution.branch_checks}

# --------------------------------------------------------------------------- scene 1: records
st.markdown('<p class="stage-tag">Walkthrough &middot; one alert through the diamond</p>', unsafe_allow_html=True)
st.markdown(f'<h2 class="stage-title">{esc(alert.alert_id)} &middot; {esc(alert.customer.name)}</h2>', unsafe_allow_html=True)
rows = "".join(
    f"<tr><td>{t.id}</td><td>{t.timestamp:%b %d}</td><td>{t.type.replace('_', ' ')}</td><td class='num'>{money(t.amount)}</td></tr>"
    for t in alert.transactions
)
st.markdown(
    f'<div class="scene"><h4>1 &nbsp; The actual records</h4>'
    f'<table class="facts"><thead><tr><th>Transaction</th><th>Date</th><th>Type</th><th>Amount</th></tr></thead><tbody>{rows}</tbody></table>'
    f'<p style="margin:.7rem 0 0;font-size:.9rem;color:var(--muted)">Recomputed from these records: {truth.qualifying_count} '
    f'sub-threshold cash deposits totalling <b>{money(truth.aggregate)}</b> '
    f'{"within" if truth.within_window else "outside"} the {STRUCTURING_WINDOW_DAYS}-day window &mdash; '
    f'{"meets this demo\u2019s potential-structuring rule and produces a FILE_SAR referral." if truth.is_structuring else "does not meet this demo\u2019s structuring rule (not-structuring)."}</p></div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- scene 2: the diamond
def diamond_svg(review) -> str:
    s = {x.agent: x for x in review.outcome.steps}
    ck = {c.agent: c for c in review.attribution.branch_checks}
    content = {c["agent"]: c for c in review.content_checks}
    hop = {(h.sender, h.recipient): h for h in review.handoffs}
    W, H, BW, BH = 900, 420, 212, 112
    pos = {TRIAGE_AGENT: (20, 154), SANCTIONS_AGENT: (350, 12), TRANSACTION_AGENT: (350, 154),
           KYC_AGENT: (350, 296), DISPOSITION_AGENT: (680, 154)}
    role = {TRIAGE_AGENT: "fans out the alert", SANCTIONS_AGENT: "screens watchlist", TRANSACTION_AGENT: "reads the deposits",
            KYC_AGENT: "verifies identity", DISPOSITION_AGENT: "merges 3 branches"}

    def node(agent: str) -> str:
        x, y = pos[agent]
        st_ = s[agent]
        c = ck.get(agent)
        nemo = content.get(agent, {})
        rail = nemo.get("rail", "skipped")
        content_passed = bool(nemo.get("passed", True))
        nemo_label = {
            "unavailable": "NeMo unavailable",
            "skipped": "NeMo skipped",
            "pending": "NeMo pending",
        }.get(str(rail), "NeMo pass" if content_passed else "NeMo flag")
        if agent == TRANSACTION_AGENT and c and not c.matches:
            fill, stroke, sw, tone = "#fff0f1", "#b12632", 2.5, "#b12632"
            line2 = f"reports: {st_.finding}"
            line3 = f"used {money(st_.used_aggregate)}; records {money(truth.aggregate)}"
        elif agent == DISPOSITION_AGENT:
            stale_conflict = review.outcome.is_conflict_case
            wrong = c is not None and not c.matches
            if stale_conflict:
                fill, stroke, sw, tone = "#fff0f1", "#b12632", 2.5, "#b12632"
            else:
                fill, stroke, sw, tone = ("#fbf3e2", "#9a6819", 2, "#9a6819") if wrong else ("#ffffff", "#d7e0e5", 1, "#17212b")
            line2 = f"merged: {st_.disposition}"
            line3 = (
                "used stale screening view"
                if stale_conflict else
                "confident: 3/3 branches clean" if st_.finding.endswith("(3/3 branches clean)") else st_.finding
            )
        elif agent == TRIAGE_AGENT:
            fill, stroke, sw, tone = "#ffffff", "#d7e0e5", 1, "#17212b"
            line2, line3 = "routes to B, C, D", ""
        else:
            if not content_passed:
                fill, stroke, sw, tone = "#fff0f1", "#b12632", 2, "#b12632"
            elif rail == "unavailable":
                fill, stroke, sw, tone = "#fbf3e2", "#c09b50", 1.5, "#765016"
            else:
                fill, stroke, sw, tone = "#ffffff", "#b7dcc6", 1.5, "#176b45"
            line2, line3 = f"reports: {st_.finding}", "content check below"
        if c is not None and not c.matches:
            correctness_label = "records: diverged"
        elif c is not None:
            correctness_label = "records: match"
        else:
            correctness_label = "records: not applicable"
        if rail == "unavailable":
            nemo_tone = "#765016"
        elif rail == "pending":
            nemo_tone = "#9a6819"
        elif not content_passed:
            nemo_tone = "#b12632"
        else:
            nemo_tone = "#176b45"
        return (f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
                f'<text x="{x + 12}" y="{y + 24}" font-size="14" font-weight="700" fill="#17212b">{st_.node_id} &#183; {esc(agent)}</text>'
                f'<text x="{x + 12}" y="{y + 42}" font-size="11.5" fill="#667580">{esc(role[agent])}</text>'
                f'<text x="{x + 12}" y="{y + 62}" font-size="13" font-weight="700" fill="{tone}">{esc(line2)}</text>'
                f'<text x="{x + 12}" y="{y + 80}" font-size="10.5" fill="{tone}">{esc(line3)}</text>'
                f'<text x="{x + 12}" y="{y + 96}" font-size="10.5" fill="#667580">{correctness_label}</text>'
                f'<text x="{x + 12}" y="{y + 108}" font-size="9.5" font-weight="700" fill="{nemo_tone}">{esc(nemo_label)}</text>')

    def edge(a: str, b: str) -> str:
        ax, ay = pos[a]; bx, by = pos[b]
        x1, y1, x2, y2 = ax + BW, ay + BH / 2, bx, by + BH / 2
        h = hop[(a, b)]
        col = "#9a6819" if h.decision == "PENDING" else ("#176b45" if h.passed else "#b12632")
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        label = "policy pending" if h.decision == "PENDING" else ("handoff OK" if h.passed else "handoff denied")
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2 - 4}" y2="{y2}" stroke="{col}" stroke-width="2" marker-end="url(#ah)"/>'
                f'<rect x="{mx - 48}" y="{my - 10}" width="96" height="19" rx="3" fill="#e7f5ed" stroke="#b7dcc6"/>'
                f'<text x="{mx}" y="{my + 4}" font-size="10.5" font-weight="700" fill="{col}" text-anchor="middle">{label}</text>')

    edges = "".join(edge(a, b) for a, b in review.outcome.edges)
    nodes = "".join(node(a) for a in pos)
    return (f'<svg class="diamond" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif" role="img" aria-label="Diamond graph">'
            f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="#667580"/></marker></defs>{edges}{nodes}</svg>')


c_step = steps[TRANSACTION_AGENT]
is_conflict_case = star.outcome.is_conflict_case
conflict_state_html = ""
if is_conflict_case:
    screening_step = steps[SANCTIONS_AGENT]
    disposition_step = steps[DISPOSITION_AGENT]
    consumed_screening_finding = disposition_step.received_findings_by_parent[SANCTIONS_AGENT]
    consumed_screening_version = disposition_step.received_versions_by_parent[SANCTIONS_AGENT]
    conflict_state_html = (
        '<div class="state-conflict">'
        f'<div class="state-version current"><b>Screening finding v{screening_step.finding_version}</b>'
        f'<small>{esc(screening_step.finding.upper())} - probable watchlist match</small></div>'
        '<div class="state-arrow">NOT RECONCILED<br>&rarr;</div>'
        f'<div class="state-version stale"><b>Disposition consumed v{consumed_screening_version}</b>'
        f'<small>{esc(consumed_screening_finding.upper())} - stale case snapshot</small></div>'
        '</div>'
        '<div class="flow-node stale"><b>Disposition agent clears customer</b>'
        '<small>the newer screening flag was not incorporated or reconciled</small></div>'
    )
flow_nemo_flags = [check for check in star.content_checks if not check["passed"]]
flow_nemo_unavailable = any(check["rail"] == "unavailable" for check in star.content_checks)
flow_nemo_label = (
    "NOT RUN" if not has_live_review else
    "FLAG" if flow_nemo_flags else
    "UNAVAILABLE" if flow_nemo_unavailable else "PASS"
)
flow_action = str(star.policy_decision.get("action", "pending"))
flow_action_label = {
    "allow": "CONTINUE",
    "require_approval": "HUMAN REVIEW",
    "deny": "BLOCK",
    "pending": "PENDING EVALUATION",
}.get(flow_action, flow_action.upper())
flow_active_outcome = {
    "allow": "continue",
    "require_approval": "human",
    "deny": "block",
}.get(flow_action)
if is_conflict_case:
    flow_case_narrative = (
        f'<p style="margin:.6rem 0 0;font-size:.9rem;color:var(--muted)">'
        f'Screening found a probable watchlist match, but disposition consumed the earlier '
        f'<b>{esc(consumed_screening_finding)}</b> snapshot and produced <b>CLEAR</b> without reconciliation. '
        f'AGT Action Governance routes this unresolved conflict to human review before the clear can commit. '
        f'The RAG, typology, narrative, and audit stages shown here are the architecture path; '
        f'this demo executes the graph and governance decision.</p></div>'
    )
else:
    flow_case_narrative = (
        f'<p style="margin:.6rem 0 0;font-size:.9rem;color:var(--muted)">C read the deposits as '
        f'{", ".join(money(a) for a in c_step.used_amounts)} instead of '
        f'{", ".join(money(a) for a in truth.qualifying_amounts)}. B and D are right. '
        f'E merges the branch findings into <b>{esc(steps[DISPOSITION_AGENT].disposition)}</b>. '
        f'The RAG, typology, narrative, and audit stages shown here are the architecture path; '
        f'this demo executes the graph and governance decisions.</p></div>'
    )
st.markdown(
    f'<div class="scene"><h4>2 &nbsp; AML execution flow - top gate to final action</h4>'
    f'<div class="flowchart" role="img" aria-label="Live AML governance flow diagram">'
    f'<div class="flow-node policy"><b>AML Policy + AGT deterministic top gate</b>'
    f'<small>access, limits, and delegated action checks before execution</small></div>'
    f'<div class="flow-arrow">&#8595;</div>'
    f'<div class="flow-node alert"><b>Alert: {esc(alert.alert_id)}</b></div>'
    f'<div class="flow-arrow">&#8595;</div>'
    f'<div class="flow-node case"><b>Orchestrator / intake agent</b><small>routes case to evidence agents</small></div>'
    f'<div class="flow-arrow">&#8595; parallel evidence gathering - AGT checks each handoff</div>'
    f'{diamond_svg(star)}'
    f'{conflict_state_html}'
    f'<div class="flow-arrow">&#8595; fan in</div>'
    f'<div class="flow-node shared"><b>Shared Case State</b><small>branch findings and evidence references merged by the decision agent</small></div>'
    f'<div class="flow-arrow">&#8595;</div>'
    f'<div class="flow-node policy"><b>Evidence / RAG - completeness</b><small>architecture stage following the fan-in</small></div>'
    f'<div class="flow-arrow">&#8595;</div>'
    f'<div class="flow-node policy"><b>Typology / Rule - deterministic match</b></div>'
    f'<div class="flow-arrow">&#8595;</div>'
    f'<div class="flow-reasoning">'
    f'<div class="flow-node case"><b>Risk reasoning / merged finding</b>'
    f'<small>record-relative result: {"PASS" if star.attribution.passed else "FLAG"}</small></div>'
    f'<div class="flow-node signal"><b>NeMo content check: {flow_nemo_label}</b>'
    f'<small>checked {len(star.content_checks)} authorized agent outputs; {len(flow_nemo_flags)} flagged'
    f'{"; service unavailable" if flow_nemo_unavailable else ""}</small></div>'
    f'</div>'
    f'<div class="flow-arrow">&#8595; authorization + correctness + NeMo signal</div>'
    f'<div class="flow-node gate"><b>AGT Action Governance - {flow_action_label}</b>'
    f'<small>{esc(str(star.policy_decision.get("matched_rule") or ("evaluate to produce the final PolicyDecision" if not has_live_review else "default policy")))}</small></div>'
    f'<div class="flow-arrow">&#8595; branches to three possible actions</div>'
    f'<div class="flow-outcomes">'
    f'<div class="flow-node block{" active" if flow_active_outcome == "block" else ""}"><b>DENY / BLOCK</b><small>authorization or correctness failure</small></div>'
    f'<div class="flow-node human{" active" if flow_active_outcome == "human" else ""}"><b>REQUIRE APPROVAL / HUMAN REVIEW</b><small>content flag or unresolved conflict</small></div>'
    f'<div class="flow-node continue{" active" if flow_active_outcome == "continue" else ""}"><b>ALLOW / CONTINUE</b><small>policy allows the action</small></div>'
    f'</div>'
    f'<div class="continue-connector"><span>continue</span><div class="continue-drop"></div></div>'
    f'<div class="flow-node alert"><b>8. Disposition Agent</b><small>CLEAR / ESCALATE / refer for SAR</small></div>'
    f'<div class="flow-arrow">&#8595;</div>'
    f'<div class="flow-node alert"><b>9. SAR / Narrative Agent</b><small>claims linked to evidence IDs</small></div>'
    f'<div class="flow-arrow">&#8595;</div>'
    f'<div class="flow-node policy"><b>10. Audit / Evidence Agent</b><small>full reconstructable record</small></div>'
    f'</div>'
    f'<p style="margin:.2rem 0 .7rem;font-size:.84rem;color:var(--muted)">'
    f'{"Amber labels are pending; evaluate the session to run the top gate, NeMo, and final policy." if not has_live_review else "Green handoff labels show AGT authorization; branch nodes report content and correctness signals."} '
    f'NeMo checks content only. The behavioral resolver checks records, and AGT Action Governance decides.</p>'
    f'{flow_case_narrative}',
    unsafe_allow_html=True,
)

if not has_live_review:
    st.info("The diagram is a no-call preview. Evaluate the graph session to run AGT policies and NeMo content checks.")
    st.stop()

# --------------------------------------------------------------------------- scene 3: governance stages
st.markdown('<p class="stage-tag">Governance stages</p>', unsafe_allow_html=True)
st.markdown('<h2 class="stage-title">The signals converge at AGT Action Governance</h2>', unsafe_allow_html=True)

n_ok = sum(h.passed for h in star.handoffs)
hop_items = "".join(
    f'<li>{DIAMOND_NODE_ID[h.sender]} &rarr; {DIAMOND_NODE_ID[h.recipient]} &nbsp;<b style="color:var(--{"green" if h.passed else "red"})">'
    f'{esc(h.decision)}</b> <span class="mono">{esc(h.rule or "")}</span></li>' for h in star.handoffs
)
pd_ = star.policy_decision
ad = star.authority_decision
nemo_flags = [check for check in star.content_checks if not check["passed"]]
nemo_unavailable = [check for check in star.content_checks if check["rail"] == "unavailable"]
nemo_executed = [check for check in star.content_checks if check["rail"] not in {"unavailable", "skipped"}]
nemo_status = "FLAG" if nemo_flags else ("PASS" if star.nemo_live else "UNAVAILABLE")
nemo_css = "flag" if nemo_flags else ("pass" if star.nemo_live else "neutral")
nemo_lane = "deny" if nemo_flags else ("pass" if star.nemo_live else "neutral")
nemo_detail = "<br>".join(
    f"{esc(check['agent'])}: {esc(check['rail'])} - {esc(check['reason'])}"
    for check in star.content_checks
)
action_name = str(pd_.get("action", "deny"))
action_label = {
    "allow": "CONTINUE",
    "require_approval": "HUMAN REVIEW",
    "deny": "BLOCK",
}.get(action_name, action_name.upper())
action_lane = {"allow": "pass", "require_approval": "review", "deny": "deny"}.get(action_name, "deny")
status_word = {"halted": "BLOCKED", "committed": "CONTINUED", "escalated": "HUMAN REVIEW"}[star.status]
correctness_label = "PASS" if star.attribution.passed else "FLAG"

l1, l2, l3, l4 = st.columns([1, 1, 1.2, 1.2])
with l1:
    st.markdown(
        f'<div class="lane pass"><p class="src">TOP GATE - AGT DETERMINISTIC AUTHORIZATION</p>'
        f'<p class="q">May each agent perform its delegated action?</p>'
        f'<p class="verdict">{n_ok} / {len(star.handoffs)} AUTHORIZED</p>'
        f'<p class="body">Every recipient action passed its pre-execution handoff policy.</p>'
        f'<ul>{hop_items}</ul></div>',
        unsafe_allow_html=True,
    )
with l2:
    st.markdown(
        f'<div class="lane {nemo_lane}">'
        f'<p class="src">MIDDLE LAYER - NEMO CONTENT MONITOR</p>'
        f'<p class="q">Did agent output trigger the configured content rail?</p>'
        f'<p class="verdict">{nemo_status}</p>'
        f'<p class="body">Checked {len(nemo_executed)} of {len(star.content_checks)} outputs; '
        f'{len(nemo_flags)} flagged, {len(nemo_unavailable)} unavailable.</p>'
        f'<p class="body" style="margin-top:.5rem;font-size:.78rem">{nemo_detail}</p></div>',
        unsafe_allow_html=True,
    )
with l3:
    st.markdown(
        f'<div class="lane {"pass" if star.attribution.passed else "deny"}">'
        f'<p class="src">CORRECTNESS SIGNAL - BEHAVIORAL RESOLVER</p>'
        f'<p class="q">Does the merged result match the source records?</p>'
        f'<p class="verdict">{correctness_label}'
        f'{" - branch " + esc(star.attribution.origin_node) if star.attribution.origin_node else ""}</p>'
        f'<p class="body">{esc(star.attribution.reason)}</p>'
        f'<p class="body" style="margin-top:.5rem;font-size:.78rem">Resolver signal: '
        f'<code>{esc(str(ad.get("decision", "not returned")))}</code>; it does not make the final action decision.</p></div>',
        unsafe_allow_html=True,
    )
with l4:
    st.markdown(
        f'<div class="lane {action_lane}">'
        f'<p class="src">BOTTOM GATE - AGT ACTION GOVERNANCE</p>'
        f'<p class="q">Final AGT PolicyDecision for commit_disposition</p>'
        f'<p class="verdict">{action_label}</p>'
        f'<p class="body">{esc(str(pd_.get("reason", "")))}</p>'
        f'<p class="body" style="margin-top:.5rem;font-size:.78rem">Rule: '
        f'<code>{esc(str(pd_.get("matched_rule") or "default"))}</code> - outcome: {status_word}</p></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div class="takeaway"><b>Decision path:</b> all {len(star.handoffs)} AGT handoffs were authorized. '
    f'NeMo reported {nemo_status.lower()} for output content. The behavioral resolver independently '
    f'{"confirmed" if star.attribution.passed else "flagged"} record-relative correctness'
    f'{" at branch " + esc(star.attribution.origin_node) if star.attribution.origin_node else ""}. '
    f'AGT Action Governance consumed those signals and returned <b>{action_label}</b>. '
    f'NeMo informed the decision; AGT made it.</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- scene 4: branch attribution
st.markdown('<p class="stage-tag">Inside the resolver</p>', unsafe_allow_html=True)
st.markdown('<h2 class="stage-title">Branch-level attribution, recomputed from records</h2>', unsafe_allow_html=True)
st.markdown('<p class="stage-help">For each node, what it reported vs. what the records say. Branch C reuses the same '
            'record-relative check as the linear chain demo (<code>ChainGovernance</code>), applied to that branch alone.</p>',
            unsafe_allow_html=True)
brows = "".join(
    f'<tr class="{"" if c.matches else "bad"}"><td><b>{c.node_id}</b> &middot; {esc(c.agent)}</td><td>{esc(c.reported)}</td>'
    f'<td>{esc(c.recomputed)}</td><td class="{"ok" if c.matches else "no"}">{"match" if c.matches else "diverged"}</td><td>{esc(c.detail)}</td></tr>'
    for c in star.attribution.branch_checks
)
masked = ", ".join(f"{DIAMOND_NODE_ID[a]} ({a})" for a in star.attribution.masked_by) or "none"
st.markdown(
    f'<div class="scene"><table class="facts"><thead><tr><th>Node</th><th>Reported</th><th>Records say</th><th>Check</th><th>Detail</th></tr></thead>'
    f'<tbody>{brows}</tbody></table>'
    f'<p style="margin:.7rem 0 0;font-size:.9rem">Origin: <b style="color:var(--red)">{esc(star.attribution.origin_node or "none")} '
    f'({esc(star.attribution.origin_agent or "-")})</b> &middot; masked by: <b>{esc(masked)}</b> &middot; '
    f'category: <code>{esc(star.attribution.category or "-")}</code></p></div>',
    unsafe_allow_html=True,
)

with st.expander("Proof: correctness signal and final AGT decision"):
    st.markdown(f"**Behavioral correctness resolver** `{ad['resolver']}` subclasses `{ad['resolver_base']}`. "
                f"It supplies a record-relative correctness signal through the resolver hook on "
                f"`{governor.engine_class or 'the local fallback'}`; it is not the final Action Governance decision.")
    st.markdown("**Resolver AuthorityDecision** - deterministic correctness signal")
    st.json(ad)
    st.markdown("**Final AGT PolicyDecision** - returned by the separate Action Governance evaluation")
    st.json(pd_)
    st.markdown(f"**AuditLog** (`{POLICY_VERSION}`, hash-chained by AGT when live)")
    st.dataframe([{k: v for k, v in row.items()} for row in star.audit], hide_index=True, width="stretch")

# --------------------------------------------------------------------------- batch
st.markdown('<p class="stage-tag">Batch</p>', unsafe_allow_html=True)
st.markdown(f'<h2 class="stage-title">The whole session: {len(reviews)} alerts through the governance pipeline</h2>', unsafe_allow_html=True)
all_hops_ok = [r for r in reviews if all(h.passed for h in r.handoffs)]
auth_but_wrong = [r for r in all_hops_ok if not r.attribution.passed]
blocked = [r for r in reviews if r.policy_decision["action"] == "deny"]
human_review = [r for r in reviews if r.policy_decision["action"] == "require_approval"]
unreconciled_conflicts = [r for r in reviews if r.attribution.category == "unreconciled-conflict"]
nemo_flagged = [r for r in reviews if any(not c["passed"] for c in r.content_checks)]
nemo_unavailable_count = sum(not r.nemo_live for r in reviews)
total_hops = sum(len(r.handoffs) for r in reviews)
ok_hops = sum(h.passed for r in reviews for h in r.handoffs)
st.markdown(
    f'<div class="score">'
    f'<div class="metric"><strong>{len(reviews)}</strong><small>alerts reviewed</small></div>'
    f'<div class="metric"><strong>{ok_hops} / {total_hops}</strong><small>AGT handoffs authorized</small></div>'
    f'<div class="metric{" hot" if nemo_flagged else ""}"><strong>{len(nemo_flagged)} / {nemo_unavailable_count}</strong><small>NeMo flags / unavailable</small></div>'
    f'<div class="metric hot"><strong>{len(auth_but_wrong)}</strong><small>correctness flags</small></div>'
    f'<div class="metric{" hot" if unreconciled_conflicts else ""}"><strong>{len(unreconciled_conflicts)}</strong><small>unreconciled conflicts</small></div>'
    f'<div class="metric{" hot" if blocked or human_review else ""}"><strong>{len(blocked)} / {len(human_review)}</strong><small>blocked / human review</small></div>'
    f'</div>',
    unsafe_allow_html=True,
)
show_all = st.toggle("Show every diamond", value=False)
rows_out = [
    {"alert": r.outcome.alert.alert_id,
     "hops authorized": f"{sum(h.passed for h in r.handoffs)}/{len(r.handoffs)}",
    "NeMo": "FLAG" if any(not c["passed"] for c in r.content_checks) else ("PASS" if r.nemo_live else "UNAVAILABLE"),
    "correctness": "PASS" if r.attribution.passed else "FLAG",
     "case type": r.attribution.category or "clean",
     "B": r.outcome.step(SANCTIONS_AGENT).finding, "C": r.outcome.step(TRANSACTION_AGENT).finding,
     "D": r.outcome.step(KYC_AGENT).finding, "merged": r.attribution.merged_disposition,
     "screening state consumed by E": (
         f"{r.outcome.step(DISPOSITION_AGENT).received_findings_by_parent.get(SANCTIONS_AGENT, 'missing')} "
         f"v{r.outcome.step(DISPOSITION_AGENT).received_versions_by_parent.get(SANCTIONS_AGENT, 'unknown')}"
     ),
     "records require": r.attribution.expected_disposition,
    "final AGT PolicyDecision": r.policy_decision["action"].upper(),
    "matched rule": r.policy_decision.get("matched_rule") or "default",
    "attributed to": r.attribution.origin_node or "",
     "outcome": r.status}
    for r in reviews if show_all or not r.attribution.passed
]
st.dataframe(rows_out, hide_index=True, width="stretch")