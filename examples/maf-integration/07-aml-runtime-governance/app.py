
"""AML reference UI running on the domain-agnostic governance framework.

Run: streamlit run app.py
"""

from __future__ import annotations

import html
import json
from datetime import date, datetime, time

import streamlit as st

from agent_sim import DISPOSITION_AGENT, SANCTIONS_AGENT, build_diamond
from agt_resolver import GraphGovernor
from domain import (
    Alert,
    CustomerProfile,
    GraphOutcome,
    SANCTIONS_WATCHLIST,
    Transaction,
)
from graph_integrity import check_graph_integrity


st.set_page_config(page_title="Governed AML Simulation", page_icon="◆", layout="wide")

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
.event { display:grid; grid-template-columns:18px 205px 105px 1fr; gap:.65rem; align-items:start; padding:.62rem 0; border-bottom:1px solid var(--line); font-size:.85rem; }
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
<<<<<<< HEAD
    '<div class="hero">'
    '<p class="kicker">Runtime agent governance on a fan-out / fan-in AML investigation graph</p>'
    '<h1>Conflicting agent views need reconciliation before clearance.</h1>'
    '<p>One alert fans out to three parallel agents and fans back in to a fourth that merges their '
    'results. In the conflict case, the screening agent is correct \u2014 it flags a probable watchlist match \u2014 '
    'but the disposition agent reads a stale version of that finding from case state and clears the customer '
    'without reconciling the flag. The watchlist itself is current; what is stale is the version of the '
    'screening finding the decider used. AGT authorizes actions; '
    'the behavioral resolver checks consistency and correctness; NeMo checks output content; final Action '
    'Governance blocks or routes the unresolved case to human review.</p>'
    '</div>',
=======
    '<div class="hero"><div class="kicker">AML REFERENCE IMPLEMENTATION</div>'
    '<h1>Enter a case. Watch the governance framework evaluate it.</h1>'
    '<p>AML is the visible example, while authorization, trace integrity, NeMo monitoring, optional correctness, '
    'Action Governance, attribution, and audit remain domain-agnostic framework capabilities.</p></div>',
>>>>>>> 0b063cce (Updated code)
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- AML input
st.markdown('<p class="section-label">RUN THE AML REFERENCE</p>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Customer and transaction information</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-help">The frontend collects case records only. The AML adapter and the generic governance '
    'architecture run behind the application.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="watchlist"><b>Demo watchlist:</b> '
    + ' &middot; '.join(f'<code>{esc(name)}</code>' for name in sorted(SANCTIONS_WATCHLIST))
    + '<br>Use an exact watchlist name to demonstrate a new screening hit that the decision agent receives as stale state.</div>',
    unsafe_allow_html=True,
)

default_transactions = [
    {"id": "T-001", "date": date(2026, 1, 5), "type": "cash_deposit", "amount": 2_500.0, "branch": "BR-01"},
    {"id": "T-002", "date": date(2026, 1, 6), "type": "cash_deposit", "amount": 1_500.0, "branch": "BR-01"},
]

with st.form("aml-case-input"):
    c1, c2 = st.columns(2)
    with c1:
        customer_name = st.text_input("Customer name", placeholder="Jane Smith")
        account_id = st.text_input("Account ID", value="ACCT-DEMO-001")
    with c2:
        occupation = st.text_input("Occupation", value="business owner")
        expected_cash = st.number_input(
            "Expected monthly cash volume", min_value=0.0, value=8_000.0, step=500.0
        )
    monitoring_rule = st.text_input(
        "Monitoring rule", value="customer and transaction activity review"
    )
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
        "Submit case to governed pipeline", type="primary", width="stretch"
    )

<<<<<<< HEAD
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
                "used a stale version of screening\u2019s finding"
                if stale_conflict else
                "confident: 3/3 branches clean" if st_.finding.endswith("(3/3 branches clean)") else st_.finding
=======
if submitted:
    st.session_state.pop("aml_review", None)
    try:
        if not customer_name.strip():
            raise ValueError("Customer name is required")
        if not account_id.strip():
            raise ValueError("Account ID is required")
        if not occupation.strip():
            raise ValueError("Occupation is required for this reference run")
        rows = (
            edited_transactions.to_dict("records")
            if hasattr(edited_transactions, "to_dict") else list(edited_transactions)
        )
        if not rows:
            raise ValueError("Add at least one transaction")
        transactions = tuple(
            Transaction(
                id=str(row["id"]),
                account_id=account_id.strip(),
                amount=float(row["amount"]),
                type=str(row["type"]),
                timestamp=normalize_date(row["date"]),
                branch=str(row["branch"]),
>>>>>>> 0b063cce (Updated code)
            )
            for row in rows
        )
        customer = CustomerProfile(
            account_id=account_id.strip(),
            name=customer_name.strip(),
            stated_occupation=occupation.strip(),
            expected_monthly_cash_volume=float(expected_cash),
        )
        alert = Alert(
            alert_id="AML-INTERACTIVE",
            account_id=account_id.strip(),
            transactions=transactions,
            customer=customer,
            monitoring_rule=monitoring_rule.strip(),
        )
        watchlist_hit = customer.name in SANCTIONS_WATCHLIST
        progress = st.progress(10, text="AML adapter is creating the agent trace")
        steps = build_diamond(alert, stale_screening_view=watchlist_hit)
        outcome = GraphOutcome(
            alert=alert,
            steps=steps,
            fault="stale-screening-view" if watchlist_hit else None,
            is_conflict_case=watchlist_hit,
            resource_id=f"aml-alert:{alert.alert_id}",
        )
        progress.progress(30, text="Normalizing the PipelineTrace")
        governor = GraphGovernor()
        progress.progress(45, text="Running authorization, NeMo, and correctness checks")
        review = governor.review(outcome)
        progress.progress(85, text="AGT Action Governance is combining signals")
        st.session_state.aml_review = {
            "review": review,
            "integrity": check_graph_integrity(outcome),
            "agt_live": governor.agt_live,
        }
        progress.progress(100, text="Governance decision complete")
        progress.empty()
    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"The case could not be processed: {exc}")

<<<<<<< HEAD
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
        f'<small>{esc(consumed_screening_finding.upper())} - stale version of screening\u2019s finding</small></div>'
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
        f'The screening agent correctly found a probable watchlist match (its current finding), but the '
        f'disposition agent consumed an earlier <b>{esc(consumed_screening_finding)}</b> version of that '
        f'finding from stale case state and produced <b>CLEAR</b> without reconciliation. This is a '
        f'multi-agent reconciliation failure, not a stale watchlist \u2014 screening was right; the decider '
        f'read an outdated version of it. '
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
=======
result = st.session_state.get("aml_review")
if result is None:
    st.info("Enter customer and transaction information to run the AML reference through the framework.")
>>>>>>> 0b063cce (Updated code)
    st.stop()

review = result["review"]
integrity = result["integrity"]
steps = {step.agent: step for step in review.outcome.steps}
screening = steps[SANCTIONS_AGENT]
decision = steps[DISPOSITION_AGENT]
consumed_finding = decision.received_findings_by_parent[SANCTIONS_AGENT]
consumed_version = decision.received_versions_by_parent[SANCTIONS_AGENT]
action = str(review.policy_decision.get("action", "deny"))
action_label = {"allow": "CONTINUE", "require_approval": "HUMAN REVIEW", "deny": "BLOCK"}.get(action, action.upper())
verdict_css = {"allow": "continue", "require_approval": "review", "deny": "block"}.get(action, "block")

# --------------------------------------------------------------------------- event stream
st.markdown('<p class="section-label">LIVE GOVERNANCE RUN</p>', unsafe_allow_html=True)
events: list[tuple[str, str, str, str]] = [
    ("AML ingestion adapter", "PASS", "pass", f"Normalized {len(review.outcome.steps)} agent steps."),
]
for handoff in review.handoffs:
    events.append((
        f"AGT · {handoff.sender} → {handoff.recipient}",
        "AUTHORIZED" if handoff.passed else "DENIED",
        "pass" if handoff.passed else "fail",
        handoff.reason or handoff.rule or "Handoff policy evaluated.",
    ))
trace_status = "PASS" if integrity.passed else "STALE / CONFLICT"
events.append((
    "Generic trace checks", trace_status,
    "pass" if integrity.passed else "review", integrity.reason,
))
for check in review.content_checks:
    if not check["passed"]:
        status, css = "FLAG", "review"
    elif check["rail"] == "unavailable":
        status, css = "UNAVAILABLE", "review"
    else:
        status, css = "PASS", "pass"
    events.append((f"NeMo · {check['agent']}", status, css, str(check["reason"])))
events.append((
    "AML correctness plug-in",
    "PASS" if review.attribution.passed else "FLAG",
    "pass" if review.attribution.passed else "review",
    review.attribution.reason,
))
events.append((
    "AGT Action Governance", action_label,
    "pass" if action == "allow" else ("review" if action == "require_approval" else "fail"),
    str(review.policy_decision.get("reason", "Final policy evaluated.")),
))

event_html = "".join(
    f'<div class="event {css}"><span class="dot"></span><span class="phase">{esc(phase)}</span>'
    f'<span class="status">{esc(status)}</span><span class="detail">{esc(detail)}</span></div>'
    for phase, status, css, detail in events
)
engine_pill = (
    '<span class="pill live">AGT LIVE</span>' if result["agt_live"]
    else '<span class="pill fallback">AGT FALLBACK</span>'
)
st.markdown(
    f'<div class="simulator">{engine_pill}{event_html}</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- state comparison and verdict
st.markdown('<p class="section-label">WHERE & WHY</p>', unsafe_allow_html=True)
if integrity.passed:
    st.success("No governance conflict: screening and the decision agent used consistent state.")
else:
    st.markdown(
        f'<div class="state-compare">'
        f'<div class="state-card"><small>{esc(screening.agent)} emitted</small>'
        f'<strong>{esc(screening.finding)} · version {screening.finding_version}</strong></div>'
        f'<div class="state-card stale"><small>{esc(decision.agent)} consumed</small>'
        f'<strong>{esc(consumed_finding)} · version {consumed_version}</strong></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

origin = (
    f"{review.attribution.origin_node or '-'} · {review.attribution.origin_agent}"
    if review.attribution.origin_agent else "No failing agent attributed"
)
st.markdown(
    f'<div class="verdict {verdict_css}"><div class="label">FINAL GOVERNANCE DECISION</div>'
    f'<h2>{esc(action_label)}</h2>'
    f'<p><b>Why:</b> {esc(review.attribution.reason)}</p>'
    f'<p><b>Policy result:</b> {esc(str(review.policy_decision.get("reason", "")))}</p>'
    f'<div class="why-grid">'
    f'<div><small>Customer</small><strong>{esc(review.outcome.alert.customer.name)}</strong></div>'
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
