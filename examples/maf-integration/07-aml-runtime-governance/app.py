"""Streamlit demo: fan-out/fan-in AML governance running as AGT's authority resolver.

The intake agent fans one alert out to three parallel branches
(customer-screening, transaction-analysis, kyc-entity); the decision agent merges
them. The transaction-analysis branch misreads the deposits and reports "not structuring"; its two correct, clean
siblings mask it, and the merge returns a confident CLEAR. Every handoff passes
AGT's static rules. The merge commit then goes through AGT's PolicyEngine, which
calls the AuthorityResolver slot. AGT ships that slot empty; ours fills it,
recomputes every branch from the records, and denies with branch attribution.

Run:  streamlit run app.py
"""

from __future__ import annotations

import html

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
from agt_resolver import GraphGovernor, POLICY_VERSION
from domain import STRUCTURING_WINDOW_DAYS, recompute_structuring

st.set_page_config(page_title="AGT Graph Governance", page_icon="◆", layout="wide")

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
/* lanes */
.lane { border:1px solid var(--line); background:#fff; padding:1.1rem 1.15rem; height:100%; }
.lane.pass { border-color:#b7dcc6; background:#fbfdfb; } .lane.deny { border:2px solid var(--red); background:var(--red-bg); } .lane.halt { border-color:#e4aeb2; background:#fff8f8; }
.lane .src { font:700 .72rem ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted); margin:0 0 .35rem; letter-spacing:.02em; }
.lane .q { color:var(--muted); font-size:.86rem; margin:0 0 .5rem; }
.lane .verdict { font-weight:800; font-size:1.25rem; margin:0 0 .5rem; letter-spacing:-.01em; }
.lane.pass .verdict { color:var(--green); } .lane.deny .verdict, .lane.halt .verdict { color:var(--red); }
.lane .body { color:var(--ink); font-size:.9rem; line-height:1.5; margin:0; }
.lane ul { margin:.3rem 0 0; padding-left:1.1rem; font-size:.84rem; } .lane li { margin:.1rem 0; }
.quote { margin:.6rem 0 0; padding:.6rem .7rem; background:#fff; border:1px solid #e4aeb2; font:.8rem/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--ink); }
.quote .agtpre { color:var(--muted); } .quote .ourtxt { color:var(--red); font-weight:700; }
.takeaway { margin:1rem 0 0; padding:1rem 1.2rem; background:var(--amber-bg); border:1px solid #ecd9ad; font-size:.96rem; line-height:1.55; }
.control-box { padding:1rem 1.2rem; border:1px solid var(--line); background:#fff; } .control-label { margin-bottom:.3rem; font-weight:700; } .control-help { color:var(--muted); font-size:.85rem; }
.score { display:grid; grid-template-columns:repeat(4,1fr); gap:.75rem; margin:1.2rem 0; } .metric { min-height:88px; padding:1rem; border:1px solid var(--line); background:#fff; } .metric strong { display:block; font-size:1.9rem; } .metric small { color:var(--muted); } .metric.hot { border-color:#e4aeb2; background:#fff8f8; } .metric.hot strong { color:var(--red); }
.diamond { width:100%; height:auto; display:block; }
@media(max-width:820px){ .pipe{grid-template-columns:1fr 1fr} .score{grid-template-columns:repeat(2,1fr)} .block-container{padding:1rem .8rem 3rem} }
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
    status_text = (f'<code>{esc(governor.engine_class)}</code> &middot; resolver registered via '
                   f'<code>engine.set_authority_resolver(BehavioralAuthorityResolver())</code>')
else:
    status_pill = '<span class="pill fallback">AGT PolicyEngine: fallback</span>'
    status_text = ('agentmesh not importable &mdash; the same resolver runs standalone. '
                   'Install <code>agent-governance-toolkit-core</code> to run it inside AGT.')
st.markdown(f'<div class="agt-bar">{status_pill}<span>{status_text}</span></div>', unsafe_allow_html=True)

st.markdown(
    '<div class="hero">'
    '<p class="kicker">Runtime agent governance on a fan-out / fan-in AML investigation graph</p>'
    '<h1>AGT authorized every handoff. Our resolver, inside AGT, still said DENY.</h1>'
    '<p>One alert fans out to three parallel agents and fans back in to a fourth that merges their '
    'results. One branch misreads the deposits. Its two siblings are correct and clean, so the merge '
    'returns a confident CLEAR. Every handoff passes AGT’s rules. The verdict that stops it comes from '
    'AGT’s own <b>PolicyEngine</b>, through the <b>AuthorityResolver</b> slot AGT ships empty. The logic '
    'in that slot is ours.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- where our logic lives
st.markdown('<p class="stage-tag">Where the decision is made</p>', unsafe_allow_html=True)
st.markdown('<h2 class="stage-title">Inside AGT’s pipeline, in the slot AGT leaves for us</h2>', unsafe_allow_html=True)
st.markdown(
    '<div class="pipe">'
    '<div class="st"><b>1 &nbsp;PolicyEngine.evaluate()</b><span>Every handoff and the final commit call AGT.</span><br><span class="who agt">AGT</span></div>'
    '<div class="st"><b>2 &nbsp;Static JSON rules</b><span>pre_tool: is this handoff on a declared edge, within delegated actions?</span><br><span class="who agt">AGT + policy file</span></div>'
    '<div class="st ours"><b>3 &nbsp;AuthorityResolver slot</b><span>AGT ships <code>DefaultAuthorityResolver</code>: allow everything. '
    'We register <code>BehavioralAuthorityResolver</code>.</span><br><span class="who us">OUR LOGIC</span></div>'
    '<div class="st"><b>4 &nbsp;PolicyDecision + AuditLog</b><span>AGT turns the resolver’s deny into its decision and hash-chains it.</span><br><span class="who agt">AGT</span></div>'
    '</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- controls
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.markdown('<div class="control-box"><div class="control-label">Batch size</div><div class="control-help">Diamonds in the session (star case at #7).</div></div>', unsafe_allow_html=True)
    count = st.slider("Diamonds", 8, 80, 40, label_visibility="collapsed")
with c2:
    st.markdown('<div class="control-box"><div class="control-label">Replay seed</div><div class="control-help">Same seed reproduces the run.</div></div>', unsafe_allow_html=True)
    seed = int(st.number_input("Seed", min_value=1, value=7, step=1, label_visibility="collapsed"))
with c3:
    st.markdown('<div class="control-box"><div class="control-label">Deterministic</div><div class="control-help">Synthetic records only. No LLM calls.</div></div>', unsafe_allow_html=True)

session = run_diamond_session(count, seed)
reviews = [governor.review(o) for o in session]
star = next(r for r in reviews if r.outcome.is_star_case)
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
    f'sub-threshold cash deposits totalling <b>{money(truth.aggregate)}</b> within {STRUCTURING_WINDOW_DAYS} days &mdash; '
    f'{"a structuring pattern that must be filed (FILE_SAR)." if truth.is_structuring else "not structuring."}</p></div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- scene 2: the diamond
def diamond_svg(review) -> str:
    s = {x.agent: x for x in review.outcome.steps}
    ck = {c.agent: c for c in review.attribution.branch_checks}
    hop = {(h.sender, h.recipient): h for h in review.handoffs}
    W, H, BW, BH = 900, 400, 212, 92
    pos = {TRIAGE_AGENT: (20, 154), SANCTIONS_AGENT: (350, 18), TRANSACTION_AGENT: (350, 154),
           KYC_AGENT: (350, 290), DISPOSITION_AGENT: (680, 154)}
    role = {TRIAGE_AGENT: "fans out the alert", SANCTIONS_AGENT: "screens watchlist", TRANSACTION_AGENT: "reads the deposits",
            KYC_AGENT: "verifies identity", DISPOSITION_AGENT: "merges 3 branches"}

    def node(agent: str) -> str:
        x, y = pos[agent]
        st_ = s[agent]
        c = ck.get(agent)
        if agent == TRANSACTION_AGENT and c and not c.matches:
            fill, stroke, sw, tone = "#fff0f1", "#b12632", 2.5, "#b12632"
            line2 = f"reports: {st_.finding}"
            line3 = f"used {money(st_.used_aggregate)} (records {money(truth.aggregate)})"
        elif agent == DISPOSITION_AGENT:
            wrong = c is not None and not c.matches
            fill, stroke, sw, tone = ("#fbf3e2", "#9a6819", 2, "#9a6819") if wrong else ("#ffffff", "#d7e0e5", 1, "#17212b")
            line2 = f"merged: {st_.disposition}"
            line3 = "confident: 3/3 branches clean" if st_.finding.endswith("(3/3 branches clean)") else st_.finding
        elif agent == TRIAGE_AGENT:
            fill, stroke, sw, tone = "#ffffff", "#d7e0e5", 1, "#17212b"
            line2, line3 = "routes to B, C, D", ""
        else:
            fill, stroke, sw, tone = "#ffffff", "#b7dcc6", 1.5, "#176b45"
            line2, line3 = f"reports: {st_.finding}", "correct vs records"
        return (f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
                f'<text x="{x + 12}" y="{y + 24}" font-size="14" font-weight="700" fill="#17212b">{st_.node_id} &#183; {esc(agent)}</text>'
                f'<text x="{x + 12}" y="{y + 42}" font-size="11.5" fill="#667580">{esc(role[agent])}</text>'
                f'<text x="{x + 12}" y="{y + 62}" font-size="13" font-weight="700" fill="{tone}">{esc(line2)}</text>'
                f'<text x="{x + 12}" y="{y + 80}" font-size="11" fill="{tone}">{esc(line3)}</text>')

    def edge(a: str, b: str) -> str:
        ax, ay = pos[a]; bx, by = pos[b]
        x1, y1, x2, y2 = ax + BW, ay + BH / 2, bx, by + BH / 2
        h = hop[(a, b)]
        col = "#176b45" if h.passed else "#b12632"
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        label = "✓ handoff OK" if h.passed else "✗ handoff denied"
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2 - 4}" y2="{y2}" stroke="{col}" stroke-width="2" marker-end="url(#ah)"/>'
                f'<rect x="{mx - 48}" y="{my - 10}" width="96" height="19" rx="3" fill="#e7f5ed" stroke="#b7dcc6"/>'
                f'<text x="{mx}" y="{my + 4}" font-size="10.5" font-weight="700" fill="{col}" text-anchor="middle">{label}</text>')

    edges = "".join(edge(a, b) for a, b in review.outcome.edges)
    nodes = "".join(node(a) for a in pos)
    return (f'<svg class="diamond" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,system-ui,sans-serif" role="img" aria-label="Diamond graph">'
            f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="#667580"/></marker></defs>{edges}{nodes}</svg>')


c_step = steps[TRANSACTION_AGENT]
st.markdown(
    f'<div class="scene"><h4>2 &nbsp; The graph: A fans out, B / C / D run in parallel, E merges</h4>{diamond_svg(star)}'
    f'<p style="margin:-.3rem 0 .7rem;font-size:.84rem;color:var(--muted)">'
    f'<b style="color:var(--green)">Green edge labels</b> = AGT permission check (was this handoff allowed?). '
    f'<b style="color:var(--red)">Red box</b> = our correctness check (is the agent\u2019s output right vs the records?). '
    f'An agent can be <b>permitted yet wrong</b> \u2014 that is the gap this governs.</p>'
    f'<p style="margin:.6rem 0 0;font-size:.9rem;color:var(--muted)">C read the deposits as '
    f'{", ".join(money(a) for a in c_step.used_amounts)} instead of {", ".join(money(a) for a in truth.qualifying_amounts)}. '
    f'B and D are right, and they are clean. E sees three clean branches and merges a confident <b>{esc(steps[DISPOSITION_AGENT].disposition)}</b>.</p></div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- scene 3: three verdict sources
st.markdown('<p class="stage-tag">Three verdict sources for this run</p>', unsafe_allow_html=True)
st.markdown('<h2 class="stage-title">Same alert, three questions</h2>', unsafe_allow_html=True)

n_ok = sum(h.passed for h in star.handoffs)
hop_items = "".join(
    f'<li>{DIAMOND_NODE_ID[h.sender]} &rarr; {DIAMOND_NODE_ID[h.recipient]} &nbsp;<b style="color:var(--{"green" if h.passed else "red"})">'
    f'{esc(h.decision)}</b> <span class="mono">{esc(h.rule or "")}</span></li>' for h in star.handoffs
)
pd_ = star.policy_decision
ad = star.authority_decision
reason = pd_["reason"] or ""
prefix = "Authority resolver denied: "
quote = (f'<span class="agtpre">{esc(prefix)}</span><span class="ourtxt">{esc(reason[len(prefix):])}</span>'
         if reason.startswith(prefix) else esc(reason))
denied = not pd_["allowed"]
status_word = {"halted": "HALTED", "committed": "COMMITTED", "escalated": "ESCALATED"}[star.status]
engine_src = "agentmesh PolicyEngine.evaluate(stage='post_tool')" if star.agt_live else "fallback (same resolver, no AGT)"

l1, l2, l3 = st.columns([1, 1.35, 0.85])
with l1:
    st.markdown(
        f'<div class="lane pass"><p class="src">SECURITY &middot; AGT PER-HOP (pre_tool)</p>'
        f'<p class="q">Was each handoff <b>permitted</b>? (permission, not correctness)</p>'        
        f'<p class="verdict">{n_ok} / {len(star.handoffs)} AUTHORIZED</p>'
        f'<p class="body">Every edge is declared and every action is delegated. Static AGT rules pass all six.</p>'
        f'<ul>{hop_items}</ul></div>',
        unsafe_allow_html=True,
    )
with l2:
    st.markdown(
        f'<div class="lane {"deny" if denied else "pass"}"><p class="src">GOVERNANCE &middot; AGT POLICYENGINE &rarr; OUR BehavioralAuthorityResolver</p>'
        f'<p class="q">Is the merged decision right, given the records?</p>'
        f'<p class="verdict">{"DENY" if denied else "ALLOW"}{" &mdash; attributed to branch " + esc(star.attribution.origin_node) if star.attribution.origin_node else ""}</p>'
        f'<p class="body">AGT’s PolicyDecision for <code>commit_disposition</code>, verbatim. '
        f'Grey is AGT’s wrapper; red is our resolver’s <code>narrowing_reason</code>:</p>'
        f'<div class="quote">{quote}</div>'
        f'<p class="body" style="margin-top:.55rem;font-size:.8rem;color:var(--muted)">source: <code>{esc(engine_src)}</code> &middot; '
        f'resolver invoked by engine: <b>{ad["invoked_by_engine"]}</b> &middot; returned <code>{esc(ad["type"])}</code></p></div>',
        unsafe_allow_html=True,
    )
with l3:
    st.markdown(
        f'<div class="lane {"halt" if star.status == "halted" else "pass"}"><p class="src">OUTCOME</p>'
        f'<p class="q">What happened to the action?</p>'
        f'<p class="verdict">{status_word}</p>'
        f'<p class="body">{"The CLEAR was not committed. A deny from the PolicyEngine halts the action; the alert goes back for investigation." if star.status == "halted" else "The disposition was committed."}</p></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div class="takeaway"><b>What to take from this:</b> six green AGT hops and a confident merged CLEAR, on a real '
    f'structuring pattern. Per-hop checks can’t see it, because no hop broke a rule. A check that only looks at the merge '
    f'can’t see it either, because 3/3 branches came back clean. Our resolver recomputes each branch from the records, '
    f'finds that branch {esc(star.attribution.origin_node)} is the one that diverged, and returns that verdict through AGT’s '
    f'own decision pipeline.</div>',
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

with st.expander("Proof: the raw AGT objects for this decision"):
    st.markdown(f"**Resolver class** `{ad['resolver']}` subclasses `{ad['resolver_base']}`; registered with "
                f"`engine.set_authority_resolver(...)` on `{governor.engine_class or 'n/a (fallback)'}`.")
    st.markdown("**PolicyDecision** (returned by `PolicyEngine.evaluate`)")
    st.json(pd_)
    st.markdown("**AuthorityDecision** (returned by our `resolve()`; AGT called it)")
    st.json(ad)
    st.markdown(f"**AuditLog** (`{POLICY_VERSION}`, hash-chained by AGT when live)")
    st.dataframe([{k: v for k, v in row.items()} for row in star.audit], hide_index=True, width="stretch")

# --------------------------------------------------------------------------- batch
st.markdown('<p class="stage-tag">Batch</p>', unsafe_allow_html=True)
st.markdown(f'<h2 class="stage-title">The whole session: {len(reviews)} diamonds through the same engine</h2>', unsafe_allow_html=True)
all_hops_ok = [r for r in reviews if all(h.passed for h in r.handoffs)]
auth_but_wrong = [r for r in all_hops_ok if not r.attribution.passed]
halted = [r for r in reviews if r.status == "halted"]
total_hops = sum(len(r.handoffs) for r in reviews)
ok_hops = sum(h.passed for r in reviews for h in r.handoffs)
st.markdown(
    f'<div class="score">'
    f'<div class="metric"><strong>{len(reviews)}</strong><small>diamonds reviewed</small></div>'
    f'<div class="metric"><strong>{ok_hops} / {total_hops}</strong><small>handoffs authorized by AGT</small></div>'
    f'<div class="metric hot"><strong>{len(auth_but_wrong)}</strong><small>authorized end-to-end but wrong</small></div>'
    f'<div class="metric{" hot" if halted else ""}"><strong>{len(halted)}</strong><small>halted by AGT &rarr; our resolver</small></div>'
    f'</div>',
    unsafe_allow_html=True,
)
show_all = st.toggle("Show every diamond", value=False)
rows_out = [
    {"alert": r.outcome.alert.alert_id,
     "hops authorized": f"{sum(h.passed for h in r.handoffs)}/{len(r.handoffs)}",
     "B": r.outcome.step(SANCTIONS_AGENT).finding, "C": r.outcome.step(TRANSACTION_AGENT).finding,
     "D": r.outcome.step(KYC_AGENT).finding, "merged": r.attribution.merged_disposition,
     "records require": r.attribution.expected_disposition,
     "AGT decision": r.policy_decision["action"].upper(), "attributed to": r.attribution.origin_node or "",
     "outcome": r.status}
    for r in reviews if show_all or not r.attribution.passed
]
st.dataframe(rows_out, hide_index=True, width="stretch")
