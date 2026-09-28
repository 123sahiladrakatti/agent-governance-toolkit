"""Standalone interactive AML runtime-governance mini-demo.

The user sets up a deposit scenario. Three AI agents then handle it
(investigation -> case -> filing). The investigation agent MISREADS the first
deposit (a fixed low value), which can flip a real structuring case into a wrong
'CLEAR'. Runtime governance checks each step as it runs and HALTS the chain the
moment the misread is detected - before the wrong decision is ever committed.

The user configures the scenario; the *agent* is the one that errs. This is
agent governance, not fraud detection. Run with:  streamlit run interactive_demo.py
"""

from __future__ import annotations

import time
import streamlit as st

# --- AML rule constants (self-contained; mirror the main demo) ---
CTR_THRESHOLD = 10_000.0
STRUCTURING_MIN_DEPOSITS = 3
STRUCTURING_AGGREGATE = 30_000.0
MISREAD_VALUE = 0.0            # the agent overlooks the first deposit entirely (reads it as nothing)

st.set_page_config(page_title="AML Runtime Governance - Interactive", page_icon="\u25c6", layout="centered")

st.markdown("""
<style>
:root { --ink:#17212b; --muted:#667580; --line:#d7e0e5; --paper:#f5f7f6; --green:#176b45; --green-bg:#e7f5ed; --red:#b12632; --red-bg:#fff0f1; --gold:#9a6819; --amber-bg:#fbf3e2; --blue:#1f5c8a; }
html, body, [class*="css"] { font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,sans-serif; }
.stApp { background:var(--paper); color:var(--ink); }
.block-container { max-width:720px; padding:2rem 1.5rem 4rem; }
.hero { padding:1.4rem 1.5rem; border:1px solid var(--line); background:#fff; margin-bottom:1rem; }
.hero h1 { margin:.1rem 0 .4rem; font-size:1.7rem; letter-spacing:-.02em; }
.hero p { margin:0; color:var(--muted); font-size:.95rem; line-height:1.5; }
.step { border:1px solid var(--line); background:#fff; padding:.9rem 1.05rem; margin:.5rem 0; }
.step.ok { border-left:5px solid var(--green); }
.step.halt { border-left:5px solid var(--red); background:var(--red-bg); }
.step.skip { border-left:5px solid var(--line); opacity:.55; }
.step .who { font-weight:700; font-size:.98rem; }
.step .detail { color:var(--muted); font-size:.88rem; margin-top:.25rem; line-height:1.45; }
.step .badge { float:right; font:700 .7rem ui-monospace,Menlo,monospace; padding:.2rem .45rem; }
.step.ok .badge { color:var(--green); background:var(--green-bg); }
.step.halt .badge { color:var(--red); background:#ffdfe1; }
.step.skip .badge { color:var(--muted); background:#eef1f0; }
.banner { padding:1.1rem 1.2rem; margin-top:1rem; font-size:1rem; line-height:1.5; }
.banner.prevented { background:var(--green-bg); border:1px solid #b6dcc6; color:#12543a; }
.banner.clean { background:var(--green-bg); border:1px solid #b6dcc6; color:#12543a; }
.note { color:var(--muted); font-size:.85rem; margin-top:.4rem; }
.deposit-tag { display:inline-block; padding:.15rem .5rem; margin:.15rem; background:#eef1f0; font-variant-numeric:tabular-nums; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)


def money(v: float) -> str:
    return f"${v:,.0f}"


def is_structuring(amounts: list[float]) -> tuple[bool, float, int]:
    """Structuring = 3+ sub-threshold cash deposits totalling >= the aggregate."""
    qualifying = [a for a in amounts if 0 < a < CTR_THRESHOLD]
    agg = sum(qualifying)
    flag = len(qualifying) >= STRUCTURING_MIN_DEPOSITS and agg >= STRUCTURING_AGGREGATE
    return flag, agg, len(qualifying)


def disposition_for(amounts: list[float]) -> str:
    flag, _, _ = is_structuring(amounts)
    return "FILE_SAR" if flag else "CLEAR"


# ---------------------------------------------------------------- header
st.markdown(
    '<div class="hero"><h1>Set up a deposit. Watch the agent \u2014 and the governance.</h1>'
    '<p>You enter the deposits a customer made. Three AI agents then handle the alert: one reads the '
    'records, one builds the case, one files the decision. The reading agent makes a mistake \u2014 and '
    'runtime governance checks each step as it runs, halting the chain before a wrong decision is committed. '
    'You set up the scenario; the <b>agent</b> is the one that errs.</p></div>',
    unsafe_allow_html=True,
)

st.markdown("**The customer's cash deposits** (each under the $10,000 reporting line is the interesting case):")
cols = st.columns(4)
defaults = [9200, 9600, 9200, 8600]
amounts: list[float] = []
for i, c in enumerate(cols):
    with c:
        amounts.append(float(st.number_input(f"Deposit {i+1}", min_value=0, value=defaults[i], step=100, key=f"dep{i}")))

run = st.button("Run the investigation", type="primary", use_container_width=True)

# Show the setup summary
truth_flag, truth_agg, truth_n = is_structuring(amounts)
truth_disp = disposition_for(amounts)
st.markdown(
    '<div class="note">You entered: '
    + " ".join(f'<span class="deposit-tag">{money(a)}</span>' for a in amounts)
    + f' &nbsp;\u2192&nbsp; total of the sub-$10k deposits: <b>{money(truth_agg)}</b> across {truth_n}. '
    + (f'This <b>is</b> a structuring pattern \u2014 the correct decision is <b>FILE_SAR</b>.'
       if truth_flag else
       'This is <b>not</b> a structuring pattern \u2014 the correct decision is <b>CLEAR</b>.')
    + '</div>',
    unsafe_allow_html=True,
)

if not run:
    st.stop()

# ---------------------------------------------------------------- the run
# What the agent SEES after misreading the first deposit (fixed low value).
misread_amounts = list(amounts)
original_first = misread_amounts[0]
misread_amounts[0] = MISREAD_VALUE
agent_flag, agent_agg, _ = is_structuring(misread_amounts)
agent_disp = disposition_for(misread_amounts)

# Governance check at step 1: does what the agent used match the real records?
misread_detected = (misread_amounts != amounts)
# The halt only matters if the misread actually flips a true structuring case to a wrong CLEAR.
harmful = truth_flag and agent_disp != truth_disp

placeholder = st.empty()

def render(step_states: dict) -> None:
    html = ""
    # Step 1 - investigation (reads records)
    s1 = step_states.get(1)
    if s1 == "ok":
        html += (f'<div class="step ok"><span class="badge">EXECUTED</span>'
                 f'<div class="who">investigation-agent \u00b7 step 1</div>'
                 f'<div class="detail">Read the deposits and reached: <b>{agent_disp}</b>.</div></div>')
    elif s1 == "halt":
        html += (f'<div class="step halt"><span class="badge">HALTED HERE</span>'
                 f'<div class="who">investigation-agent \u00b7 step 1</div>'
                 f'<div class="detail">Overlooked the first deposit of <b>{money(original_first)}</b> entirely (read it as nothing). '
                 f'Recomputing from the real records gives {money(truth_agg)} '
                 f'({truth_disp}), but the agent concluded {agent_disp}. Governance halts the chain here.</div></div>')
    # Steps 2 and 3
    for idx, who, role in [(2, "case-agent", "would build the case file"), (3, "filing-agent", "would file the decision")]:
        s = step_states.get(idx)
        if s == "ok":
            html += (f'<div class="step ok"><span class="badge">EXECUTED</span>'
                     f'<div class="who">{who} \u00b7 step {idx}</div>'
                     f'<div class="detail">Carried the decision forward: <b>{agent_disp}</b>.</div></div>')
        elif s == "skip":
            html += (f'<div class="step skip"><span class="badge">NEVER RAN</span>'
                     f'<div class="who">{who} \u00b7 step {idx}</div>'
                     f'<div class="detail">{role.capitalize()} \u2014 but the chain was already halted, so it never ran.</div></div>')
    placeholder.markdown(html, unsafe_allow_html=True)


# Animate step by step
if harmful:
    # step 1 runs, gets halted; 2 and 3 never run
    render({1: "ok"}); time.sleep(0.7)
    render({1: "halt"}); time.sleep(0.5)
    render({1: "halt", 2: "skip", 3: "skip"})
    st.markdown(
        f'<div class="banner prevented"><b>Consequential action prevented.</b> The agent overlooked '
        f'a {money(original_first)} deposit and was about to clear a real structuring case. '
        f'Governance caught it at step 1 and halted \u2014 the case and filing agents never ran, so the wrong '
        f'<b>CLEAR</b> was never filed. The correct decision ({truth_disp}) is preserved for a human to action.</div>',
        unsafe_allow_html=True,
    )
else:
    # Either not structuring, or the misread didn't change the outcome -> chain completes, no false alarm.
    render({1: "ok"}); time.sleep(0.5)
    render({1: "ok", 2: "ok"}); time.sleep(0.5)
    render({1: "ok", 2: "ok", 3: "ok"})
    if not truth_flag:
        st.markdown(
            f'<div class="banner clean"><b>No halt \u2014 and correctly so.</b> These deposits are not a structuring '
            f'pattern (correct decision: {truth_disp}). Even though the agent misread the first deposit, the outcome '
            f'is unchanged, so governance does not stop the chain. It only halts when a mistake would actually change '
            f'the decision \u2014 it does not cry wolf.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="banner clean"><b>No halt.</b> Overlooking that deposit did not change the final decision ({truth_disp}), '
            f'so governance lets the chain proceed. It halts only when a mistake would actually flip the outcome.</div>',
            unsafe_allow_html=True,
        )

st.caption("Synthetic demonstration of runtime agent governance. Not a real AML detector, SAR system, or compliance decision.")