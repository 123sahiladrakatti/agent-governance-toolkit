"""GACO demo: advisory context-window sizing for the healthcare contraindication case.

Standalone (independent of app.py):

    streamlit run gaco_app.py

Shows the measured floor, the suggested smallest-safe window, how far the client's current
setting is from it, and the probe ladder. GACO is advisory: it never changes any pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

import gaco
from instrumentation import case_json_for, load_harness

HERE = Path(__file__).parent
DEFINITION = HERE / "client_pipelines" / "healthcare_pipeline.json"
RECORDS = HERE / "harness_cases" / "healthcare_contraindication.json"
SAVED_REPORT = HERE / "gaco_report.json"
SAVED_COMMAND = "python gaco.py --initial-window 4096 --json gaco_report.json"
DEFAULT_INITIAL_WINDOW = 4096  # the input's starting value only; the client can change it

MODES = ("Load saved report", "Run GACO live")


# ---------------------------------------------------------------- defensive report accessors
def _tokens(value: Any) -> int | None:
    """A number that may be stored bare or as {"tokens"/"floor": n, ...}."""
    if isinstance(value, Mapping):
        value = value.get("tokens", value.get("floor"))
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _search(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return report.get("search") or {}


def _probe_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    details = _search(report).get("probe_details")
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
    for i, probe in enumerate(report.get("probes") or _search(report).get("probes") or [], 1):
        if isinstance(probe, Mapping):
            window, passed = probe.get("window"), probe.get("passed")
        else:
            window, passed = (list(probe) + [None, None])[:2]
        rows.append({"#": i, "window": window, "result": "PASS" if passed else "FAIL"})
    return rows


def _runs_used(report: Mapping[str, Any], rows: list[dict[str, Any]]) -> int:
    for key in ("model_runs", "runs_used"):
        if isinstance(report.get(key), int):
            return report[key]
    return len(rows)


def _ceiling(report: Mapping[str, Any]) -> Any:
    return _search(report).get("hi") or (report.get("settings") or {}).get("max_window")


# ---------------------------------------------------------------- rendering
def render(report: Mapping[str, Any], source: str) -> None:
    rows = _probe_rows(report)
    floor = _tokens(report.get("floor"))
    suggested = _tokens(report.get("suggested_window"))
    initial = _tokens(report.get("initial_window"))
    found = bool(report.get("found")) and suggested is not None
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
            relation = f" (the same as the initial {initial})"
        else:
            side = "below" if suggested < initial else "above"
            relation = f" ({abs(suggested - initial)} tokens {side} the initial {initial})"
        st.success(f"### Suggested window: {suggested}{relation}.")
    else:
        ceiling = _ceiling(report)
        reason = _search(report).get("stop_reason")
        st.warning(
            f"**No safe window found** up to the ceiling{f' of {ceiling} tokens' if ceiling else ''}"
            f"{f' ({reason})' if reason else ''}. The case may not fit this model's window at all: it may "
            "need less per-case evidence, or a model with a larger context window."
        )

    cols = st.columns(5)
    cols[0].metric("Floor · measured", floor if floor is not None else "—",
                   help="Exact token count of the decision agent's prompt plus its reply room. No model runs.")
    cols[1].metric("Suggested · found by search", suggested if found else "—",
                   help="The smallest window that passed every run, above every window that ever failed.")
    cols[2].metric("Initial window (client)", initial if initial is not None else "—")
    cols[3].metric("Direction", report.get("direction", "—"))
    cols[4].metric("Real runs used", _runs_used(report, rows))

    floor_parts = report.get("floor") if isinstance(report.get("floor"), Mapping) else {}
    if floor_parts.get("prompt_tokens") is not None:
        st.caption(
            f"Floor = {floor_parts.get('prompt_tokens')} prompt tokens + {floor_parts.get('output_room_tokens')} "
            f"reply room ({floor_parts.get('output_room_source', 'config')}) + {floor_parts.get('margin_tokens', 0)} "
            f"margin, for {floor_parts.get('agent', 'the decision agent')}."
        )

    st.subheader("Probe ladder")
    st.caption("Each row is one real run of the clean case (no synthesized history), in the order GACO ran it.")
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info("No probes were run (the measured floor is already above the ceiling).")
    search = _search(report)
    if search.get("violations"):
        st.warning("Monotonicity violated, so it was treated as model variance and the suggestion was widened:\n\n"
                   + "\n".join(f"- {v}" for v in search["violations"]))
    if report.get("confidence"):
        st.caption(report["confidence"])

    st.subheader("What this does and does not show")
    notes = list(getattr(gaco, "NOTES", ())) or [
        "Advisory: GACO suggests a window; it does not change any pipeline or settings.",
        "The search finds where the required evidence fits and survives; it does not prove the model reasons "
        "over it well in every case.",
        "This is for the shown case only; across a workload, take the largest suggested window.",
    ]
    if rows and all(r["result"] == "PASS" for r in rows):
        notes.append(
            "Every probe passed: this is a clean case. The floor is an exact count and the search confirmed it, "
            "rather than resolving a failure. GACO never probes below the floor, so it cannot suggest less."
        )
    if report.get("evidence_check") == "measured-only" or any(
            d.get("evidence_check") == "measured-only" for d in search.get("probe_details") or []):
        notes.append(
            "Measured-only: this runtime cannot expose the model-visible prompt, so 'fits and survives' is "
            "measured from token counts and the runtime's processed-token count, not verified against what the "
            "model saw."
        )
    st.markdown("\n".join(f"- {n}" for n in notes))

    with st.expander("Full text report (gaco.format_report)"):
        try:
            st.code(gaco.format_report(report), language=None)
        except (KeyError, TypeError, ValueError) as exc:
            st.caption(f"The text report could not be built from this report: {exc}")


# ---------------------------------------------------------------- page
st.set_page_config(page_title="GACO window sizing", page_icon="◆", layout="wide")
st.title("GACO — advisory context-window sizing")
st.markdown(
    "Suggests the smallest window at which a case's required evidence fits and survives. "
    "**Advisory:** GACO recommends a window; it never changes the client's pipeline, prompts, data, or settings."
)

mode = st.radio("Report source", MODES, horizontal=True)

if mode == MODES[0]:
    if not SAVED_REPORT.exists():
        st.info(f"No saved report yet. Create one from this folder with:\n\n```\n{SAVED_COMMAND}\n```")
    else:
        try:
            saved = json.loads(SAVED_REPORT.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            st.error(f"{SAVED_REPORT.name} could not be read: {exc}")
        else:
            render(saved, SAVED_REPORT.name)
else:
    st.warning("A live run makes several real llama3.1:8b calls through Ollama and takes about a minute.")
    initial_window = st.number_input("Client's initial context window (tokens)", min_value=1,
                                     value=DEFAULT_INITIAL_WINDOW, step=1)
    if st.button("Run GACO", type="primary"):
        st.session_state.pop("gaco_live", None)
        with st.spinner("Running GACO: measuring the floor, then probing windows with real model runs"):
            try:
                with open(RECORDS, encoding="utf-8") as handle:
                    case = case_json_for(str(DEFINITION), json.load(handle))
                harness = load_harness(case)
                st.session_state.gaco_live = {"report": gaco.optimize(case, int(initial_window), harness=harness)}
            except Exception as exc:  # Ollama down, harness rejected, model error: show it, no traceback
                st.session_state.gaco_live = {"error": f"{type(exc).__name__}: {exc}"}
    live = st.session_state.get("gaco_live")
    if live and live.get("error"):
        st.error(f"GACO could not run: {live['error']}")
    elif live:
        render(live["report"], "live run")
