"""Fast, read-only Streamlit view over materialized governance batches."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import streamlit as st

from batch_store import BatchStore, DEFAULT_BATCH_DB
from healthcare.batch import BATCH_MODES


def _esc(value: object) -> str:
    return html.escape(str(value))


@st.cache_data(ttl=2, show_spinner=False)
def _latest(path: str, domain: str, modified_ns: int) -> tuple[Any, list[dict[str, Any]]]:
    del modified_ns  # cache invalidation key; the filesystem supplied it
    store = BatchStore(path)
    batch = store.latest_batch(domain)
    rows = [] if batch is None else store.case_summaries(batch.batch_id)
    return batch, rows


@st.cache_data(ttl=10, show_spinner=False)
def _document(path: str, batch_id: str, sequence: int, modified_ns: int) -> dict[str, Any] | None:
    del modified_ns
    return BatchStore(path).case_document(batch_id, sequence)


def _stage_rows(document: dict[str, Any]) -> list[dict[str, str]]:
    governance = document.get("governance") or {}
    review = governance.get("review") or {}
    integrity = governance.get("integrity") or {}
    attribution = review.get("attribution") or {}
    policy = review.get("policy_decision") or {}
    handoffs = review.get("handoffs") or []
    content = [row for row in review.get("content_checks") or [] if row.get("rail") != "skipped"]
    authorized = all(row.get("passed") for row in handoffs)
    nemo_flagged = any(not row.get("passed", False) for row in content)
    nemo_unavailable = any(row.get("rail") == "unavailable" for row in content)
    nemo_result = "FLAG" if nemo_flagged else ("UNAVAILABLE" if nemo_unavailable else "PASS")
    nemo_explanation = (
        "Content rail flagged an agent output." if nemo_flagged else
        "Content rail was unavailable; this is not presented as a pass." if nemo_unavailable else
        "No content violation detected."
    )
    return [
        {"governance check": "Pipeline ingestion", "result": "PASS",
         "explanation": "Backend payload was normalized and stored."},
        {"governance check": "Deterministic authorization", "result": "PASS" if authorized else "DENY",
         "explanation": "All declared handoffs were authorized." if authorized else "A handoff was not authorized."},
        {"governance check": "Trace integrity", "result": "PASS" if integrity.get("passed") else "FLAG",
         "explanation": str(integrity.get("reason") or "No trace result stored.")},
        {"governance check": "NeMo content", "result": nemo_result, "explanation": nemo_explanation},
        {"governance check": "Correctness plug-in", "result": "PASS" if attribution.get("passed") else "FLAG",
         "explanation": str(attribution.get("reason") or "No correctness result stored.")},
        {"governance check": "Action Governance", "result": str(policy.get("action") or "unknown").upper(),
         "explanation": str(policy.get("reason") or "No final policy reason stored.")},
    ]


@st.fragment(run_every="2s")
def render_batch_dashboard(domain: str, database: Path | str = DEFAULT_BATCH_DB) -> None:
    """Render the latest batch without running agents or governance in Streamlit."""

    path = Path(database)
    st.markdown('<p class="section-label">BATCH GOVERNANCE · MATERIALIZED BACKEND RESULTS</p>',
                unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Governance operations dashboard</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-help">The company batch is ingested and governed by a separate backend worker. '
        'This page reads already-materialized results from the system store, so it does not wait for pipeline '
        'execution or model calls. Refreshing this page is a database read only.</p>',
        unsafe_allow_html=True,
    )

    if not path.is_file():
        st.info(
            "No governed batch has reached the frontend store yet. The dashboard will populate automatically "
            "after the backend ingestion worker commits its first case."
        )
        return

    modified_ns = path.stat().st_mtime_ns
    batch, rows = _latest(str(path), domain, modified_ns)
    if batch is None:
        st.info(f"No {domain} batch has been ingested into the result store yet.")
        return

    if st.button("Refresh from backend", key=f"refresh_batch_{domain}"):
        _latest.clear()
        _document.clear()
        st.rerun()

    if batch.status == "PROCESSING":
        fraction = batch.processed_cases / batch.total_cases if batch.total_cases else 0
        st.progress(fraction, text=f"Backend processing: {batch.processed_cases} of {batch.total_cases} cases stored")
    elif batch.status == "FAILED":
        st.warning("The latest backend batch stopped early. Successfully stored cases remain available below.")
    else:
        st.success(f"Latest batch available · completed {batch.completed_at}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Processed", f"{batch.processed_cases}/{batch.total_cases}")
    c2.metric("Continue", batch.continue_count)
    c3.metric("Human review", batch.review_count)
    c4.metric("Blocked", batch.block_count)
    c5.metric("Errors", batch.error_count)
    mode = BATCH_MODES.get(batch.mode, batch.mode)
    st.caption(
        f"Source: {batch.source_name} · mode: {mode} · source version: {batch.source_sha256[:12]}… · "
        f"batch: {batch.batch_id[:8]}"
    )

    st.dataframe([
        {
            "case": row.get("case_id") or f"row {row['source_row']}",
            "subject": row["subject"],
            "agent disposition": row.get("disposition") or "-",
            "trace": row.get("trace_check") or "-",
            "NeMo": row.get("nemo") or "-",
            "governance verdict": row.get("verdict") or "ERROR",
            "where / why": row.get("error") or row.get("caught_by") or "-",
        }
        for row in rows
    ], hide_index=True, width="stretch")

    if not rows:
        st.info("The batch exists, but no case results have been committed yet.")
        return

    by_sequence = {int(row["sequence"]): row for row in rows}
    selected = st.selectbox(
        "Inspect a governed case",
        tuple(by_sequence),
        format_func=lambda sequence: (
            f"{by_sequence[sequence].get('case_id') or 'row ' + str(by_sequence[sequence]['source_row'])} · "
            f"{by_sequence[sequence]['subject']} · {by_sequence[sequence].get('verdict') or 'ERROR'}"
        ),
    )
    document = _document(str(path), batch.batch_id, selected, modified_ns)
    if document is None:
        st.warning("The selected result is no longer available; refresh to read the newest batch state.")
        return
    summary = document["summary"]
    if summary.get("error"):
        st.error(summary["error"])
        return

    governance = document.get("governance") or {}
    review = governance.get("review") or {}
    attribution = review.get("attribution") or {}
    verdict = summary.get("verdict") or "ERROR"
    if verdict == "CONTINUE":
        st.success("CONTINUE · All available governance signals passed.")
    elif verdict == "HUMAN REVIEW":
        st.warning(f"HUMAN REVIEW · {_esc(attribution.get('reason') or summary.get('caught_by'))}")
    else:
        st.error(f"{_esc(verdict)} · {_esc(attribution.get('reason') or summary.get('caught_by'))}")

    st.dataframe(_stage_rows(document), hide_index=True, width="stretch")
    steps = ((governance.get("outcome") or {}).get("steps") or [])
    if steps:
        st.markdown("**Agent execution trace**")
        st.dataframe([{
            "agent": step.get("agent"),
            "finding": step.get("finding"),
            "version": step.get("finding_version"),
            "disposition": step.get("disposition") or "-",
            "received from": ", ".join(step.get("received_from") or []) or "root",
        } for step in steps], hide_index=True, width="stretch")
    st.caption(
        "The complete normalized payload, records, trace, content checks, correctness attribution, policy decision, "
        "model metadata, and audit data remain stored in the backend result document. The client view exposes only "
        "the operational decision and attribution."
    )


__all__ = ["render_batch_dashboard"]
