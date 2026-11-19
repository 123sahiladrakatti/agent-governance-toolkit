"""Long-running demo: the same case, window, and check, with little vs. a lot of history.

Case 1 runs the care-decision agent with a small accumulated history; Case 2 runs it with a
large synthesized history, as if the agent had been retaining context for months. The
only thing that changes between the cases is how much accumulated history is in the agent's
context. Each run goes through ``instrumentation.run_case``, so the verdict is the existing
deterministic trace-integrity check plus the harness's required-evidence check.

Case 2 is an accelerated simulation: the history is synthesized filler, not months of
real operation. The filler is neutral operations notes and is checked never to contain the
required finding's content. The demo observes and records; it does not change the case
records, the finding, the window, or the agent's decision logic.

    python longrun_demo.py                       # values from harness.longrun_demo
    python longrun_demo.py --window 4096 --early 200 --aged 2400
"""

from __future__ import annotations

import copy
import importlib
import json
import re
from datetime import date, timedelta
from typing import Any, Mapping

from instrumentation import Harness, HarnessConfigError, _resolve, case_json_for, load_harness, run_case

PLACEMENTS = ("recent_end", "buried_mid", "overflowed")
# Neutral, plausible accumulated history: operational notes from prior runs. None of these
# words describe a clinical finding; the guard below enforces that for the actual finding.
_EVENTS = (
    "intake routed to the review queue",
    "queue rebalanced across shifts",
    "audit entry archived",
    "handoff notes filed by the night team",
    "batch import completed",
    "bed census updated",
    "shift roster confirmed",
    "transport request closed",
    "billing export scheduled",
    "interpreter booking confirmed",
    "dashboard refresh completed",
    "document scan indexed",
    "callback list cleared",
    "supply order placed",
    "workstation login audited",
    "visitor log rotated",
)


class FillerGuardError(ValueError):
    """Synthesized history would contain the required finding's content; the demo refuses it."""


# ---------------------------------------------------------------- the required finding
def _model_agent(harness: Harness):
    agent = next((a for a in harness.agents.values() if a.calls_model and a.required_evidence), None)
    if agent is None:
        raise HarnessConfigError("no model-calling agent with required_evidence is declared in the harness")
    return agent


def required_finding(case_json: Mapping[str, Any], harness: Harness | None = None) -> dict[str, Any]:
    """The finding the demo tracks: its id, emitted version and value, and its line in the prompt.

    Computed from the pipeline's own deterministic branches and prompt builder; no LLM call.
    """

    harness = harness or load_harness(case_json)
    pipe, demo = harness.pipeline, harness.config.get("longrun_demo", {})
    agent = _model_agent(harness)
    evidence_id = demo.get("finding") or agent.required_evidence[0]
    if evidence_id not in agent.required_evidence:
        raise HarnessConfigError(f"longrun_demo.finding {evidence_id!r} is not required by {agent.name}")
    codecs = _resolve(pipe["records_codecs"])
    records = codecs[case_json["system"]["schema"]][1](case_json["records"])
    module = importlib.import_module(pipe["module"])
    state = getattr(module, pipe["run_function"])(records)  # deterministic branches only
    source = harness.config["evidence_formats"][agent.evidence_format]["source"]
    steps = state[source["state_key"]]
    builder = _resolve(demo.get("prompt_builder") or "")
    config = _resolve(pipe["model_config_factory"])(**{pipe.get("model_config_field", "model"): harness.model["name"]})
    _, lines = builder(records, {e: steps[e] for e in agent.required_evidence}, config)
    step = steps[evidence_id]
    return {
        "id": evidence_id,
        "version": int(getattr(step, source["version_field"])),
        "finding": str(getattr(step, source["finding_field"])),
        "line": lines[evidence_id],
        "agent": agent.name,
    }


# ---------------------------------------------------------------- synthesized history
def _banned_words(finding: Mapping[str, Any]) -> set[str]:
    words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", finding["line"])}
    words |= {part.lower() for part in re.split(r"[^A-Za-z0-9]+", f"{finding['id']} {finding['finding']}") if part}
    return words


def _guard(text: str, finding: Mapping[str, Any], harness: Harness) -> None:
    for fmt in harness.config["evidence_formats"].values():
        if re.search(fmt["pattern"], text, re.M):
            raise FillerGuardError("synthesized history matches the evidence format")
    if "[REPORT" in text:
        raise FillerGuardError("synthesized history contains a report line")
    present = {w.lower() for w in re.findall(r"[A-Za-z0-9]+", text)}
    leaked = sorted(_banned_words(finding) & present)
    if leaked:
        raise FillerGuardError(f"synthesized history contains the finding's words: {', '.join(leaked)}")


def synthesize_accumulated_context(
    size_tokens: int,
    finding: Mapping[str, Any],
    placement: str,
    *,
    harness: Harness,
    start_date: str | None = None,
    cadence_days: int | None = None,
    depth: float | None = None,
) -> dict[str, Any]:
    """Accumulated prior-run notes of about ``size_tokens`` (harness tokenizer), placed around the finding.

    ``recent_end``: all history before the finding (the finding is the most recent item).
    ``buried_mid``: older history before the finding, newer history after it. ``depth`` is the
    fraction of the history that is newer than the finding (piled on top of it); the default
    0.5 buries it in the middle, 0.9 buries it deep under the newest 90%.
    ``overflowed``: all history after the finding, pushing it toward the front of the context.
    """

    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {', '.join(PLACEMENTS)}")
    demo = harness.config.get("longrun_demo", {})
    start = date.fromisoformat(start_date or demo["start_date"])
    cadence = int(cadence_days if cadence_days is not None else demo["cadence_days"])
    entries: list[str] = []
    total = 0
    run = 0
    while total < size_tokens:
        run += 1
        events = "; ".join(_EVENTS[(run * 3 + k) % len(_EVENTS)] for k in range(3))
        entry = f"{(start + timedelta(days=(run - 1) * cadence)).isoformat()} run {run:03d} · HC-PRIOR-{run:04d}: {events}."
        entries.append(entry)
        total += harness.count(entry + "\n", with_bos=False)
    if placement == "recent_end":
        before, after = entries, []
    elif placement == "buried_mid":
        newer = depth if depth is not None else 0.5
        if not 0.0 < newer < 1.0:
            raise ValueError("depth must be between 0 and 1 (exclusive)")
        split = int(len(entries) * (1.0 - newer))
        before, after = entries[:split], entries[split:]
    else:
        before, after = [], entries
    context = {"before_finding": "\n".join(before), "after_finding": "\n".join(after)}
    for text in context.values():
        _guard(text, finding, harness)
    tokens = sum(harness.count(text, with_bos=False) for text in context.values() if text)
    return {
        **context,
        "placement": placement,
        "depth": (len(after) / len(entries)) if entries else 0.0,
        "tokens_after_finding": harness.count(context["after_finding"], with_bos=False) if after else 0,
        "tokens": tokens,
        "runs": len(entries),
        "span_days": (len(entries) - 1) * cadence if entries else 0,
        "first_date": start.isoformat() if entries else None,
        "last_date": (start + timedelta(days=(len(entries) - 1) * cadence)).isoformat() if entries else None,
    }


# ---------------------------------------------------------------- mechanism
def classify_mechanism(trace: list[dict[str, Any]], finding_id: str | None = None) -> str:
    """``overflow``, ``degradation``, or ``indeterminate``, from what the trace measured.

    overflow: the runtime processed fewer tokens than were assembled and the finding fell in
    the truncated part. degradation: the finding was inside the kept window but the agent did
    not use it. Anything the trace cannot settle is ``indeterminate``; nothing is assumed.
    """

    for record in trace:
        if not record.get("model_call"):
            continue
        for evidence_id in ([finding_id] if finding_id else record.get("required_evidence") or []):
            assembled, processed = record.get("assembled_tokens"), record.get("processed_tokens")
            truncated = bool(record.get("overflow_error")) or (
                assembled is not None and processed is not None and assembled > processed
            )
            inside = (record.get("evidence_in_kept_window_estimate") or {}).get(evidence_id)
            used = (record.get("evidence_used") or {}).get(evidence_id)
            if truncated and inside is False:
                return "overflow"
            if (not truncated or inside is True) and used is False:
                return "degradation"
    return "indeterminate"


# ---------------------------------------------------------------- the demo
def _case(case_json, window, context, finding, harness) -> dict[str, Any]:
    trace, verdict = run_case(
        case_json, window,
        {"before_finding": context["before_finding"], "after_finding": context["after_finding"]},
        harness=harness,
    )
    record = next((r for r in trace if r.get("agent") == finding["agent"]), {})
    fid = finding["id"]
    return {
        "placement": context["placement"],
        "depth": context.get("depth"),
        "tokens_after_finding": context.get("tokens_after_finding"),
        "history_tokens": context["tokens"],
        "history_runs": context["runs"],
        "history_span_days": context["span_days"],
        "history_dates": [context["first_date"], context["last_date"]],
        "window": window,
        "occupied_tokens": record.get("assembled_tokens"),
        "processed_tokens": record.get("processed_tokens"),
        "truncated": record.get("truncated"),
        "finding_present_in_prompt": (record.get("evidence_present_in_prompt") or {}).get(fid),
        "finding_inside_window": (record.get("evidence_in_kept_window_estimate") or {}).get(fid),
        "agent_used_finding": (record.get("evidence_used") or {}).get(fid),
        "decision": record.get("decision"),
        "passed": bool(verdict.get("passed")),
        "verdict": {k: verdict.get(k) for k in ("passed", "category", "reason", "integrity_passed",
                                                "required_evidence_carried", "evidence_check")},
        "mechanism": None if verdict.get("passed") else classify_mechanism(trace, fid),
        "confirmation": record.get("confirmation"),
        "trace": [{k: v for k, v in r.items() if k != "assembled_prompt"} for r in trace],
    }


def run_demo(
    case_json: Mapping[str, Any],
    window: int,
    early_size: int,
    aged_size: int,
    *,
    harness: Harness | None = None,
) -> dict[str, Any]:
    """Run Case 1 (early, little history) and Case 2 (aged, large history) at the same window."""

    harness = harness or load_harness(case_json)
    original_records = copy.deepcopy(case_json["records"])
    finding = required_finding(case_json, harness)

    def synth(size, placement):
        return synthesize_accumulated_context(size, finding, placement, harness=harness)

    case1 = _case(case_json, window, synth(early_size, "recent_end"), finding, harness)
    attempts = [_case(case_json, window, synth(aged_size, "buried_mid"), finding, harness)]
    burial_insufficient = attempts[0]["passed"]
    if burial_insufficient:
        # Burial inside the window did not lose the finding; grow the history further so it
        # pushes the finding past the window. The size derives from the window argument.
        attempts.append(_case(case_json, window, synth(max(aged_size, window), "overflowed"), finding, harness))
    if case_json["records"] != original_records:
        raise RuntimeError("the demo altered the case records")
    return {
        "finding": {k: finding[k] for k in ("id", "version", "finding", "agent")},
        "window": window,
        "case1": case1,
        "case2": attempts[-1],
        "case2_attempts": attempts,
        "burial_alone_insufficient": burial_insufficient,
        "model": harness.model["name"],
    }


# ---------------------------------------------------------------- report
def _yes_no(value: Any) -> str:
    return "yes" if value is True else "no" if value is False else "unknown"


def format_report(report: Mapping[str, Any]) -> str:
    c1, c2 = report["case1"], report["case2"]

    def occupied(case):
        return (f"{case['occupied_tokens']} tokens ({case['history_tokens']} of history, "
                f"~{case['history_runs']} prior runs over {case['history_span_days']} days)")

    def placement(case):
        label = {"recent_end": "near end (after the history)", "buried_mid": "buried mid-history",
                 "overflowed": "overflowed (history after it)"}[case["placement"]]
        if case is c2 and report["burial_alone_insufficient"]:
            label = "buried mid-history -> " + label + "; burial alone was insufficient"
        return label

    def inside(case):
        if case["finding_inside_window"] is True:
            return "yes" if case["placement"] != "buried_mid" else "yes, but buried mid-context"
        if case["finding_inside_window"] is False:
            return "no (pushed past the kept window)"
        return "unknown"

    def check(case):
        if case["passed"]:
            return "PASS"
        return "CATCHES LOSS" if case is c2 else f"FLAGGED ({case['verdict']['category']})"

    rows = [
        ("occupied context", occupied(c1), occupied(c2)),
        ("window", str(c1["window"]), str(c2["window"])),
        ("finding placement", placement(c1), placement(c2)),
        ("finding inside window?", inside(c1), inside(c2)),
        ("agent used finding?", _yes_no(c1["agent_used_finding"]), _yes_no(c2["agent_used_finding"])),
        ("decision", str(c1["decision"]), str(c2["decision"])),
        ("governance check", check(c1), check(c2)),
        ("failure mechanism", c1["mechanism"] or "-", c2["mechanism"] or "-"),
    ]
    months = c2["history_span_days"] / 30.4
    span = f"~{months:.0f} months" if months >= 1.5 else f"~{c2['history_span_days']} days"
    head1, head2 = "Case 1 (early)", f"Case 2 ({span} of history, simulated)"
    width = max(len(r[0]) for r in rows)
    w1 = max(len(head1), *(len(r[1]) for r in rows))
    w2 = max(len(head2), *(len(r[2]) for r in rows))
    out = [f"| {'':{width}} | {head1:{w1}} | {head2:{w2}} |",
           f"|{'-' * (width + 2)}|{'-' * (w1 + 2)}|{'-' * (w2 + 2)}|"]
    out += [f"| {a:{width}} | {b:{w1}} | {c:{w2}} |" for a, b, c in rows]
    headline, notes = _overflow_summary(report)
    return "\n".join([*out, "", headline, "", *(f"- {n}" for n in notes)])


def _overflow_summary(report: Mapping[str, Any]) -> tuple[str, list[str]]:
    c1, c2 = report["case1"], report["case2"]
    fid = report["finding"]["id"]
    if c1["passed"] and not c2["passed"]:
        headline = (
            "Same system, same data, same check. Clean for the first few runs; after simulated months of "
            f"accumulated context, the required finding is lost ({c2['mechanism']}) and the governance check "
            "catches it."
        )
    elif not c1["passed"]:
        headline = (
            f"Case 1 already fails at window {c1['window']} ({c1['verdict']['category']}): the base case does not "
            "fit cleanly, so there is no early/aged contrast to show. Choose a larger window (for example with "
            "instrumentation.smallest_safe_window) and rerun."
        )
    else:
        headline = "Neither case lost the finding in these runs; no loss was observed at this window and history size."
    notes = [
        "Case 2 is an accelerated simulation: the accumulated context is synthesized prior-run notes, not "
        "months of real operation. The filler is checked never to contain the finding's content.",
        f"Case 1: no loss observed in these runs (finding '{fid}' used, check passed)." if c1["passed"] else
        "Case 1: the check flagged this run.",
        "The window is the same for both cases; only the amount of accumulated history differs.",
        "'Finding inside window' is a token-position estimate from the runtime's processed-token count: this "
        "runtime cannot show the model-visible prompt (measured-only).",
        "These are real LLM runs; results are probabilistic.",
    ]
    if report["burial_alone_insufficient"]:
        buried = report["case2_attempts"][0]
        notes.append(
            f"Burying the finding mid-history did not lose it: at {buried['occupied_tokens']} occupied tokens "
            f"({buried['history_tokens']} of history, inside the window) the agent still used it and the check "
            "passed, so no degradation was observed. Case 2 was rerun with the history grown past the window "
            "(overflowed)."
        )
    return headline, notes


# ---------------------------------------------------------------- degradation demo
def _degradation_settings(harness: Harness, overrides: Mapping[str, Any]) -> dict[str, Any]:
    settings = dict(harness.config.get("longrun_demo", {}).get("degradation", {}))
    settings.update({k: v for k, v in overrides.items() if v is not None})
    missing = [k for k in ("growth_factor", "max_aged_tokens", "max_window", "margin_tokens") if k not in settings]
    if missing:
        raise HarnessConfigError(f"harness.longrun_demo.degradation is missing {', '.join(missing)}")
    return settings


def _window_for(occupied: int, window: int, settings: Mapping[str, Any]) -> int | None:
    """The smallest window, growing by ``growth_factor``, that holds ``occupied`` plus the margin."""
    needed = occupied + int(settings["margin_tokens"])
    ceiling = int(settings["max_window"])
    if needed > ceiling:
        return None
    while window < needed:
        window = int(window * float(settings["growth_factor"]))
    return min(window, ceiling)


def run_degradation_demo(
    case_json: Mapping[str, Any],
    window_large: int,
    early_size: int,
    aged_size: int,
    depth: float,
    *,
    harness: Harness | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Degradation ("lost in the middle"): the finding stays inside the window but goes unused.

    Case 1 runs with a little history before the finding. Case 2 buries the finding at ``depth``
    under ``aged_size`` tokens of history that still fits inside the window: the runtime must
    process every assembled token (no truncation), or the run is not a degradation case and is
    retried with a larger window. If the model still uses the finding, the history is grown by
    ``growth_factor`` up to ``max_aged_tokens`` (the window grows with it, up to ``max_window``)
    to find the degradation onset. If none is found, the report says so; it never fakes a loss.
    Settings not passed come from ``harness.longrun_demo.degradation``.
    """

    harness = harness or load_harness(case_json)
    settings = _degradation_settings(harness, overrides)
    original_records = copy.deepcopy(case_json["records"])
    finding = required_finding(case_json, harness)

    def synth(size, placement, at_depth=None):
        return synthesize_accumulated_context(size, finding, placement, harness=harness, depth=at_depth)

    case1 = _case(case_json, window_large, synth(early_size, "recent_end"), finding, harness)
    base = (case1["occupied_tokens"] or 0) - case1["history_tokens"]
    sweep: list[dict[str, Any]] = []
    window, size, outcome = window_large, aged_size, None
    while outcome is None:
        fitted = _window_for(base + size, window, settings)
        if fitted is None:
            # The ceiling window cannot hold this much history: shrink the history to fit once.
            ceiling = int(settings["max_window"])
            size = min(size, ceiling - base - int(settings["margin_tokens"]))
            if size <= 0 or any(a["history_tokens"] >= size for a in sweep):
                outcome = "window-ceiling"
                break
            fitted = ceiling
        window = fitted
        attempt = _case(case_json, window, synth(size, "buried_mid", depth), finding, harness)
        attempt["inside_window_proven"] = (
            attempt["truncated"] is False and attempt["occupied_tokens"] is not None
            and attempt["processed_tokens"] is not None
            and attempt["occupied_tokens"] <= attempt["processed_tokens"]
        )
        sweep.append(attempt)
        if not attempt["inside_window_proven"] or attempt["mechanism"] == "overflow":
            attempt["invalid"] = "truncated: the window was too small for this history; not a degradation case"
            grown = _window_for((attempt["occupied_tokens"] or base + size) + int(settings["margin_tokens"]),
                                int(window * float(settings["growth_factor"])), settings)
            if grown is None:
                outcome = "window-ceiling"
            else:
                window = grown
            continue
        if not attempt["passed"]:
            outcome = "degradation" if attempt["mechanism"] == "degradation" else "other-failure"
            break
        next_size = int(size * float(settings["growth_factor"]))
        if next_size > int(settings["max_aged_tokens"]):
            outcome = "robust"
            break
        size = next_size
    if case_json["records"] != original_records:
        raise RuntimeError("the demo altered the case records")
    valid = [a for a in sweep if a.get("inside_window_proven")]
    case2 = (valid or sweep or [None])[-1]
    return {
        "kind": "degradation",
        "finding": {k: finding[k] for k in ("id", "version", "finding", "agent")},
        "window_large": window_large,
        "depth": depth,
        "case1": case1,
        "case2": case2,
        "sweep": sweep,
        "outcome": outcome,
        "onset": None if outcome != "degradation" else {
            "history_tokens": case2["history_tokens"], "occupied_tokens": case2["occupied_tokens"],
            "window": case2["window"], "depth": case2["depth"], "tokens_after_finding": case2["tokens_after_finding"],
        },
        "max_history_tested": max((a["history_tokens"] for a in valid), default=0),
        "settings": settings,
        "model": harness.model["name"],
    }


def format_degradation_report(report: Mapping[str, Any]) -> str:
    c1, c2 = report["case1"], report["case2"]

    def occupied(case):
        return (f"{case['occupied_tokens']} tokens ({case['history_tokens']} of history, "
                f"~{case['history_runs']} prior runs over {case['history_span_days']} days)")

    def inside(case):
        if case.get("truncated") is False and case["occupied_tokens"] == case["processed_tokens"]:
            return f"yes, NOT truncated ({case['processed_tokens']} of {case['occupied_tokens']} tokens processed)"
        if case.get("truncated"):
            return f"no, truncated ({case['processed_tokens']} of {case['occupied_tokens']} processed)"
        return "unknown"

    def position(case):
        if case["placement"] == "recent_end":
            return "near end (after the history)"
        return (f"buried deep: {case['depth']:.0%} of the history is newer than it "
                f"(~{case['tokens_after_finding']} tokens on top)")

    def check(case):
        if case["passed"]:
            return "PASS"
        return "CATCHES LOSS" if case is c2 else f"FLAGGED ({case['verdict']['category']})"

    months = c2["history_span_days"] / 30.4 if c2 else 0
    span = f"~{months:.0f} months" if months >= 1.5 else f"~{c2['history_span_days']} days"
    head1, head2 = "Case 1 (early)", f"Case 2 ({span} of history, buried deep)"
    rows = [
        ("occupied context", occupied(c1), occupied(c2)),
        ("window", str(c1["window"]), str(c2["window"])),
        ("finding inside window?", inside(c1), inside(c2)),
        ("finding position", position(c1), position(c2)),
        ("agent used finding?", _yes_no(c1["agent_used_finding"]), _yes_no(c2["agent_used_finding"])),
        ("decision", str(c1["decision"]), str(c2["decision"])),
        ("governance check", check(c1), check(c2)),
        ("mechanism", c1["mechanism"] or "-",
         f"{c2['mechanism']} (detected)" if c2["mechanism"] else "-"),
    ]
    width = max(len(r[0]) for r in rows)
    w1 = max(len(head1), *(len(r[1]) for r in rows))
    w2 = max(len(head2), *(len(r[2]) for r in rows))
    out = [f"| {'':{width}} | {head1:{w1}} | {head2:{w2}} |", f"|{'-' * (width + 2)}|{'-' * (w1 + 2)}|{'-' * (w2 + 2)}|"]
    out += [f"| {a:{width}} | {b:{w1}} | {c:{w2}} |" for a, b, c in rows]
    headline, notes = _degradation_summary(report)
    sweep = ["Sweep (each row is a real run; only non-truncated runs count as degradation evidence):"]
    for a in report["sweep"]:
        sweep.append(
            f"  window {a['window']:>6} · history {a['history_tokens']:>6} · occupied {a['occupied_tokens']} · "
            f"processed {a['processed_tokens']} · truncated {a['truncated']} · used {_yes_no(a['agent_used_finding'])}"
            f" · {'PASS' if a['passed'] else 'FAIL'}" + (f" · {a['mechanism']}" if a["mechanism"] else "")
            + (" · INVALID (truncated)" if a.get("invalid") else "")
        )
    return "\n".join([*out, "", headline, "", *sweep, "", *(f"- {n}" for n in notes)])


def _degradation_summary(report: Mapping[str, Any]) -> tuple[str, list[str]]:
    c1, c2 = report["case1"], report["case2"]
    model, fid = report["model"], report["finding"]["id"]
    outcome = report["outcome"]
    windows = sorted({c1["window"], *(a["window"] for a in report["sweep"])})
    same_window = ("same large window" if len(windows) == 1 else
                   f"a large window (grown {windows[0]} -> {windows[-1]} only to keep the history inside it)")
    if not c1["passed"]:
        headline = (f"Case 1 already fails at window {c1['window']} ({c1['verdict']['category']}), so there is no "
                    "early/aged contrast to show.")
    elif outcome == "degradation":
        headline = (
            f"Same system, same data, {same_window}. The finding is inside the window the whole time. After "
            "simulated months of accumulated history, the model stops using it (degradation) and the governance "
            "check catches it; a bigger window does not help."
        )
    elif outcome == "robust":
        headline = (
            f"Same system, same data, {same_window}. No degradation observed up to "
            f"{report['max_history_tested']} tokens of accumulated history with the finding buried at depth "
            f"{report['depth']:.0%}, inside the window and never truncated: {model} held the finding throughout. "
            "No loss observed in these runs."
        )
    elif outcome == "other-failure":
        headline = (f"Case 2 failed, but not by losing the buried finding '{fid}' (mechanism: "
                    f"{c2['mechanism']}; {c2['verdict']['reason']}). This is not a degradation result.")
    else:
        headline = (f"The history could not be kept inside the window up to the configured ceiling "
                    f"({report['settings']['max_window']} tokens), so degradation could not be tested further.")
    notes = [
        "Case 2 is an accelerated simulation: the accumulated history is synthesized prior-run notes, not months "
        "of real operation. The filler is checked never to contain the finding's content.",
        f"Case 1: no loss observed in these runs (finding '{fid}' used, check passed)." if c1["passed"] else
        "Case 1: the check flagged this run.",
        "Degradation evidence requires no truncation: the runtime processed every assembled token, so the "
        "finding was inside the kept window. 'Inside window' is measured from the runtime's processed-token "
        "count; this runtime cannot show the model-visible prompt (measured-only).",
        "These are real LLM runs; results are probabilistic.",
    ]
    return headline, notes


def report_summary(report: Mapping[str, Any]) -> tuple[str, list[str]]:
    """The headline and notes for a ``run_demo`` or ``run_degradation_demo`` report (same wording as the CLI)."""
    if report.get("kind") == "degradation":
        return _degradation_summary(report)
    return _overflow_summary(report)


def _main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Long-running context-loss demo (Case 1 vs Case 2).")
    parser.add_argument("--definition", default="client_pipelines/healthcare_pipeline.json")
    parser.add_argument("--records", default="harness_cases/healthcare_contraindication.json")
    parser.add_argument("--window", type=int, help="defaults to harness.longrun_demo.window")
    parser.add_argument("--early", type=int, help="defaults to harness.longrun_demo.early_tokens")
    parser.add_argument("--aged", type=int, help="defaults to harness.longrun_demo.aged_tokens")
    parser.add_argument("--json", help="also write the full report (with traces) to this file")
    parser.add_argument("--mode", choices=("overflow", "degradation"), default="overflow")
    parser.add_argument("--depth", type=float, help="degradation: defaults to harness.longrun_demo.degradation.depth")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with open(args.records, encoding="utf-8") as handle:
        case = case_json_for(args.definition, json.load(handle))
    harness = load_harness(case)
    demo = harness.config.get("longrun_demo", {})
    if args.mode == "degradation":
        demo = {**demo, **demo.get("degradation", {})}
    values = {name: getattr(args, arg) if getattr(args, arg) is not None else demo.get(key)
              for name, arg, key in (("window", "window", "window"), ("early", "early", "early_tokens"),
                                     ("aged", "aged", "aged_tokens"))}
    if args.mode == "degradation":
        values["depth"] = args.depth if args.depth is not None else demo.get("depth")
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise SystemExit(f"set {', '.join(missing)} on the command line or in harness.longrun_demo")
    if args.mode == "degradation":
        report = run_degradation_demo(case, values["window"], values["early"], values["aged"], values["depth"],
                                      harness=harness)
        print(format_degradation_report(report))
    else:
        report = run_demo(case, values["window"], values["early"], values["aged"], harness=harness)
        print(format_report(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)


if __name__ == "__main__":
    _main()


__all__ = [
    "FillerGuardError",
    "PLACEMENTS",
    "classify_mechanism",
    "format_degradation_report",
    "format_report",
    "report_summary",
    "required_finding",
    "run_degradation_demo",
    "run_demo",
    "synthesize_accumulated_context",
]
