"""GACO window optimizer: suggest the smallest context window at which a case's evidence survives.

GACO is advisory. For one client case it finds the smallest context window at which the decision
agent's mandatory evidence fits, survives (is not truncated) and is carried into the decision,
and it SUGGESTS that window. It never sets the window and never edits the client's pipeline,
prompts, data, or settings.

Scope: context loss only, meaning evidence that does not fit because the window is too small.
It runs the clean case (no synthesized history, no compaction, no pinning) and searches the
window. The pass/fail oracle is the harness verdict from ``instrumentation.run_case``: the
existing deterministic trace-integrity check plus the required-evidence check.

    python gaco.py --initial-window 4096
    python gaco.py --initial-window 1024 --json gaco_report.json

No window length is hard-coded here. The floor is measured from the case; the search bounds
come from the harness config (``model.advertised_window``, optional ``harness.gaco``).

Optional ``harness.gaco`` settings and their defaults:
    prompt_builder       the decision agent's prompt builder ("pkg.module:function"); falls
                         back to harness.longrun_demo.prompt_builder.
    output_room_tokens   tokens reserved for the agent's reply. If unset, read from the model
                         config field named by output_room_field (default "num_predict"), i.e.
                         the reply length the agent actually requests.
    floor_margin_tokens  extra tokens added to the measured floor (default 0: the floor is an
                         exact count).
    max_window           search ceiling (default model.advertised_window; the smaller wins).
    growth_factor        ratio for the upward search (default 2: doubling; a ratio, not a
                         window length).
    confirm_probes       repeat runs required at the suggested window (default 1).
    resolution           binary-search resolution in tokens (default harness.sizing.resolution,
                         else 1).
"""

from __future__ import annotations

import copy
import importlib
import json
import math
from typing import Any, Mapping

from instrumentation import Harness, HarnessConfigError, _resolve, case_json_for, load_harness, run_case
from longrun_demo import required_finding

DEFAULT_OUTPUT_ROOM_FIELD = "num_predict"   # the reply-length field of the agent's own model config
DEFAULT_FLOOR_MARGIN_TOKENS = 0             # the floor is an exact count; a margin is opt-in
DEFAULT_GROWTH_FACTOR = 2.0                 # doubling for the upward search (a ratio, not a window)
DEFAULT_CONFIRM_PROBES = 1                  # repeat runs required at the suggested window


# ---------------------------------------------------------------- settings
def _settings(harness: Harness) -> dict[str, Any]:
    gaco = harness.config.get("gaco", {})
    try:
        advertised = int(harness.model["advertised_window"])
    except KeyError:
        raise HarnessConfigError("harness.model.advertised_window is required to bound the search") from None
    max_window = min(advertised, int(gaco["max_window"])) if gaco.get("max_window") else advertised
    growth = float(gaco.get("growth_factor", DEFAULT_GROWTH_FACTOR))
    if growth <= 1.0:
        raise HarnessConfigError("harness.gaco.growth_factor must be greater than 1")
    return {
        "advertised_window": advertised,
        "max_window": max_window,
        "floor_margin_tokens": int(gaco.get("floor_margin_tokens", DEFAULT_FLOOR_MARGIN_TOKENS)),
        "growth_factor": growth,
        "confirm_probes": max(0, int(gaco.get("confirm_probes", DEFAULT_CONFIRM_PROBES))),
        "resolution": max(1, int(gaco.get("resolution") or harness.config.get("sizing", {}).get("resolution") or 1)),
    }


def _evidence_check(harness: Harness) -> str:
    return "measured-and-verified" if harness.observability.get("can_capture_model_visible_prompt") else "measured-only"


# ---------------------------------------------------------------- 1. the floor (no model runs)
def measure_floor_detail(case_json: Mapping[str, Any], *, harness: Harness | None = None) -> dict[str, Any]:
    """The measured floor and its parts: prompt tokens + output room + margin. No model runs.

    The prompt is assembled by the pipeline's own prompt builder from its own deterministic
    branch outputs, rendered with the runtime's chat template and counted with the runtime's
    tokenizer. The decision agent sends the case file as a single user message (as
    ``llm_decision._chat`` does), so this is the same count the trace records as
    ``assembled_tokens``.
    """

    harness = harness or load_harness(case_json)
    finding = required_finding(case_json, harness)
    agent = harness.agent(finding["agent"])
    pipe, gaco = harness.pipeline, harness.config.get("gaco", {})
    builder_ref = gaco.get("prompt_builder") or harness.config.get("longrun_demo", {}).get("prompt_builder")
    if not builder_ref:
        raise HarnessConfigError("harness.gaco.prompt_builder is required to measure the floor")
    codecs = _resolve(pipe["records_codecs"])
    records = codecs[case_json["system"]["schema"]][1](case_json["records"])
    module = importlib.import_module(pipe["module"])
    state = getattr(module, pipe["run_function"])(records)  # deterministic branches only: no model call
    source = harness.config["evidence_formats"][agent.evidence_format]["source"]
    steps = state[source["state_key"]]
    model_config = _resolve(pipe["model_config_factory"])(
        **{pipe.get("model_config_field", "model"): harness.model["name"]}
    )
    prompt, _ = _resolve(builder_ref)(records, {e: steps[e] for e in agent.required_evidence}, model_config)
    prompt_tokens = harness.count(harness.render([{"role": "user", "content": prompt}]))
    if "output_room_tokens" in gaco:
        output_room, output_source = int(gaco["output_room_tokens"]), "harness.gaco.output_room_tokens"
    else:
        field = gaco.get("output_room_field", DEFAULT_OUTPUT_ROOM_FIELD)
        if not hasattr(model_config, field):
            raise HarnessConfigError(f"the agent's model config has no {field!r} field for the output room")
        output_room, output_source = int(getattr(model_config, field)), f"model config {field}"
    margin = _settings(harness)["floor_margin_tokens"]
    return {
        "floor": prompt_tokens + output_room + margin,
        "prompt_tokens": prompt_tokens,
        "output_room_tokens": output_room,
        "output_room_source": output_source,
        "margin_tokens": margin,
        "agent": agent.name,
        "required_evidence": list(agent.required_evidence),
        "finding": {k: finding[k] for k in ("id", "version", "finding")},
        "tag": "measured",
    }


def measure_floor(case_json: Mapping[str, Any], *, harness: Harness | None = None) -> int:
    """The smallest window the decision agent could ever need for this case (measured, no model runs)."""
    return measure_floor_detail(case_json, harness=harness)["floor"]


# ---------------------------------------------------------------- 2. the oracle (one model run)
def _run_probe(case_json: Mapping[str, Any], window: int, harness: Harness) -> dict[str, Any]:
    trace, verdict = run_case(case_json, window, None, harness=harness)  # the clean case: no history
    record = next((r for r in trace if r.get("model_call")), {})
    return {
        "window": window,
        "passed": bool(verdict.get("passed")),
        "category": verdict.get("category"),
        "reason": verdict.get("reason"),
        "assembled_tokens": record.get("assembled_tokens"),
        "processed_tokens": record.get("processed_tokens"),
        "truncated": record.get("truncated"),
        "decision": record.get("decision"),
        "evidence_check": verdict.get("evidence_check"),
        "model_calls": (1 + int(record.get("auxiliary_model_calls") or 0)) if record else 0,
    }


def probe(case_json: Mapping[str, Any], window: int, *, harness: Harness | None = None) -> bool:
    """One real run of the clean case at ``window``; True when every required finding was carried."""
    return _run_probe(case_json, window, harness or load_harness(case_json))["passed"]


# ---------------------------------------------------------------- 3. the search
def _grow(window: int, growth: float, hi: int) -> int:
    return min(hi, max(window + 1, int(math.ceil(window * growth))))


def _monotonicity(results: Mapping[int, list[bool]]) -> list[str]:
    """Every observation that breaks 'a larger window cannot make fitting evidence stop fitting'."""
    issues = [f"window {w}: the same window both passed and failed ({sum(rs)} of {len(rs)} runs passed)"
              for w, rs in sorted(results.items()) if len(set(rs)) > 1]
    passed = sorted(w for w, rs in results.items() if any(rs))
    failed = sorted(w for w, rs in results.items() if not all(rs))
    issues += [f"window {f} failed although the smaller window {p} passed" for p in passed for f in failed if p < f]
    return issues


def _search(case_json, initial_window, harness, floor, lo, hi, settings) -> dict[str, Any]:
    initial = int(initial_window)
    if initial <= 0:
        raise ValueError("initial_window must be a positive number of tokens")
    lo = floor if lo is None else int(lo)
    hi = settings["max_window"] if hi is None else int(hi)
    resolution, growth, confirm = settings["resolution"], settings["growth_factor"], settings["confirm_probes"]
    results: dict[int, list[bool]] = {}
    details: list[dict[str, Any]] = []

    def run(window: int, purpose: str) -> bool:
        if window < lo:
            raise AssertionError(f"GACO never probes below the floor ({window} < {lo})")
        detail = {**_run_probe(case_json, window, harness), "purpose": purpose}
        results.setdefault(window, []).append(detail["passed"])
        details.append(detail)
        return detail["passed"]

    out: dict[str, Any] = {
        "floor": floor, "initial_window": initial, "lo": lo, "hi": hi, "resolution": resolution,
        "initial_probed": False, "initial_passed": None,
    }
    if lo > hi:
        return {**out, "direction": "none", "found": False, "suggested_window": None,
                "stop_reason": f"the floor ({lo}) is above the largest window GACO may try ({hi})",
                "probes": [], "probe_details": [], "violations": [], "monotonic": True, "widened": False}

    ok: int | None = None
    bad: int | None = None
    if initial < lo:
        direction = "up"     # below the measured floor: under-provisioned by measurement, not probed
        if run(lo, "floor"):
            ok = lo
        else:
            bad = lo
    else:
        out["initial_probed"] = True
        out["initial_passed"] = run(initial, "initial")
        if out["initial_passed"]:
            ok = initial
            if initial <= lo:
                direction = "none"
            else:
                direction = "down"
                if run(lo, "floor"):
                    ok = lo
                else:
                    bad = lo
        else:
            direction, bad = "up", initial
    if ok is None:          # upward: grow geometrically until a window passes or hi fails
        window = bad
        while ok is None and window < hi:
            window = _grow(window, growth, hi)
            if run(window, "grow"):
                ok = window
            else:
                bad = window
    if ok is not None and bad is not None:
        while ok - bad > resolution:
            middle = (bad + ok) // 2
            if run(middle, "bisect"):
                ok = middle
            else:
                bad = middle
    candidate = ok

    # Monotonicity guard: the suggestion must have passed on every run (it is re-run `confirm`
    # times) and must lie above every window that ever failed. A flip widens upward; it is
    # never averaged away or trusted.
    suggested: int | None = None
    while results:
        failed = [w for w, rs in results.items() if not all(rs)]
        top_fail = max(failed, default=None)
        clean = sorted(w for w, rs in results.items() if all(rs) and (top_fail is None or w > top_fail))
        if clean:
            if len(results[clean[0]]) < 1 + confirm:
                run(clean[0], "confirm")
                continue
            suggested = clean[0]
            break
        if top_fail >= hi:
            break           # even hi failed: stop, do not guess
        run(_grow(top_fail, growth, hi), "widen")
    violations = _monotonicity(results)
    stop_reason = None if suggested is not None else (
        f"even the largest window tried ({hi}) failed" if hi in results else "no window passed")
    return {
        **out,
        "direction": direction,
        "found": suggested is not None,
        "suggested_window": suggested,
        "search_candidate": candidate,
        "widened": suggested is not None and candidate is not None and suggested > candidate,
        "stop_reason": stop_reason,
        "probes": [(d["window"], d["passed"]) for d in details],
        "probe_details": details,
        "violations": violations,
        "monotonic": not violations,
    }


def smallest_safe_window(
    case_json: Mapping[str, Any],
    initial_window: int,
    *,
    harness: Harness | None = None,
    lo: int | None = None,
    hi: int | None = None,
) -> dict[str, Any]:
    """Smallest window that passes, pivoting on the client's ``initial_window``.

    ``lo`` defaults to the measured floor (never probed below); ``hi`` to the model's advertised
    window, or ``harness.gaco.max_window`` if smaller. If the initial window passes, the search goes
    down; if it fails (or is below the floor), it goes up. Each probe is one real run.
    """
    harness = harness or load_harness(case_json)
    settings = _settings(harness)
    return _search(case_json, initial_window, harness, measure_floor(case_json, harness=harness), lo, hi, settings)


# ---------------------------------------------------------------- 4. orchestration
def optimize(
    case_json: Mapping[str, Any],
    initial_window: int,
    *,
    harness: Harness | None = None,
    lo: int | None = None,
    hi: int | None = None,
) -> dict[str, Any]:
    """Measure the floor, search for the smallest safe window, and report the suggestion (advisory)."""

    harness = harness or load_harness(case_json)
    original = copy.deepcopy(dict(case_json))
    settings = _settings(harness)
    floor = measure_floor_detail(case_json, harness=harness)
    search = _search(case_json, initial_window, harness, floor["floor"], lo, hi, settings)
    if dict(case_json) != original:
        raise RuntimeError("GACO altered the client's case or pipeline definition")

    suggested, initial = search["suggested_window"], search["initial_window"]
    comparison = None
    if suggested is not None:
        delta = suggested - initial
        comparison = {
            "relation": "smaller" if delta < 0 else "larger" if delta > 0 else "same",
            "delta_tokens": abs(delta),
            "percent_of_initial": round(100.0 * abs(delta) / initial, 1),
        }
    evidence_check = _evidence_check(harness)
    model_runs = len(search["probe_details"])
    model_calls = sum(d["model_calls"] for d in search["probe_details"])
    if suggested is not None:
        held = sum(1 for window, passed in search["probes"] if window == suggested and passed)
        confidence = (
            f"Confidence: the floor ({floor['floor']} tokens) is measured (exact tokenizer count, no model runs); "
            f"the suggested window ({suggested}) was found by search over {model_runs} real runs and passed on "
            f"all {held} run(s) at that window (it is re-run before it is suggested); the probe results were "
            + ("monotonic" if search["monotonic"] else
               "NOT monotonic, so the suggestion was widened conservatively above every failing window")
            + f"; evidence check: {evidence_check}."
        )
    else:
        confidence = (f"Confidence: the floor ({floor['floor']} tokens) is measured; no window passed, so there "
                      f"is no suggestion ({model_runs} real runs); evidence check: {evidence_check}.")
    return {
        "kind": "gaco-window-suggestion",
        "advisory": True,
        "model": harness.model["name"],
        "case_id": case_json["records"].get(harness.pipeline.get("run_id_field", "case_id")),
        "finding": floor["finding"],
        "floor": floor,
        "initial_window": initial,
        "suggested_window": {"tokens": suggested, "tag": "found-by-search"},
        "found": search["found"],
        "direction": search["direction"],
        "comparison": comparison,
        "search": search,
        "probes": search["probes"],
        "model_runs": model_runs,
        "model_calls": model_calls,
        "evidence_check": evidence_check,
        "confidence": confidence,
        "settings": settings,
    }


# ---------------------------------------------------------------- 5. report
NOTES = (
    "Advisory: GACO suggests a window. It does not set it, and it does not change the client's pipeline, "
    "prompts, data, or settings.",
    "The search finds where the required evidence fits and survives (a clean, monotonic fit threshold). It "
    "does not prove the model will reason over that evidence well in every case; that needs separate "
    "behavioural validation. This is a confidence-bounded suggestion, not a provable optimum.",
    "'Safe' is relative to the tested case: the suggested window fits this case and does not guarantee "
    "larger, unseen cases. With several sample cases, take the maximum of the per-case floors and "
    "suggestions across the workload before choosing a single window.",
    "These are real model runs. The fit threshold is stable, but the oracle call is probabilistic at the "
    "margins, so the full probe ladder is listed for audit.",
)


def format_report(report: Mapping[str, Any]) -> str:
    search, floor = report["search"], report["floor"]
    initial, suggested = report["initial_window"], report["suggested_window"]["tokens"]
    finding = report["finding"]
    lines = [
        f"GACO window suggestion (advisory) · {report['model']} · case {report['case_id']} · "
        f"tracked finding {finding['id']} v{finding['version']} ({finding['finding']})",
        "",
        f"Floor [measured, no model runs]: {floor['floor']} tokens = {floor['prompt_tokens']} prompt "
        f"+ {floor['output_room_tokens']} output room ({floor['output_room_source']}) + "
        f"{floor['margin_tokens']} margin, for {floor['agent']}.",
        f"Initial window (client setting): {initial} -> " + (
            ("PASS" if search["initial_passed"] else "FAIL") if search["initial_probed"]
            else "not probed (below the measured floor, so under-provisioned by measurement)"),
        f"Search: direction {search['direction']}, range [{search['lo']}, {search['hi']}], "
        f"resolution {search['resolution']} tokens.",
        "Probe ladder (each row is one real run of the clean case):",
    ]
    for i, d in enumerate(search["probe_details"], 1):
        lines.append(
            f"  {i:>2}. window {d['window']:>7} · {d['purpose']:<7} · {'PASS' if d['passed'] else 'FAIL'}"
            f" · assembled {d['assembled_tokens']} · processed {d['processed_tokens']} · truncated {d['truncated']}"
            + (f" · {d['category']}" if not d["passed"] and d["category"] else "")
        )
    if not search["probe_details"]:
        lines.append("  (none)")
    lines.append("Monotonicity: " + ("consistent with a fit threshold." if search["monotonic"] else
                                     "VIOLATED, treated as model variance, not as a fit threshold:"))
    lines += [f"  - {v}" for v in search["violations"]]
    if search.get("widened"):
        lines.append(f"  Suggestion widened from the bisection result {search['search_candidate']} to "
                     f"{suggested}, the smallest window above every failing run.")
    lines += [
        f"Model runs: {report['model_runs']} pipeline runs ({report['model_calls']} model calls, including the "
        "runtime's window-membership measurements).",
        report["confidence"],
        "",
        *(f"- {n}" for n in NOTES),
    ]
    if report["evidence_check"] == "measured-only":
        lines.append(
            "- Measured-only: this runtime cannot expose the model-visible prompt "
            "(observability.can_capture_model_visible_prompt is false), so 'fits and survives' is measured from "
            "token counts and the runtime's processed-token count, not verified against what the model saw."
        )
    lines.append("")
    if suggested is not None:
        comparison = report["comparison"]
        if comparison["relation"] == "same":
            relation = f"same as your initial setting of {initial}"
        else:
            side = "below" if comparison["relation"] == "smaller" else "above"
            relation = f"{comparison['delta_tokens']} tokens {side} your initial setting of {initial}"
        lines.append(f"Suggested context window: {suggested}  ({relation}).")
    else:
        lines.append(
            f"No safe context window was found up to {search['hi']} tokens ({search['stop_reason']}). This case "
            "may not fit this model's window at all. Reduce the per-case evidence, or use a model with a larger "
            "context window."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- 6. CLI
def _main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="GACO: suggest the smallest safe context window for one case.")
    parser.add_argument("--definition", default="client_pipelines/healthcare_pipeline.json")
    parser.add_argument("--records", default="harness_cases/healthcare_contraindication.json")
    parser.add_argument("--initial-window", type=int, required=True, help="the client's current context window")
    parser.add_argument("--json", help="also write the full report (with the probe ladder) to this file")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with open(args.records, encoding="utf-8") as handle:
        case = case_json_for(args.definition, json.load(handle))
    harness = load_harness(case)
    report = optimize(case, args.initial_window, harness=harness)
    print(format_report(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)


if __name__ == "__main__":
    _main()


__all__ = [
    "format_report",
    "measure_floor",
    "measure_floor_detail",
    "optimize",
    "probe",
    "smallest_safe_window",
]
