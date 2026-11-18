"""Advisory long-running context-loss projection for a customer pipeline.

This module is the projective sibling of the window-sizing harness in
``instrumentation.py``.  It measures a short sequence of real runs, estimates how
occupied context grows as retained history/memory accumulates, locates the first
trace-integrity failure by synthesizing larger starting states, and validates the
estimated onset with accelerated fast-forward runs.

The result is an extrapolation, never a guarantee.  Synthesized state is an
accelerated proxy for elapsed operation; this module observes and warns but does
not change a client's window, prompt, retention, or compaction settings.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Mapping, Protocol, Sequence

from instrumentation import Harness, HarnessConfigError, case_json_for, load_harness, run_case

Trace = list[dict[str, Any]]
Verdict = dict[str, Any]
RunOracle = Callable[..., tuple[Trace, Verdict]]


class AccumulatedStateAdapter(Protocol):
    """Customer/runtime adapter for retained state.

    Implementations must measure and synthesize the same value accepted by the
    configured pipeline's ``accumulated_state_argument``.
    """

    def empty(self) -> Any: ...

    def token_count(self, state: Any) -> int: ...

    def advance(
        self,
        state: Any,
        case: Mapping[str, Any],
        trace: Trace,
        verdict: Verdict,
        run_index: int,
    ) -> Any: ...

    def synthesize(self, target_tokens: int, exemplar: Any) -> Any: ...


class TextAccumulatedStateAdapter:
    """Generic text-memory adapter backed by the customer's declared tokenizer.

    ``advance`` retains a compact per-run record made only from observed customer
    records, agent decisions, and the governance verdict. ``synthesize`` repeats
    that measured memory shape to reach a requested size; it does not add a
    domain-specific fact or alter the pipeline.
    """

    def __init__(self, harness: Harness) -> None:
        self.harness = harness

    def empty(self) -> str:
        return ""

    def token_count(self, state: Any) -> int:
        if not state:
            return 0
        return len(self.harness.tokenizer.encode(str(state), add_special_tokens=False).ids)

    def advance(
        self,
        state: Any,
        case: Mapping[str, Any],
        trace: Trace,
        verdict: Verdict,
        run_index: int,
    ) -> str:
        entry = {
            "run_index": run_index,
            "records": case.get("records"),
            "agent_decisions": {
                record.get("agent"): record.get("decision")
                for record in trace
                if record.get("decision") is not None
            },
            "governance": {
                "passed": verdict.get("passed"),
                "category": verdict.get("category"),
            },
        }
        encoded = json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)
        return f"{state}\n{encoded}".lstrip()

    def synthesize(self, target_tokens: int, exemplar: Any) -> str:
        if target_tokens <= 0:
            return ""
        seed = str(exemplar or '{"retained_memory":"measured-run"}')
        # Repeating the observed state shape is an accelerated proxy, not a claim
        # that these exact records occurred. Find the closest token count without
        # assuming a tokens-per-character ratio for the customer's tokenizer.
        corpus = seed
        while self.token_count(corpus) < target_tokens:
            corpus = f"{corpus}\n{corpus}"
        low, high = 0, len(corpus)
        while high - low > 1:
            middle = (low + high) // 2
            if self.token_count(corpus[:middle]) < target_tokens:
                low = middle
            else:
                high = middle
        options = [corpus[:low], corpus[:high]]
        return min(options, key=lambda value: abs(self.token_count(value) - target_tokens))


@dataclass(frozen=True)
class GrowthObservation:
    run_index: int
    occupied_tokens: int
    accumulated_tokens_before: int
    accumulated_increment_tokens: int
    passed: bool


@dataclass(frozen=True)
class LinearGrowthFit:
    sample_runs: int
    intercept: float | None
    tokens_per_run: float | None
    r_squared: float | None
    residual_standard_error: float | None
    intercept_interval: tuple[float, float] | None
    slope_interval: tuple[float, float] | None
    confidence: float
    adequate: bool
    quality: str


@dataclass(frozen=True)
class ThresholdProbe:
    accumulated_tokens: int
    occupied_tokens: int
    passed: bool
    truncated: bool | None
    mechanism: str | None
    category: str | None


@dataclass(frozen=True)
class LossOnsetThreshold:
    occupied_tokens: int
    accumulated_tokens: int
    just_below_occupied_tokens: int
    just_below_accumulated_tokens: int
    dominant_mechanism: str
    probes: tuple[ThresholdProbe, ...]


@dataclass(frozen=True)
class CrossoverProjection:
    estimated_run: float
    run_range: tuple[float, float]
    elapsed_seconds_range: tuple[float, float] | None
    elapsed_range: tuple[str, str] | None
    confidence: float


@dataclass(frozen=True)
class FastForwardValidation:
    early_runs_clean: bool
    early_run_count: int
    just_below_passed: bool
    just_below_occupied_tokens: int
    fast_forward_passed: bool
    fast_forward_occupied_tokens: int
    mechanism_observed: str
    crossover_confirmed: bool
    accelerated_proxy: bool = True


@dataclass(frozen=True)
class LongRunProjection:
    window: int
    observations: tuple[GrowthObservation, ...]
    growth_fit: LinearGrowthFit
    threshold: LossOnsetThreshold | None
    projection: CrossoverProjection | None
    validation: FastForwardValidation | None
    headline: str
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _model_record(trace: Trace) -> dict[str, Any]:
    records = [record for record in trace if record.get("model_call") and record.get("assembled_tokens") is not None]
    if not records:
        raise HarnessConfigError("long-run projection needs at least one observed model call with assembled_tokens")
    return max(records, key=lambda record: int(record["assembled_tokens"]))


def _invoke(
    oracle: RunOracle,
    case: Mapping[str, Any],
    window: int,
    state: Any,
    harness: Harness | None,
) -> tuple[Trace, Verdict]:
    if harness is None:
        return oracle(case, window, state)
    return oracle(case, window, state, harness=harness)


def _student_critical(confidence: float, degrees_of_freedom: int) -> float:
    """Small-sample t approximation (Cornish-Fisher), avoiding a SciPy dependency."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if degrees_of_freedom <= 0:
        return math.inf
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    df = float(degrees_of_freedom)
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) / (384.0 * df**3)
    )


def fit_linear_growth(
    observations: Sequence[GrowthObservation],
    *,
    confidence: float = 0.95,
    minimum_r_squared: float = 0.80,
) -> LinearGrowthFit:
    """Ordinary least-squares occupied tokens vs. one-based run index."""

    count = len(observations)
    if count < 3:
        return LinearGrowthFit(
            count, None, None, None, None, None, None, confidence, False,
            "insufficient: at least three measured runs are needed to assess a trend",
        )
    xs = [float(observation.run_index) for observation in observations]
    ys = [float(observation.occupied_tokens) for observation in observations]
    x_bar, y_bar = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - x_bar) ** 2 for x in xs)
    if sxx == 0:
        return LinearGrowthFit(
            count, None, None, None, None, None, None, confidence, False,
            "poor: run indices contain no variation",
        )
    slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / sxx
    intercept = y_bar - slope * x_bar
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    residual_sum = sum(value**2 for value in residuals)
    total_sum = sum((y - y_bar) ** 2 for y in ys)
    r_squared = 1.0 if total_sum == 0 and residual_sum == 0 else (
        0.0 if total_sum == 0 else max(0.0, 1.0 - residual_sum / total_sum)
    )
    residual_error = math.sqrt(residual_sum / (count - 2))
    critical = _student_critical(confidence, count - 2)
    slope_margin = critical * residual_error / math.sqrt(sxx)
    intercept_margin = critical * residual_error * math.sqrt(1.0 / count + x_bar**2 / sxx)
    adequate = slope > 0 and r_squared >= minimum_r_squared
    if slope <= 0:
        quality = "poor: measured occupied context is not increasing"
    elif r_squared < minimum_r_squared:
        quality = f"poor: linear R² {r_squared:.3f} is below {minimum_r_squared:.3f}"
    else:
        quality = f"adequate linear fit (R² {r_squared:.3f})"
    return LinearGrowthFit(
        count,
        intercept,
        slope,
        r_squared,
        residual_error,
        (intercept - intercept_margin, intercept + intercept_margin),
        (slope - slope_margin, slope + slope_margin),
        confidence,
        adequate,
        quality,
    )


def _mechanism(trace: Trace, verdict: Verdict) -> str:
    overflow, degradation = 0, 0
    for record in trace:
        if not record.get("model_call"):
            continue
        assembled, processed = record.get("assembled_tokens"), record.get("processed_tokens")
        truncated = bool(record.get("truncated") or record.get("overflow_error"))
        if assembled is not None and processed is not None and assembled > processed:
            truncated = True
        if truncated:
            overflow += 1
            continue
        required = record.get("required_evidence") or []
        present = record.get("evidence_present_in_prompt") or {}
        visible = record.get("evidence_visible") or {}
        used = record.get("evidence_used") or {}
        lost_for_agent = (verdict.get("evidence_not_carried") or {}).get(record.get("agent"), [])
        evidence_available = any(visible.get(item) is True for item in required) or (
            not record.get("model_visibility_confirmed")
            and any(present.get(item) is True for item in required)
        )
        evidence_unused = any(used.get(item) is False or item in lost_for_agent for item in required)
        if evidence_available and evidence_unused:
            degradation += 1
    if verdict.get("category") == "runtime-overflow":
        overflow = max(overflow, 1)
    if overflow > degradation:
        return "overflow"
    if degradation > overflow:
        return "degradation"
    if overflow and degradation:
        return "mixed"
    return "indeterminate"


def _probe(
    oracle: RunOracle,
    case: Mapping[str, Any],
    window: int,
    state: Any,
    adapter: AccumulatedStateAdapter,
    harness: Harness | None,
) -> tuple[ThresholdProbe, Trace, Verdict]:
    trace, verdict = _invoke(oracle, case, window, state, harness)
    record = _model_record(trace)
    mechanism = None if verdict.get("passed") else _mechanism(trace, verdict)
    probe = ThresholdProbe(
        accumulated_tokens=adapter.token_count(state),
        occupied_tokens=int(record["assembled_tokens"]),
        passed=bool(verdict.get("passed")),
        truncated=record.get("truncated"),
        mechanism=mechanism,
        category=verdict.get("category"),
    )
    return probe, trace, verdict


def _find_threshold(
    oracle: RunOracle,
    case: Mapping[str, Any],
    window: int,
    adapter: AccumulatedStateAdapter,
    exemplar_state: Any,
    growth_step: int,
    resolution: int,
    maximum_occupied_tokens: int,
    harness: Harness | None,
) -> LossOnsetThreshold | None:
    cache: dict[int, tuple[ThresholdProbe, Trace, Verdict]] = {}

    def at(target: int) -> tuple[ThresholdProbe, Trace, Verdict]:
        state = adapter.synthesize(max(0, target), exemplar_state)
        actual = adapter.token_count(state)
        if actual not in cache:
            cache[actual] = _probe(oracle, case, window, state, adapter, harness)
        return cache[actual]

    baseline = at(0)
    if not baseline[0].passed:
        return LossOnsetThreshold(
            baseline[0].occupied_tokens,
            0,
            baseline[0].occupied_tokens,
            0,
            baseline[0].mechanism or "indeterminate",
            (baseline[0],),
        )

    low = 0
    high = max(1, growth_step)
    while True:
        candidate = at(high)[0]
        if not candidate.passed:
            break
        if (
            candidate.occupied_tokens >= maximum_occupied_tokens
            or candidate.accumulated_tokens >= maximum_occupied_tokens
        ):
            return None
        low = high
        high *= 2

    while high - low > max(1, resolution):
        middle = (low + high) // 2
        candidate = at(middle)[0]
        if candidate.passed:
            low = middle
        else:
            high = middle

    below, onset = at(low)[0], at(high)[0]
    ordered = tuple(sorted((value[0] for value in cache.values()), key=lambda item: item.accumulated_tokens))
    return LossOnsetThreshold(
        onset.occupied_tokens,
        onset.accumulated_tokens,
        below.occupied_tokens,
        below.accumulated_tokens,
        onset.mechanism or "indeterminate",
        ordered,
    )


def _project(
    fit: LinearGrowthFit,
    threshold: LossOnsetThreshold,
    cadence_seconds: float | None,
) -> CrossoverProjection | None:
    if not fit.adequate or fit.tokens_per_run is None or fit.intercept is None:
        return None
    slope_low, slope_high = fit.slope_interval or (fit.tokens_per_run, fit.tokens_per_run)
    intercept_low, intercept_high = fit.intercept_interval or (fit.intercept, fit.intercept)
    if slope_high <= 0 or slope_low <= 0:
        return None
    point = (threshold.occupied_tokens - fit.intercept) / fit.tokens_per_run
    earliest = (threshold.just_below_occupied_tokens - intercept_high) / slope_high
    latest = (threshold.occupied_tokens - intercept_low) / slope_low
    run_range = (max(1.0, min(earliest, latest)), max(1.0, max(earliest, latest)))
    seconds_range = None
    elapsed_range = None
    if cadence_seconds is not None:
        if cadence_seconds <= 0:
            raise ValueError("cadence_seconds must be positive")
        seconds_range = tuple(max(0.0, (run - 1.0) * cadence_seconds) for run in run_range)
        elapsed_range = tuple(str(timedelta(seconds=round(value))) for value in seconds_range)
    return CrossoverProjection(point, run_range, seconds_range, elapsed_range, fit.confidence)


def project_long_running_context(
    cases: Sequence[Mapping[str, Any]],
    window: int,
    *,
    oracle: RunOracle = run_case,
    harness: Harness | None = None,
    state_adapter: AccumulatedStateAdapter | None = None,
    cadence_seconds: float | None = None,
    confidence: float = 0.95,
    minimum_r_squared: float = 0.80,
    resolution: int | None = None,
    maximum_occupied_tokens: int | None = None,
) -> LongRunProjection:
    """Measure, project, and fast-forward long-running context accumulation.

    ``cases`` are the few available real/sample runs, in execution order. ``window``
    is the measured/configured client window and is deliberately required rather
    than embedded in this algorithm. No fixed run horizon is used: bracketing grows
    from the measured per-run state increment and stops at the customer model's
    advertised limit (or ``maximum_occupied_tokens`` supplied by the caller).
    """

    if not cases:
        raise ValueError("at least one measured case is required")
    if window <= 0:
        raise ValueError("window must be positive")
    harness = harness or (load_harness(cases[0]) if oracle is run_case else None)
    if state_adapter is None:
        if harness is None:
            raise ValueError("a state_adapter is required when no Harness is supplied")
        state_adapter = TextAccumulatedStateAdapter(harness)
    if resolution is None:
        resolution = int((harness.config.get("sizing", {}) if harness else {}).get("resolution", 1))
    if maximum_occupied_tokens is None:
        if harness is None:
            raise ValueError("maximum_occupied_tokens is required when no Harness is supplied")
        maximum_occupied_tokens = int(harness.model["advertised_window"])

    state = state_adapter.empty()
    observations: list[GrowthObservation] = []
    measured: list[tuple[Mapping[str, Any], Trace, Verdict]] = []
    for run_index, case in enumerate(cases, start=1):
        state_before = state_adapter.token_count(state)
        trace, verdict = _invoke(oracle, case, window, state, harness)
        occupied = int(_model_record(trace)["assembled_tokens"])
        next_state = state_adapter.advance(state, case, trace, verdict, run_index)
        increment = state_adapter.token_count(next_state) - state_before
        if increment < 0:
            raise ValueError("accumulated-state adapter reduced retained state during measurement")
        observations.append(GrowthObservation(
            run_index, occupied, state_before, increment, bool(verdict.get("passed")),
        ))
        measured.append((case, trace, verdict))
        state = next_state

    fit = fit_linear_growth(
        observations,
        confidence=confidence,
        minimum_r_squared=minimum_r_squared,
    )
    positive_increments = [item.accumulated_increment_tokens for item in observations
                           if item.accumulated_increment_tokens > 0]
    growth_step = round(statistics.median(positive_increments)) if positive_increments else 0
    threshold: LossOnsetThreshold | None = None
    projection: CrossoverProjection | None = None
    validation: FastForwardValidation | None = None
    caveats = [
        f"Projection is extrapolated from {len(observations)} measured runs and depends on the growth model holding.",
        "Fast-forward uses synthesized accumulated context as an accelerated proxy; it is not elapsed months of operation.",
        "The module is advisory: it does not change the client pipeline, prompts, window, retention, or compaction.",
        "The estimated onset applies to the supplied workload and does not guarantee unseen cases.",
    ]

    if growth_step <= 0:
        caveats.append("No positive accumulated-memory increment was measured, so an onset sweep was not projected.")
    else:
        # Use the largest measured real-run prompt as the representative stress case.
        stress_index = max(range(len(observations)), key=lambda index: observations[index].occupied_tokens)
        threshold = _find_threshold(
            oracle,
            measured[stress_index][0],
            window,
            state_adapter,
            state,
            growth_step,
            resolution,
            maximum_occupied_tokens,
            harness,
        )
        if threshold is None:
            caveats.append(
                "No loss onset was observed within the measured model limit; no crossover is claimed."
            )
        else:
            projection = _project(fit, threshold, cadence_seconds)
            below_state = state_adapter.synthesize(threshold.just_below_accumulated_tokens, state)
            onset_state = state_adapter.synthesize(threshold.accumulated_tokens, state)
            below_probe, _, _ = _probe(
                oracle, measured[stress_index][0], window, below_state, state_adapter, harness,
            )
            onset_probe, onset_trace, onset_verdict = _probe(
                oracle, measured[stress_index][0], window, onset_state, state_adapter, harness,
            )
            observed_mechanism = _mechanism(onset_trace, onset_verdict)
            validation = FastForwardValidation(
                early_runs_clean=all(item.passed for item in observations),
                early_run_count=len(observations),
                just_below_passed=below_probe.passed,
                just_below_occupied_tokens=below_probe.occupied_tokens,
                fast_forward_passed=onset_probe.passed,
                fast_forward_occupied_tokens=onset_probe.occupied_tokens,
                mechanism_observed=observed_mechanism,
                crossover_confirmed=(
                    all(item.passed for item in observations)
                    and below_probe.passed
                    and not onset_probe.passed
                    and observed_mechanism == threshold.dominant_mechanism
                ),
            )
            if not validation.crossover_confirmed:
                caveats.append(
                    "The accelerated validation did not reproduce a clean-below/failing-at-onset crossover; "
                    "the projection is not confirmed."
                )

    if projection and threshold and all(item.passed for item in observations):
        estimated = max(1, math.ceil(projection.estimated_run))
        lower = max(1, math.floor(projection.run_range[0]))
        upper = max(lower, math.ceil(projection.run_range[1]))
        headline = (
            f"No loss observed in the first {len(observations)} runs. At the observed growth rate, "
            f"context loss is projected to begin around run {estimated} (range {lower}–{upper}), "
            f"via {threshold.dominant_mechanism}."
        )
    elif not all(item.passed for item in observations):
        first = next(item.run_index for item in observations if not item.passed)
        headline = (
            f"Context loss was observed in the measured runs, first at run {first}; "
            "a future no-loss crossover is therefore not claimed."
        )
    elif not fit.adequate:
        headline = (
            f"No loss observed in the first {len(observations)} runs, but the measured growth fit is "
            f"{fit.quality}; no reliable crossover range is projected."
        )
    else:
        headline = (
            f"No loss observed in the first {len(observations)} runs. No loss onset was observed within "
            "the measured search limit, so no crossover is claimed."
        )

    return LongRunProjection(
        window,
        tuple(observations),
        fit,
        threshold,
        projection,
        validation,
        headline,
        tuple(caveats),
    )


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Project long-running context loss from a few measured runs.")
    parser.add_argument("--definition", default="client_pipelines/healthcare_pipeline.json")
    parser.add_argument("--records", nargs="+", required=True, help="Run records in chronological order")
    parser.add_argument("--window", type=int, help="Client window; defaults to harness.model.current_window")
    parser.add_argument("--cadence-seconds", type=float)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--minimum-r-squared", type=float, default=0.80)
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()

    cases = []
    for records_path in args.records:
        with open(records_path, encoding="utf-8") as handle:
            cases.append(case_json_for(args.definition, json.load(handle)))
    harness = load_harness(cases[0])
    report = project_long_running_context(
        cases,
        args.window or int(harness.model["current_window"]),
        harness=harness,
        cadence_seconds=args.cadence_seconds,
        confidence=args.confidence,
        minimum_r_squared=args.minimum_r_squared,
    )
    payload = json.dumps(report.to_dict(), indent=2, default=str)
    print(report.headline)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    _main()


__all__ = [
    "AccumulatedStateAdapter",
    "CrossoverProjection",
    "FastForwardValidation",
    "GrowthObservation",
    "LinearGrowthFit",
    "LongRunProjection",
    "LossOnsetThreshold",
    "TextAccumulatedStateAdapter",
    "ThresholdProbe",
    "fit_linear_growth",
    "project_long_running_context",
]
