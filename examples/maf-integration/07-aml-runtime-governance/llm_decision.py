"""Real-LLM fan-in decision agent with a deliberately small context window.

Domain-agnostic machinery shared by the AML and healthcare reference pipelines.
A domain supplies a ``DecisionSpec`` (its branch reports, decision rules, and
case-file wording); this module builds the case file, makes the real Ollama
call, parses the reply, and measures what fit in the model's window.

The agent receives the branch reports as a case file in arrival order: the
spec's early reports first, then a routine activity log that accumulates while
the other branches work, then the late reports and the decision task. When the
case file is longer than ``num_ctx``, Ollama keeps only the most recent tokens,
so an early report genuinely falls out of the agent's context.

Nothing about the loss is planted. ``LLMDecision`` records what the agent itself
reported consuming (this becomes the decision step's
``received_findings_by_parent``), Ollama's own telemetry on which reports were
inside the kept window, and the raw model output.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Mapping

import httpx

from system_description import TraceStep

_NOT_PRESENT = {"", "not-present", "notpresent", "missing", "unknown", "none", "n/a", "na", "absent", "not-available"}


class LLMDecisionError(RuntimeError):
    """The LLM decision agent could not run or its output could not be parsed."""

    def __init__(self, message: str, raw_output: str | None = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


@dataclass(frozen=True)
class LLMDecisionConfig:
    model: str = "qwen2.5:3b"
    num_ctx: int = 1024
    activity_log_lines: int = 40
    temperature: float = 0.0
    seed: int = 7
    num_predict: int = 160
    base_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    timeout_s: float = 300.0
    measure_window: bool = True

    @classmethod
    def from_env(cls, prefix: str, default_model: str, **overrides) -> "LLMDecisionConfig":
        """Config from ``{prefix}_MODEL`` / ``{prefix}_NUM_CTX`` (e.g. ``AML_LLM_NUM_CTX``)."""
        values = {
            "model": os.environ.get(f"{prefix}_MODEL", default_model),
            "num_ctx": int(os.environ.get(f"{prefix}_NUM_CTX", "1024")),
        }
        return cls(**{**values, **overrides})


@dataclass(frozen=True)
class ReportSpec:
    agent: str        # parent agent that emitted the report
    key: str          # JSON key the decision agent uses for it
    label: str        # human label in the case file
    early: bool       # arrives before the activity log (and so is the one at risk)


@dataclass(frozen=True)
class DecisionSpec:
    """What a domain tells the generic decision agent."""

    file_title: str                                   # e.g. "CASE FILE"
    agent_role: str                                   # e.g. "the AML decision agent"
    decision_noun: str                                # e.g. "case disposition"
    reports: tuple[ReportSpec, ...]
    rules: str                                        # decision rules, ending without a period
    dispositions: tuple[str, ...]
    log_title: str                                    # e.g. "[CASE ACTIVITY LOG]"
    log_events: tuple[str, ...]
    record_id: Callable[[object], str]
    detail: Callable[[TraceStep, object], str]        # descriptive detail for one report
    log_ref: Callable[[object, int], str]             # suffix for activity-log line i ("" for none)
    status_hint: str = "<status or not-present>"      # placeholder shown for each report in the reply format


@dataclass(frozen=True)
class LLMDecision:
    disposition: str
    consumed: dict[str, str]                 # parent agent -> finding the agent says it used
    not_consumed: tuple[str, ...]            # parents the agent reported as not present
    self_report: dict[str, str]              # raw self-reported statuses by report key
    in_window: dict[str, bool | None]        # parent agent -> inside Ollama's kept window (None = unmeasured)
    kept_tokens: int                         # prompt tokens Ollama actually evaluated
    prompt_truncated: bool                   # Ollama dropped the start of the prompt
    model: str
    num_ctx: int
    raw_output: str
    prompt: str


def _activity_log(spec: DecisionSpec, records: object, lines: int) -> str:
    return "\n".join(
        f"- 2026-01-{5 + i // 24:02d} {8 + (i % 10):02d}:{(i * 7) % 60:02d} "
        f"{spec.log_events[i % len(spec.log_events)]}{spec.log_ref(records, i)}."
        for i in range(lines)
    )


def _report_line(spec: DecisionSpec, report: ReportSpec, step: TraceStep, records: object) -> str:
    return (
        f"[REPORT {step.agent} v{step.finding_version}] {report.label}: "
        f"{step.finding.upper()}. {spec.detail(step, records)}."
    )


def build_case_file(
    spec: DecisionSpec,
    records: object,
    branches: Mapping[str, TraceStep],
    config: LLMDecisionConfig,
) -> tuple[str, dict[str, str]]:
    """The decision agent's input, plus each report's line for window measurement."""

    lines = {report.agent: _report_line(spec, report, branches[report.agent], records) for report in spec.reports}
    keys = ", ".join(f'"{report.key}": "{spec.status_hint}"' for report in spec.reports)
    prompt = "\n".join([
        f"{spec.file_title} {spec.record_id(records)} - branch reports and case activity, in arrival order.",
        *(lines[report.agent] for report in spec.reports if report.early),
        spec.log_title,
        _activity_log(spec, records, config.activity_log_lines),
        *(lines[report.agent] for report in spec.reports if not report.early),
        "",
        f"DECISION TASK: You are {spec.agent_role}. Decide the {spec.decision_noun} using only the "
        "branch reports that appear in this case file.",
        f"Rules: {spec.rules}. If a report does not appear in the case file, "
        'record its status as "not-present" and do not guess it.',
        f'Reply with JSON only: {{{keys}, "disposition": "{"|".join(spec.dispositions)}"}}',
    ])
    return prompt, lines


def _normalize_status(value: object) -> str:
    return re.sub(r"[\s_]+", "-", str(value if value is not None else "").strip().lower()).strip(".")


def _normalize_disposition(value: object, dispositions: tuple[str, ...]) -> str | None:
    text = re.sub(r"[\s\-]+", "_", str(value or "").strip().upper()).strip("_.\"'")
    return text if text in dispositions else None


def _json_objects(text: str) -> list[str]:
    """Balanced ``{...}`` substrings, so prose or code fences around the JSON are tolerated."""
    found, depth, start = [], 0, None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                found.append(text[start:index + 1])
    return found


def parse_decision_output(text: str, spec: DecisionSpec) -> tuple[str, dict[str, str]]:
    """Parse the agent's reply into ``(disposition, self-reported statuses)``.

    Accepts JSON with any casing/spacing, JSON wrapped in prose or code fences,
    answers split across several JSON objects, and ``disposition: X`` lines.
    Raises ``LLMDecisionError`` rather than guessing when no single disposition
    can be identified.
    """

    report_keys = tuple(report.key for report in spec.reports)
    # Models sometimes split the answer across several objects, e.g. statuses in one
    # and the disposition in the next, so merge every JSON object in the reply.
    merged: dict[str, object] = {}
    for candidate in _json_objects(text):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            merged.update({str(key).strip().lower(): value for key, value in data.items()})
    disposition = _normalize_disposition(merged.get("disposition"), spec.dispositions)
    if disposition is not None:
        return disposition, {key: _normalize_status(merged.get(key)) for key in report_keys}

    longest = max(len(d) for d in spec.dispositions) + 2
    for value in re.findall(rf"disposition\W{{0,4}}\s*([A-Za-z_\- ]{{4,{longest}}})", text, re.I):
        disposition = _normalize_disposition(value.split()[0] if value.split() else value, spec.dispositions)
        if disposition is not None:
            statuses = {
                key: _normalize_status(match.group(1)) if (match := re.search(
                    rf"{key}\W{{0,4}}\s*([A-Za-z\-_ ]{{2,24}})", text, re.I)) else ""
                for key in report_keys
            }
            return disposition, statuses

    mentioned = {d for d in spec.dispositions if re.search(rf"\b{d.replace('_', '[_ -]?')}\b", text, re.I)}
    if len(mentioned) == 1:
        return mentioned.pop(), {key: "" for key in report_keys}
    raise LLMDecisionError(
        f"Could not parse a single disposition ({' / '.join(spec.dispositions)}) from the decision agent's output.",
        raw_output=text,
    )


def _chat(config: LLMDecisionConfig, prompt: str, num_predict: int) -> dict:
    try:
        response = httpx.post(
            f"{config.base_url.rstrip('/')}/api/chat",
            json={
                "model": config.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "num_ctx": config.num_ctx,
                    "temperature": config.temperature,
                    "seed": config.seed,
                    "num_predict": num_predict,
                },
            },
            timeout=config.timeout_s,
        )
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise LLMDecisionError(
            f"Ollama call failed for model {config.model!r} at {config.base_url}: {exc}"
        ) from exc


def _window_membership(
    config: LLMDecisionConfig, prompt: str, lines: Mapping[str, str], kept: int
) -> dict[str, bool | None]:
    """Is each report inside the kept window? Measured with Ollama at the same ``num_ctx``.

    A report is in the window iff the text from its line to the end of the prompt
    fits in the ``kept`` tokens. Measuring a text longer than ``num_ctx`` truncates
    it to ``kept`` as well, so a measurement equal to ``kept`` is ambiguous; it is
    resolved by measuring the text *after* the line: if that alone fills the
    window, the line itself cannot fit.
    """

    def measure(text: str) -> int:
        return int(_chat(config, text, num_predict=1)["prompt_eval_count"])

    membership: dict[str, bool | None] = {}
    for agent, line in lines.items():
        start = prompt.index(line)
        with_line = measure(prompt[start:])
        if with_line < kept:
            membership[agent] = True
        elif with_line > kept:
            membership[agent] = False
        else:
            after_line = prompt[start + len(line):].lstrip("\n")
            membership[agent] = False if measure(after_line) >= kept else True
    return membership


def run_llm_decision(
    spec: DecisionSpec,
    records: object,
    branches: Mapping[str, TraceStep],
    config: LLMDecisionConfig | None = None,
) -> LLMDecision:
    """Run the decision agent as a real Ollama call and record what it saw vs. decided."""

    config = config or LLMDecisionConfig()
    prompt, lines = build_case_file(spec, records, branches, config)
    reply = _chat(config, prompt, num_predict=config.num_predict)
    raw = str(reply.get("message", {}).get("content", ""))
    disposition, self_report = parse_decision_output(raw, spec)
    kept = int(reply.get("prompt_eval_count") or 0)
    in_window = (
        _window_membership(config, prompt, lines, kept) if config.measure_window
        else {agent: None for agent in lines}
    )
    truncated = False in in_window.values() or kept == config.num_ctx // 2 + 2

    consumed: dict[str, str] = {}
    not_consumed: list[str] = []
    for report in spec.reports:
        status = self_report.get(report.key, "")
        if status in _NOT_PRESENT:
            not_consumed.append(report.agent)
        else:
            consumed[report.agent] = status
    return LLMDecision(
        disposition=disposition,
        consumed=consumed,
        not_consumed=tuple(not_consumed),
        self_report=self_report,
        in_window=in_window,
        kept_tokens=kept,
        prompt_truncated=truncated,
        model=config.model,
        num_ctx=config.num_ctx,
        raw_output=raw,
        prompt=prompt,
    )


__all__ = [
    "DecisionSpec",
    "LLMDecision",
    "LLMDecisionConfig",
    "LLMDecisionError",
    "ReportSpec",
    "build_case_file",
    "parse_decision_output",
    "run_llm_decision",
]
