"""Real-LLM AML decision agent with a deliberately small context window.

Only the decision agent is an LLM; the branch agents stay deterministic. The
agent receives the three branch reports as a case file in arrival order: the
screening report first, then the routine case activity log that accumulates
while the other branches work, then the transaction and KYC reports and the
decision task. A local Ollama model runs it with a small ``num_ctx``. When the
case file is longer than that window, Ollama keeps only the most recent tokens,
so the early screening report genuinely falls out of the agent's context.

Nothing about the loss is planted in the trace. ``LLMDecision`` records:

* what the agent itself reported consuming, which becomes the decision step's
  ``received_findings_by_parent`` (a report it says was not present is left out);
* Ollama's own telemetry: how many prompt tokens it kept, and for each report
  whether its position was inside that kept window;
* the raw model output and the parsed disposition.

Settings validated with ``experiments/context_loss_probe.py``: qwen2.5:3b (and
llama3.1:8b), ``num_ctx=1024``, 40 activity-log lines, temperature 0.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import httpx

from aml.system import KYC_AGENT, SANCTIONS_AGENT, TRANSACTION_AGENT
from domain import Alert, Disposition
from system_description import TraceStep

DISPOSITIONS: tuple[Disposition, ...] = ("CLEAR", "ESCALATE", "FILE_SAR")
REPORT_KEYS = {SANCTIONS_AGENT: "screening", TRANSACTION_AGENT: "transactions", KYC_AGENT: "kyc"}
REPORT_LABELS = {
    SANCTIONS_AGENT: "Sanctions screening",
    TRANSACTION_AGENT: "Structuring analysis",
    KYC_AGENT: "KYC profile",
}
# Arrival order in the case file: screening reports first, the others after the activity log.
EARLY_REPORTS = (SANCTIONS_AGENT,)
LATE_REPORTS = (TRANSACTION_AGENT, KYC_AGENT)
_NOT_PRESENT = {"", "not-present", "notpresent", "missing", "unknown", "none", "n/a", "na", "absent", "not-available"}
_ACTIVITY_EVENTS = (
    "statement batch imported; no action required",
    "document checklist reviewed; all items on file",
    "branch confirmation received for deposit slip",
    "queue rebalanced; case remains with current team",
    "customer correspondence archived",
    "ledger reconciliation completed for the period",
)


class LLMDecisionError(RuntimeError):
    """The LLM decision agent could not run or its output could not be parsed."""

    def __init__(self, message: str, raw_output: str | None = None) -> None:
        super().__init__(message)
        self.raw_output = raw_output


@dataclass(frozen=True)
class LLMDecisionConfig:
    model: str = field(default_factory=lambda: os.environ.get("AML_LLM_MODEL", "qwen2.5:3b"))
    num_ctx: int = field(default_factory=lambda: int(os.environ.get("AML_LLM_NUM_CTX", "1024")))
    activity_log_lines: int = 40
    temperature: float = 0.0
    seed: int = 7
    num_predict: int = 160
    base_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    timeout_s: float = 300.0
    measure_window: bool = True


@dataclass(frozen=True)
class LLMDecision:
    disposition: Disposition
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


def _activity_log(alert: Alert, lines: int) -> str:
    transactions = alert.transactions or ()
    rows = []
    for i in range(lines):
        ref = transactions[i % len(transactions)] if transactions else None
        where = f" (ref {ref.id}, {ref.branch})" if ref else ""
        rows.append(
            f"- 2026-01-{5 + i // 24:02d} {8 + (i % 10):02d}:{(i * 7) % 60:02d} "
            f"{_ACTIVITY_EVENTS[i % len(_ACTIVITY_EVENTS)]}{where}."
        )
    return "\n".join(rows)


def _report_detail(step: TraceStep, alert: Alert) -> str:
    """Descriptive detail built only from what the branch emitted (and the case it ran on)."""
    customer = alert.customer
    if step.agent == SANCTIONS_AGENT:
        return f"'{customer.name}' {'is' if step.flagged else 'is not'} on the sanctions watchlist"
    if step.agent == TRANSACTION_AGENT:
        return (
            f"{len(step.used_amounts)} sub-threshold cash deposits totalling "
            f"${step.used_aggregate:,.0f}"
        )
    return (
        f"profile on file: {customer.stated_occupation}, "
        f"expected ${customer.expected_monthly_cash_volume:,.0f}/mo cash"
    )


def _report_line(step: TraceStep, alert: Alert) -> str:
    return (
        f"[REPORT {step.agent} v{step.finding_version}] {REPORT_LABELS[step.agent]}: "
        f"{step.finding.upper()}. {_report_detail(step, alert)}."
    )


def build_case_file(alert: Alert, branches: Mapping[str, TraceStep], config: LLMDecisionConfig) -> tuple[str, dict[str, str]]:
    """The decision agent's input, plus each report's line for window measurement."""

    lines = {agent: _report_line(branches[agent], alert) for agent in REPORT_KEYS}
    prompt = "\n".join([
        f"CASE FILE {alert.alert_id} - branch reports and case activity, in arrival order.",
        *(lines[agent] for agent in EARLY_REPORTS),
        "[CASE ACTIVITY LOG]",
        _activity_log(alert, config.activity_log_lines),
        *(lines[agent] for agent in LATE_REPORTS),
        "",
        "DECISION TASK: You are the AML decision agent. Decide the case disposition using only the "
        "branch reports that appear in this case file.",
        "Rules: FILE_SAR if sanctions screening is HIT or transactions show STRUCTURING; "
        "ESCALATE if KYC is INCOMPLETE; otherwise CLEAR. If a report does not appear in the case file, "
        'record its status as "not-present" and do not guess it.',
        'Reply with JSON only: {"screening": "<status or not-present>", "transactions": "<status or not-present>", '
        '"kyc": "<status or not-present>", "disposition": "CLEAR|ESCALATE|FILE_SAR"}',
    ])
    return prompt, lines


def _normalize_status(value: object) -> str:
    return re.sub(r"[\s_]+", "-", str(value if value is not None else "").strip().lower()).strip(".")


def _normalize_disposition(value: object) -> Disposition | None:
    text = re.sub(r"[\s\-]+", "_", str(value or "").strip().upper()).strip("_.\"'")
    return text if text in DISPOSITIONS else None  # type: ignore[return-value]


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


def parse_decision_output(text: str) -> tuple[Disposition, dict[str, str]]:
    """Parse the agent's reply into ``(disposition, self-reported statuses)``.

    Accepts JSON with any casing/spacing, JSON wrapped in prose or code fences,
    answers split across several JSON objects, and ``disposition: X`` lines.
    Raises ``LLMDecisionError`` rather than guessing when no single disposition
    can be identified.
    """

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
    disposition = _normalize_disposition(merged.get("disposition"))
    if disposition is not None:
        statuses = {key: _normalize_status(merged.get(key)) for key in REPORT_KEYS.values()}
        return disposition, statuses

    labelled = re.findall(r"disposition\W{0,4}\s*([A-Za-z_\- ]{4,12})", text, re.I)
    for value in labelled:
        disposition = _normalize_disposition(value.split()[0] if value.split() else value)
        if disposition is not None:
            statuses = {
                key: _normalize_status(match.group(1)) if (match := re.search(
                    rf"{key}\W{{0,4}}\s*([A-Za-z\-_ ]{{2,20}})", text, re.I)) else ""
                for key in REPORT_KEYS.values()
            }
            return disposition, statuses

    mentioned = {d for d in DISPOSITIONS if re.search(rf"\b{d.replace('_', '[_ -]?')}\b", text, re.I)}
    if len(mentioned) == 1:
        return mentioned.pop(), {key: "" for key in REPORT_KEYS.values()}
    raise LLMDecisionError(
        "Could not parse a single disposition (CLEAR / ESCALATE / FILE_SAR) from the decision agent's output.",
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
    alert: Alert,
    branches: Mapping[str, TraceStep],
    config: LLMDecisionConfig | None = None,
) -> LLMDecision:
    """Run the decision agent as a real Ollama call and record what it saw vs. decided."""

    config = config or LLMDecisionConfig()
    prompt, lines = build_case_file(alert, branches, config)
    reply = _chat(config, prompt, num_predict=config.num_predict)
    raw = str(reply.get("message", {}).get("content", ""))
    disposition, self_report = parse_decision_output(raw)
    kept = int(reply.get("prompt_eval_count") or 0)
    in_window = (
        _window_membership(config, prompt, lines, kept) if config.measure_window
        else {agent: None for agent in lines}
    )
    truncated = False in in_window.values() or kept == config.num_ctx // 2 + 2

    consumed: dict[str, str] = {}
    not_consumed: list[str] = []
    for agent, key in REPORT_KEYS.items():
        status = self_report.get(key, "")
        if status in _NOT_PRESENT:
            not_consumed.append(agent)
        else:
            consumed[agent] = status
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


def describe_lost(decision: LLMDecision, emitted: Mapping[str, TraceStep]) -> Sequence[str]:
    """Parents whose emitted finding the agent did not consume as emitted."""
    return tuple(
        agent for agent in REPORT_KEYS
        if decision.consumed.get(agent) != _normalize_status(emitted[agent].finding)
    )


__all__ = [
    "DISPOSITIONS",
    "LLMDecision",
    "LLMDecisionConfig",
    "LLMDecisionError",
    "build_case_file",
    "describe_lost",
    "parse_decision_output",
    "run_llm_decision",
]
