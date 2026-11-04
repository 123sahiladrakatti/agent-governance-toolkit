#!/usr/bin/env python3
"""Isolated, real-Ollama AML context-window experiment.

This script does not import or modify the application pipeline. It compares a short
full-context prompt, a long prompt at a constrained context allocation, and the exact
same long prompt with context restored. Decision and recall calls are separate.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DISPOSITIONS = ("CLEAR", "ESCALATE", "FILE_SAR", "INSUFFICIENT_EVIDENCE")
CONDITIONS = ("full_context", "constrained_context", "restored_context")
SCREENING_MARKER = "AML_SCREENING_FACT_7F3A"
POLICY = """DECISION POLICY (apply exactly):
1. An explicit positive watchlist screening finding requires ESCALATE.
2. FILE_SAR requires a separately stated SAR policy condition. No such condition exists in this case.
3. CLEAR is permitted only when an explicit negative watchlist screening finding, low transaction risk, and current KYC are all retained.
4. If the screening finding is unavailable, return INSUFFICIENT_EVIDENCE. Never silently treat missing evidence as a negative result.
"""
SCREENING = f"""<{SCREENING_MARKER}>
SCREENING_STATUS=HIT
SCREENING_RISK=HIGH
SCREENING_ENTITY=WATCHLIST_MATCH
</{SCREENING_MARKER}>"""
TRANSACTION = """<TRANSACTION_FINDING>
TRANSACTION_RISK=LOW
TRANSACTION_PATTERN=NORMAL
</TRANSACTION_FINDING>"""
KYC = """<KYC_FINDING>
KYC_STATUS=CURRENT
KYC_RISK=LOW
</KYC_FINDING>"""
DECISION_SUFFIX = """Return one decision for synthetic case AML-CTX-001.
Use only the evidence retained in this request. Do not invent absent findings.
Before deciding, scan the retained request for the tagged screening-finding block and apply its actual value if present.
Return JSON with disposition, rationale, and evidence_cited.
"""


@dataclass
class CallResult:
    execution_status: str
    raw_response: str | None = None
    parsed: dict[str, Any] | None = None
    parse_status: str = "NOT_APPLICABLE"
    error: str | None = None
    prompt_eval_count: int | None = None
    output_eval_count: int | None = None
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    context_window_requested: int | None = None
    context_window_effective: int | None = None


class OllamaClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def version(self) -> dict[str, Any]:
        return self._request("/api/version")

    def show(self, model: str) -> dict[str, Any]:
        return self._request("/api/show", {"model": model})

    def tags(self) -> dict[str, Any]:
        return self._request("/api/tags")

    def process_context(self, model: str) -> int | None:
        try:
            processes = self._request("/api/ps").get("models", [])
        except Exception:
            return None
        for process in processes:
            if process.get("name") == model or process.get("model") == model:
                value = process.get("context_length")
                return int(value) if isinstance(value, (int, float)) else None
        return None

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        num_ctx: int,
        temperature: float,
        seed: int,
        schema: dict[str, Any],
        num_predict: int,
    ) -> CallResult:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            "keep_alive": "5m",
            "options": {
                "num_ctx": num_ctx,
                "temperature": temperature,
                "seed": seed,
                "num_predict": num_predict,
            },
        }
        try:
            response = self._request("/api/generate", payload)
            raw = str(response.get("response", ""))
            return CallResult(
                execution_status="SUCCESS" if raw.strip() else "EMPTY_RESPONSE",
                raw_response=raw,
                prompt_eval_count=_int_or_none(response.get("prompt_eval_count")),
                output_eval_count=_int_or_none(response.get("eval_count")),
                total_duration_ns=_int_or_none(response.get("total_duration")),
                load_duration_ns=_int_or_none(response.get("load_duration")),
                context_window_requested=num_ctx,
                context_window_effective=self.process_context(model),
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return CallResult(
                execution_status="HTTP_ERROR",
                error=f"HTTP {exc.code}: {body}",
                context_window_requested=num_ctx,
            )
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            return CallResult(
                execution_status="RUNTIME_ERROR",
                error=f"{type(exc).__name__}: {exc}",
                context_window_requested=num_ctx,
            )


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def filler(target_chars: int) -> str:
    lines: list[str] = []
    index = 1
    while sum(len(line) + 1 for line in lines) < target_chars:
        lines.append(
            f"REFERENCE_NOTE_{index:04d}: routine synthetic account chronology; "
            f"administrative checksum={index * 7919 % 100000:05d}; this note contains no branch finding."
        )
        index += 1
    return "\n".join(lines)


def place_finding(finding: str, padding: str, placement: str) -> str:
    if placement == "early":
        return "\n\n".join((finding, padding))
    if placement == "late":
        return "\n\n".join((padding, finding))
    midpoint = len(padding) // 2
    split = padding.find("\n", midpoint)
    split = midpoint if split < 0 else split
    return "\n\n".join((padding[:split], finding, padding[split:]))


def decision_prompt(*, long: bool, input_length: int, placement: str) -> str:
    evidence = SCREENING if not long else place_finding(SCREENING, filler(input_length), placement)
    return "\n\n".join((
        "SYNTHETIC AML DECISION EXPERIMENT. All identifiers are fictional.",
        evidence,
        TRANSACTION,
        KYC,
        POLICY,
        DECISION_SUFFIX,
    ))


def recall_prompt(*, long: bool, input_length: int, placement: str) -> str:
    evidence = SCREENING if not long else place_finding(SCREENING, filler(input_length), placement)
    return "\n\n".join((
        "SYNTHETIC AML EVIDENCE RECALL PROBE. Do not make a disposition.",
        evidence,
        TRANSACTION,
        KYC,
        "Report SCREENING_STATUS from an explicit SCREENING_FINDING. If no explicit screening finding is retained, report UNKNOWN. Return JSON only.",
    ))


DECISION_SCHEMA = {
    "type": "object",
    "required": ["disposition", "rationale", "evidence_cited"],
    "properties": {
        "disposition": {"type": "string", "enum": list(DISPOSITIONS)},
        "rationale": {"type": "string"},
        "evidence_cited": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}
RECALL_SCHEMA = {
    "type": "object",
    "required": ["screening_status", "evidence_quote"],
    "properties": {
        "screening_status": {"type": "string", "enum": ["HIT", "NO_HIT", "UNKNOWN"]},
        "evidence_quote": {"type": "string"},
    },
    "additionalProperties": False,
}


def parse_json_result(call: CallResult, required: tuple[str, ...]) -> CallResult:
    if call.execution_status != "SUCCESS" or not call.raw_response:
        return call
    try:
        parsed = json.loads(call.raw_response)
    except json.JSONDecodeError as exc:
        call.parse_status = "INVALID_JSON"
        call.error = f"JSONDecodeError: {exc}"
        return call
    if not isinstance(parsed, dict) or any(key not in parsed for key in required):
        call.parse_status = "INVALID_STRUCTURE"
        call.error = f"Required fields missing: {required}"
        return call
    call.parsed = parsed
    call.parse_status = "PARSED"
    return call


def cited_screening(parsed: dict[str, Any] | None) -> bool:
    if not parsed:
        return False
    text = " ".join(str(item) for item in parsed.get("evidence_cited", []))
    text += " " + str(parsed.get("rationale", ""))
    return bool(re.search(r"screen|watchlist|\bhit\b", text, re.IGNORECASE))


def classify_visibility(
    *,
    prompt_has_finding: bool,
    recall: CallResult,
    constrained_token_reduction: bool,
) -> str:
    if not prompt_has_finding:
        return "NOT_SUPPLIED"
    status = (recall.parsed or {}).get("screening_status")
    if status == "HIT":
        return "VERIFIED_BY_SEPARATE_RECALL"
    if status in {"UNKNOWN", "NO_HIT"} and constrained_token_reduction:
        return "INFERRED_NOT_RETAINED"
    if recall.execution_status != "SUCCESS" or recall.parse_status != "PARSED":
        return "UNKNOWN_RECALL_FAILED"
    return "UNKNOWN"


def call_to_dict(call: CallResult) -> dict[str, Any]:
    return asdict(call)


def safe_rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def summarize_condition(trials: list[dict[str, Any]]) -> dict[str, Any]:
    dispositions = Counter(
        trial["decision"].get("parsed", {}).get("disposition", "UNPARSED")
        if trial["decision"].get("parsed") else "UNPARSED"
        for trial in trials
    )
    recall = Counter(
        trial["recall"].get("parsed", {}).get("screening_status", "UNPARSED")
        if trial["recall"].get("parsed") else "UNPARSED"
        for trial in trials
    )
    total = len(trials)
    return {
        "trials": total,
        "dispositions": dict(dispositions),
        "recall_statuses": dict(recall),
        "correct_escalations": dispositions["ESCALATE"],
        "incorrect_clears": dispositions["CLEAR"],
        "other_incorrect_dispositions": dispositions["FILE_SAR"],
        "abstentions": dispositions["INSUFFICIENT_EVIDENCE"],
        "parsing_failures": dispositions["UNPARSED"],
        "runtime_failures": sum(
            trial["decision"]["execution_status"] != "SUCCESS" for trial in trials
        ),
        "evidence_recall_success": recall["HIT"],
        "evidence_recall_failure": total - recall["HIT"],
        "decision_accuracy": safe_rate(dispositions["ESCALATE"], total),
        "incorrect_clear_rate": safe_rate(dispositions["CLEAR"], total),
        "evidence_recall_rate": safe_rate(recall["HIT"], total),
        "invalid_output_rate": safe_rate(dispositions["UNPARSED"], total),
    }


def markdown_report(report: dict[str, Any]) -> str:
    config = report["configuration"]
    summaries = report["summaries"]
    lines = [
        "# Real-LLM context-loss probe",
        "",
        f"**Conclusion: {report['conclusion']}**",
        "",
        "## Runtime",
        "",
        f"- Ollama: `{config.get('ollama_version')}`",
        f"- Model: `{config['model']}`",
        f"- Model identifier: `{config.get('model_identifier')}`",
        f"- Temperature: `{config['temperature']}`; seed: `{config['seed']}`",
        f"- Requested contexts: full `{config['full_num_ctx']}`, constrained `{config['num_ctx']}`, restored `{config['restored_num_ctx']}`",
        f"- Finding placement: `{config['finding_placement']}`; filler target: `{config['input_length']}` characters",
        f"- Trials per condition: `{config['trials']}`",
        "",
        "## Controlled comparison",
        "",
        "| Condition | Prompt contains screening | Recall HIT | ESCALATE | CLEAR | Abstain | Runtime/parse failures |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        summary = summaries[condition]
        failures = summary["runtime_failures"] + summary["parsing_failures"]
        lines.append(
            f"| {condition} | yes | {summary['evidence_recall_success']}/{summary['trials']} | "
            f"{summary['correct_escalations']}/{summary['trials']} | "
            f"{summary['incorrect_clears']}/{summary['trials']} | "
            f"{summary['abstentions']}/{summary['trials']} | {failures}/{summary['trials']} |"
        )
    lines.extend((
        "",
        "## Causal checks",
        "",
        f"- Long-prompt token reduction at constrained context: `{report['causal_checks']['constrained_token_reduction']}`",
        f"- Full-context accuracy: `{summaries['full_context']['correct_escalations']}/{summaries['full_context']['trials']}`",
        f"- Constrained incorrect-clear rate: `{summaries['constrained_context']['incorrect_clears']}/{summaries['constrained_context']['trials']}`",
        f"- Restoration recovery rate: `{summaries['restored_context']['correct_escalations']}/{summaries['restored_context']['trials']}`",
        "",
        "A failed recall does not directly expose Ollama's retained token sequence. When prompt-token processing is reduced and the separate recall probe fails, the artifact labels the specific finding's visibility as `INFERRED_NOT_RETAINED`, not directly verified truncation.",
        "",
        "## Integration gate",
        "",
        report["integration_gate_reason"],
        "",
    ))
    return "\n".join(lines)


def model_identifier(show: dict[str, Any], tags: dict[str, Any], model: str) -> str | None:
    for item in tags.get("models", []):
        if item.get("name") == model or item.get("model") == model:
            digest = item.get("digest")
            if isinstance(digest, str):
                return digest
    # `/api/show` does not consistently return the digest across Ollama versions.
    for key in ("digest", "model"):
        value = show.get(key)
        if isinstance(value, str):
            return value
    return None


def cli_version() -> str | None:
    try:
        result = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=10, check=False
        )
        return (result.stdout or result.stderr).strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    client = OllamaClient(args.base_url, args.timeout)
    started = utc_now()
    try:
        version = client.version().get("version")
        show = client.show(args.model)
        tags = client.tags()
        startup_error = None
    except Exception as exc:
        version, show, tags = None, {}, {}
        startup_error = f"{type(exc).__name__}: {exc}"

    short_decision = decision_prompt(long=False, input_length=0, placement=args.finding_placement)
    long_decision = decision_prompt(
        long=True, input_length=args.input_length, placement=args.finding_placement
    )
    short_recall = recall_prompt(long=False, input_length=0, placement=args.finding_placement)
    long_recall = recall_prompt(
        long=True, input_length=args.input_length, placement=args.finding_placement
    )
    prompts = {
        "full_context": {"decision": short_decision, "recall": short_recall},
        "constrained_context": {"decision": long_decision, "recall": long_recall},
        "restored_context": {"decision": long_decision, "recall": long_recall},
    }
    contexts = {
        "full_context": args.full_num_ctx,
        "constrained_context": args.num_ctx,
        "restored_context": args.restored_num_ctx,
    }
    results: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}

    if startup_error is None:
        for condition in CONDITIONS:
            for trial_index in range(args.trials):
                seed = args.seed if not args.vary_seed else args.seed + trial_index
                decision = parse_json_result(
                    client.generate(
                        model=args.model,
                        prompt=prompts[condition]["decision"],
                        num_ctx=contexts[condition],
                        temperature=args.temperature,
                        seed=seed,
                        schema=DECISION_SCHEMA,
                        num_predict=args.num_predict,
                    ),
                    ("disposition", "rationale", "evidence_cited"),
                )
                recall = parse_json_result(
                    client.generate(
                        model=args.model,
                        prompt=prompts[condition]["recall"],
                        num_ctx=contexts[condition],
                        temperature=args.temperature,
                        seed=seed,
                        schema=RECALL_SCHEMA,
                        num_predict=min(args.num_predict, 128),
                    ),
                    ("screening_status", "evidence_quote"),
                )
                results[condition].append({
                    "trial": trial_index + 1,
                    "seed": seed,
                    "decision": call_to_dict(decision),
                    "recall": call_to_dict(recall),
                    "evidence_supplied": ["screening", "transaction", "kyc"],
                    "evidence_cited_screening": cited_screening(decision.parsed),
                    "evidence_visibility": "PENDING_CAUSAL_COMPARISON",
                })

    restored_counts = [
        trial["decision"].get("prompt_eval_count")
        for trial in results["restored_context"]
        if trial["decision"].get("prompt_eval_count") is not None
    ]
    constrained_counts = [
        trial["decision"].get("prompt_eval_count")
        for trial in results["constrained_context"]
        if trial["decision"].get("prompt_eval_count") is not None
    ]
    token_reduction = bool(
        restored_counts and constrained_counts and max(constrained_counts) < min(restored_counts)
    )
    for condition in CONDITIONS:
        for trial in results[condition]:
            recall = CallResult(**trial["recall"])
            trial["evidence_visibility"] = classify_visibility(
                prompt_has_finding=True,
                recall=recall,
                constrained_token_reduction=condition == "constrained_context" and token_reduction,
            )

    summaries = {condition: summarize_condition(results[condition]) for condition in CONDITIONS}
    verified_sequence = bool(
        args.trials > 0
        and summaries["full_context"]["correct_escalations"] == args.trials
        and summaries["constrained_context"]["incorrect_clears"] > 0
        and summaries["restored_context"]["correct_escalations"] == args.trials
        and token_reduction
        and summaries["constrained_context"]["evidence_recall_success"]
            < summaries["restored_context"]["evidence_recall_success"]
    )
    conclusion = "CONTEXT LOSS VERIFIED" if verified_sequence else "CONTEXT LOSS NOT VERIFIED"
    gate_reason = (
        "The isolated experiment produced at least one incorrect CLEAR under reduced token processing, full/restored trials all escalated, and separate recall degraded. Pipeline integration may be considered only after reviewing the raw artifacts."
        if verified_sequence else
        "Stop before pipeline integration. The required reproducible full → incorrect CLEAR → restored sequence was not established by this run."
    )
    report = {
        "experiment": "real-ollama-context-loss-probe/v1",
        "started_at": started,
        "completed_at": utc_now(),
        "conclusion": conclusion,
        "integration_gate_open": verified_sequence,
        "integration_gate_reason": gate_reason,
        "configuration": {
            "base_url": args.base_url,
            "ollama_version": version,
            "ollama_cli_version": cli_version(),
            "model": args.model,
            "model_identifier": model_identifier(show, tags, args.model),
            "model_details": show.get("details"),
            "model_info": show.get("model_info"),
            "num_ctx": args.num_ctx,
            "full_num_ctx": args.full_num_ctx,
            "restored_num_ctx": args.restored_num_ctx,
            "temperature": args.temperature,
            "seed": args.seed,
            "vary_seed": args.vary_seed,
            "input_length": args.input_length,
            "finding_placement": args.finding_placement,
            "trials": args.trials,
            "num_predict": args.num_predict,
            "python": sys.version,
            "platform": platform.platform(),
            "startup_error": startup_error,
        },
        "authoritative_case": {
            "case_id": "AML-CTX-001",
            "screening": {"status": "HIT", "risk": "HIGH", "entity": "WATCHLIST_MATCH"},
            "transaction": {"risk": "LOW", "pattern": "NORMAL"},
            "kyc": {"status": "CURRENT", "risk": "LOW"},
            "policy": POLICY,
            "expected_disposition": "ESCALATE",
        },
        "controlled_inputs": {
            condition: {
                "decision_prompt": prompts[condition]["decision"],
                "recall_prompt": prompts[condition]["recall"],
                "decision_prompt_chars": len(prompts[condition]["decision"]),
                "recall_prompt_chars": len(prompts[condition]["recall"]),
                "context_window_requested": contexts[condition],
                "screening_in_constructed_prompt": SCREENING_MARKER in prompts[condition]["decision"],
            }
            for condition in CONDITIONS
        },
        "causal_checks": {
            "constrained_token_reduction": token_reduction,
            "constrained_prompt_eval_counts": constrained_counts,
            "restored_prompt_eval_counts": restored_counts,
            "specific_retained_sequence_directly_observable": False,
            "visibility_interpretation": (
                "Specific screening loss is inferred only when reduced prompt processing and separate recall failure coincide."
            ),
        },
        "summaries": summaries,
        "trials": results,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3.1:latest")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--num-ctx", type=int, default=512, help="Constrained context allocation")
    parser.add_argument("--full-num-ctx", type=int, default=2048)
    parser.add_argument("--restored-num-ctx", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vary-seed", action="store_true")
    parser.add_argument("--input-length", type=int, default=12000, help="Filler target in characters")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--finding-placement", choices=("early", "middle", "late"), default="early")
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.num_ctx, args.full_num_ctx, args.restored_num_ctx, args.trials) <= 0:
        raise SystemExit("Context sizes and trial count must be positive")
    report = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "context_loss_probe.json"
    md_path = args.output_dir / "context_loss_probe.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({
        "conclusion": report["conclusion"],
        "integration_gate_open": report["integration_gate_open"],
        "json": str(json_path),
        "markdown": str(md_path),
        "summaries": report["summaries"],
    }, indent=2))
    return 0 if report["configuration"]["startup_error"] is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
