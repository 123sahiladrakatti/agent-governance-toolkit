"""NeMo Guardrails adapter for agent-output content checks.

NeMo is an optional, non-deterministic signal producer. Its verdict is consumed
by Action Governance; it does not authorize actions or determine AML correctness.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict


RAIL_NAME = "self_check_output"
_RAILS = None


class NemoVerdict(TypedDict):
    passed: bool
    rail: str
    reason: str


class NemoSignal(NemoVerdict):
    status: str


def _load_rails():
    from nemoguardrails import LLMRails, RailsConfig

    config_path = Path(__file__).resolve().parent / "nemo_spike" / "config"
    return LLMRails(RailsConfig.from_path(str(config_path)))


def _response_text(result: object) -> str:
    response = getattr(result, "response", result)
    if isinstance(response, dict):
        response = response.get("response", response)
    if isinstance(response, list):
        return "\n".join(
            str(message.get("content", ""))
            for message in response
            if isinstance(message, dict) and message.get("role") == "assistant"
        )
    return str(response)


def nemo_check(agent_output: str) -> NemoVerdict:
    """Run the configured NeMo output rail on an existing agent message.

    NeMo is loaded lazily so demos and tests remain usable without the optional
    package or a live Ollama service. A missing runtime is explicitly marked as
    unavailable; callers must surface that status rather than treat it as a pass.
    """
    global _RAILS

    try:
        if _RAILS is None:
            _RAILS = _load_rails()
        result = _RAILS.generate(
            messages=[{"role": "assistant", "content": agent_output}],
            options={"rails": {"dialog": False, "output": True}},
        )
        checked_output = _response_text(result).strip()
        if checked_output == agent_output.strip():
            return {"passed": True, "rail": RAIL_NAME, "reason": "NeMo output rail allowed the content."}
        return {
            "passed": False,
            "rail": RAIL_NAME,
            "reason": "NeMo output rail blocked or changed the agent content.",
        }
    except Exception:
        return {"passed": True, "rail": "unavailable", "reason": "nemo not running"}


def summarize_nemo_checks(checks: list[dict[str, object]]) -> tuple[NemoSignal, bool]:
    """Summarize per-agent signals without hiding missing or skipped checks."""
    executed = [check for check in checks if check.get("rail") != "skipped"]
    failures = [check for check in executed if not check.get("passed", False)]
    unavailable = [check for check in executed if check.get("rail") == "unavailable"]

    if failures:
        check = failures[0]
        signal: NemoSignal = {
            "passed": False,
            "rail": str(check.get("rail", RAIL_NAME)),
            "reason": str(check.get("reason", "NeMo flagged agent content.")),
        }
    elif unavailable:
        signal = {
            "passed": True,
            "rail": "unavailable",
            "reason": f"NeMo did not run for {len(unavailable)} agent output(s).",
        }
    elif not executed:
        signal = {
            "passed": True,
            "rail": "skipped",
            "reason": "No authorized agent output was available for NeMo.",
        }
    else:
        signal = {
            "passed": True,
            "rail": RAIL_NAME,
            "reason": "NeMo allowed all authorized agent outputs.",
        }

    signal["status"] = (
        "flagged" if not signal["passed"] else
        "passed" if signal["rail"] == RAIL_NAME else signal["rail"]
    )
    return signal, bool(executed) and not unavailable