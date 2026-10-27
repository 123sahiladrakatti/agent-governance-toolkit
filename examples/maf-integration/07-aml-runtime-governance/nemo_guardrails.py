"""NeMo Guardrails adapter for agent-output content checks.

NeMo is an optional, non-deterministic signal producer. Its verdict is consumed
by Action Governance; it does not authorize actions or determine AML correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict


RAIL_NAME = "self_check_output"
_RAILS = None
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "nemo_spike" / "config"


@dataclass(frozen=True)
class NemoRuntimeConfig:
    """Customer-supplied NeMo runtime configuration.

    ``model`` replaces the configured main model name. ``prompts`` maps NeMo
    task names (for example ``self_check_output``) to prompt content.
    """

    config_path: Path = DEFAULT_CONFIG_PATH
    model: str | None = None
    prompts: dict[str, str] = field(default_factory=dict)
    review_instruction: str = "Review the following agent output."


class NemoVerdict(TypedDict):
    passed: bool
    rail: str
    reason: str


class NemoSignal(NemoVerdict):
    status: str


def _load_rails(settings: NemoRuntimeConfig | None = None):
    from nemoguardrails import LLMRails, RailsConfig

    settings = settings or NemoRuntimeConfig()
    config = RailsConfig.from_path(str(settings.config_path))
    if settings.model is not None:
        main_model = next((model for model in config.models if getattr(model, "type", None) == "main"), None)
        if main_model is None:
            raise ValueError("NeMo config does not define a main model")
        main_model.model = settings.model
    for task, content in settings.prompts.items():
        prompt = next((prompt for prompt in config.prompts if getattr(prompt, "task", None) == task), None)
        if prompt is None:
            raise ValueError(f"NeMo config does not define prompt task {task!r}")
        prompt.content = content
    return LLMRails(config)


def _output_rail_decision(result: object) -> str | None:
    log = getattr(result, "log", None)
    for rail in getattr(log, "activated_rails", ()) or ():
        if getattr(rail, "type", None) != "output":
            continue
        for action in getattr(rail, "executed_actions", ()) or ():
            if getattr(action, "action_name", None) != "self_check_output":
                continue
            outcome = getattr(action, "return_value", None)
            decision = getattr(outcome, "decision", None)
            return str(getattr(decision, "value", decision)).lower() if decision is not None else None
    return None


def _check_with_rails(rails, agent_output: str, review_instruction: str) -> NemoVerdict:
    result = rails.generate(
        messages=[
            {"role": "user", "content": review_instruction},
            {"role": "assistant", "content": agent_output},
        ],
        options={
            "rails": {"dialog": False, "output": True},
            "log": {"activated_rails": True},
        },
    )
    decision = _output_rail_decision(result)
    if decision == "allow":
        return {"passed": True, "rail": RAIL_NAME, "reason": "NeMo output rail allowed the content."}
    if decision == "block":
        return {"passed": False, "rail": RAIL_NAME, "reason": "NeMo output rail blocked the content."}
    return {"passed": True, "rail": "unavailable", "reason": "NeMo output rail did not execute."}


class NemoGuardrailsChecker:
    """Callable, lazily loaded NeMo checker for one customer's settings."""

    def __init__(self, settings: NemoRuntimeConfig) -> None:
        self.settings = settings
        self._rails = None

    def __call__(self, agent_output: str) -> NemoVerdict:
        try:
            if self._rails is None:
                self._rails = _load_rails(self.settings)
            return _check_with_rails(
                self._rails,
                agent_output,
                self.settings.review_instruction,
            )
        except Exception:
            return {"passed": True, "rail": "unavailable", "reason": "nemo not running"}


def nemo_check(agent_output: str) -> NemoVerdict:
    """Run the default AML NeMo output rail on an existing agent message.

    NeMo is loaded lazily so demos and tests remain usable without the optional
    package or a live Ollama service. A missing runtime is explicitly marked as
    unavailable; callers must surface that status rather than treat it as a pass.
    """
    global _RAILS

    try:
        if _RAILS is None:
            _RAILS = _load_rails()
        return _check_with_rails(
            _RAILS,
            agent_output,
            NemoRuntimeConfig().review_instruction,
        )
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
