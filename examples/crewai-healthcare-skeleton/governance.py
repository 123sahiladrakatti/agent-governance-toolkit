"""Baseline runtime governance: ACS + Rego at four intervention points.

This is the *conventional* governance layer for the two-agent system -- a
per-call policy gate, evaluated with `EnforcementMode.EVALUATE_ONLY`. It
observes and records; it never blocks. That is deliberate: the point of this
step is to watch where a standard stateless gate fires, and where it stays
silent, across a real multi-agent run.

What this layer can see, by construction:
  - one payload at a time, at one intervention point
  - the tool name, when the manifest declares it

What it cannot see, also by construction:
  - that a payload crossed from one agent to another
  - anything that happened on an earlier call, or an earlier turn
  - how many agents have touched the same record

Those three blind spots are the subject of the next step. Nothing here tries to
fix them.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path

from agent_control_specification import AgentControl, EnforcementMode

from run_trace import GOV, TRACE, truncate

INPUT = "input"
PRE_TOOL_CALL = "pre_tool_call"
POST_TOOL_CALL = "post_tool_call"
OUTPUT = "output"

# Decisions that a real enforcing deployment would have acted on.
BLOCKING_DECISIONS = {"deny", "escalate"}


@dataclass(frozen=True)
class Verdict:
    """One governance decision, tagged with enough context to group later."""

    turn: int
    point: str
    subject: str  # tool name, or "task" / "final answer"
    agent: str
    decision: str
    reason: str

    @property
    def would_block(self) -> bool:
        return self.decision in BLOCKING_DECISIONS


class _AsyncBridge:
    """Runs ACS's async API from sync CrewAI callbacks.

    A dedicated loop on its own thread, rather than `asyncio.run`, because these
    calls happen inside CrewAI's execution and we cannot assume the calling
    thread is free of a running loop.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="acs-bridge", daemon=True
        )
        self._thread.start()

    def run(self, coro, timeout: float = 30.0):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


class Governance:
    """Evaluates ACS intervention points and records every verdict."""

    def __init__(self, manifest: Path) -> None:
        self.manifest = manifest
        self.control = AgentControl.from_path(str(manifest))
        self.verdicts: list[Verdict] = []
        self._bridge = _AsyncBridge()

    # -- the four intervention points -------------------------------------
    def check_input(self, body: str, *, agent: str) -> Verdict:
        return self._evaluate(
            INPUT, {"input": {"body": body}}, agent=agent, subject="task"
        )

    def check_pre_tool_call(self, tool: str, args: object, *, agent: str) -> Verdict:
        payload = {"tool_call": {"name": tool, "args": args}}
        return self._evaluate(PRE_TOOL_CALL, payload, agent=agent, subject=tool)

    def check_post_tool_call(self, tool: str, result: object, *, agent: str) -> Verdict:
        payload = {
            "tool_call": {"name": tool},
            "tool_result": {"value": str(result)},
        }
        return self._evaluate(POST_TOOL_CALL, payload, agent=agent, subject=tool)

    def check_output(self, content: str, *, agent: str) -> Verdict:
        return self._evaluate(
            OUTPUT,
            {"response": {"content": content}},
            agent=agent,
            subject="final answer",
        )

    # -- plumbing ----------------------------------------------------------
    def _evaluate(
        self, point: str, payload: dict, *, agent: str, subject: str
    ) -> Verdict:
        try:
            result = self._bridge.run(
                self.control.evaluate_intervention_point(
                    point, payload, mode=EnforcementMode.EVALUATE_ONLY
                )
            )
            decision = result.verdict.decision.value
            reason = str(result.verdict.reason or "")
        except Exception as exc:  # a policy failure must not kill the run
            decision, reason = "error", f"{type(exc).__name__}: {exc}"

        verdict = Verdict(
            turn=TRACE.turn,
            point=point,
            subject=subject,
            agent=agent,
            decision=decision,
            reason=reason,
        )
        self.verdicts.append(verdict)
        marker = "WOULD BLOCK" if verdict.would_block else decision.upper()
        TRACE.record(
            agent,
            GOV,
            f"{point} {subject} -> {marker}",
            truncate(reason, 160),
        )
        return verdict

    def for_turn(self, turn: int) -> list[Verdict]:
        return [verdict for verdict in self.verdicts if verdict.turn == turn]

    def close(self) -> None:
        self._bridge.close()


def tally(verdicts: list[Verdict]) -> dict[str, int]:
    """Count verdicts by decision, so callers can print a stable summary."""
    counts: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict.decision] = counts.get(verdict.decision, 0) + 1
    return counts


def format_tally(verdicts: list[Verdict]) -> str:
    counts = tally(verdicts)
    if not counts:
        return "no evaluations"
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
