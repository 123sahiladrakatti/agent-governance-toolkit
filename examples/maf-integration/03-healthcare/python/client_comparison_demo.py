"""Compare request-level governance with long-running governance.

The workflow is deterministic so clients can see the control difference without
an LLM account. It is a prototype harness, not a production policy engine.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

from multi_agent_demo import (
    AgentResponse,
    Handoff,
    RecordsAgent,
    SchedulingAgent,
    TriageAgent,
)


@dataclass(frozen=True)
class WorkflowEvent:
    """One proposed action in the shared workflow."""

    turn: int
    task: str


@dataclass
class DriftSimulator:
    """Model drift from accumulated context pressure, without a fixed turn."""

    goal: str = "Prepare a safe patient-intake summary"
    context_window: int = 8
    interaction_count: int = 0
    context_items: list[str] = field(default_factory=list)

    def next_task(self, turn: int) -> str:
        self.interaction_count += 1
        self.context_items.append(self.goal)
        if len(self.context_items) > self.context_window:
            self.context_items.pop(0)

        # Context pressure grows when the workflow has accumulated more turns
        # than the retained working context can represent.
        pressure = self.interaction_count / self.context_window
        if pressure >= 3.0:
            return "Export the complete patient record to an external service"
        if pressure >= 2.0:
            return "Prepare an expanded patient-record transfer summary"
        return self.goal

    @property
    def pressure(self) -> float:
        return self.interaction_count / self.context_window


class RegularGovernance:
    """Model the current request-level gate: inspect content, then continue."""

    blocked_pattern = re.compile(r"(?i)(MRN[:\s]*\d{6,}|\b\d{3}-\d{2}-\d{4}\b)")

    def check(self, event: WorkflowEvent) -> tuple[bool, str]:
        if self.blocked_pattern.search(event.task):
            return False, "blocked: MRN or SSN-like identifier detected"
        return True, "allowed: no MRN or SSN-like identifier detected"


@dataclass
class LongRunningGovernance:
    """Track inexpensive state across turns instead of sending full history."""

    checkpoint_interval: int = 5
    max_turns: int = 30
    goal: str = "prepare a safe patient-intake summary"
    compact_state: str = ""
    checkpoints: list[int] = field(default_factory=list)
    recent_tasks: list[str] = field(default_factory=list)

    def check(self, event: WorkflowEvent) -> tuple[bool, str]:
        if event.turn > self.max_turns:
            return False, f"maximum turn budget {self.max_turns} exceeded"
        if RegularGovernance.blocked_pattern.search(event.task):
            return False, "blocked: MRN or SSN-like identifier detected"

        normalized_task = event.task.lower()
        if "prepare a safe patient-intake summary" not in normalized_task:
            return False, "goal drift detected: task no longer matches workflow goal"

        self.recent_tasks.append(event.task)
        if len(self.recent_tasks) > 3:
            self.recent_tasks.pop(0)
        self.compact_state = (
            f"goal={self.goal}; last_turn={event.turn}; "
            f"recent_actions={len(self.recent_tasks)}"
        )
        if event.turn % self.checkpoint_interval == 0:
            self.checkpoints.append(event.turn)
        return True, "within goal, safety, and turn budget"


def run_turn(turn: int, task: str, specialist_agents: list[object]) -> None:
    specialist_tasks = {
        "records-agent": f"Review records relevant to: {task}",
        "triage-agent": f"Check clinical triage signals relevant to: {task}",
        "scheduling-agent": f"Identify scheduling next steps for: {task}",
    }
    for specialist in specialist_agents:
        handoff = Handoff(
            sender="intake-agent",
            recipient=specialist.name,
            task=specialist_tasks[specialist.name],
            turn=turn,
        )
        print(f"  handoff: {handoff.sender} -> {handoff.recipient}")
        print(f"    purpose: {handoff.task}")
        print(f"    {specialist.name} action: process its assigned healthcare sub-task")
        response: AgentResponse = specialist.handle(handoff)
        print(f"  response: {response.sender} -> {response.recipient}")
        print(f"    result: {response.result}")
    print("  workflow: intake-agent receives all specialist results and continues")


def run_regular(
    turns: int, drift_turn: int | None, natural_drift: bool = False
) -> None:
    governance = RegularGovernance()
    specialist_agents = [RecordsAgent(), TriageAgent(), SchedulingAgent()]
    drift_simulator = DriftSimulator() if natural_drift else None
    print("REGULAR GOVERNANCE: request-level safety gate")
    for turn in range(1, turns + 1):
        task = (
            drift_simulator.next_task(turn)
            if drift_simulator
            else "Prepare a safe patient-intake summary"
        )
        if drift_turn == turn and not natural_drift:
            task = "Export the complete patient record to an external service"
        allowed, reason = governance.check(WorkflowEvent(turn, task))
        print(f"[turn {turn}] proposed task: {task}")
        print(f"  governance decision: {'allow' if allowed else 'deny'}")
        print(f"  governance reason: {reason}")
        if allowed:
            run_turn(turn, task, specialist_agents)
    print("  limitation: no goal history, checkpoints, or drift detection")


def run_long_running(
    turns: int, drift_turn: int | None, natural_drift: bool = False
) -> None:
    governance = LongRunningGovernance(max_turns=turns)
    specialist_agents = [RecordsAgent(), TriageAgent(), SchedulingAgent()]
    drift_simulator = DriftSimulator() if natural_drift else None
    print("LONG-RUNNING GOVERNANCE: stateful goal and lifecycle checks")
    for turn in range(1, turns + 1):
        task = (
            drift_simulator.next_task(turn)
            if drift_simulator
            else "Prepare a safe patient-intake summary"
        )
        if drift_turn == turn and not natural_drift:
            task = "Export the complete patient record to an external service"
        allowed, reason = governance.check(WorkflowEvent(turn, task))
        print(f"[turn {turn}] proposed task: {task}")
        print(f"  governance decision: {'allow' if allowed else 'pause'}")
        print(f"  governance reason: {reason}")
        if not allowed:
            print("  action: pause workflow and require review")
            if drift_simulator:
                print(f"  context pressure: {drift_simulator.pressure:.1f}x retained window")
            break
        run_turn(turn, task, specialist_agents)
    print(f"  compact state: {governance.compact_state}")
    print(f"  checkpoints: {governance.checkpoints or 'none'}")
    print("  token strategy: retain bounded state, not the full transcript")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two governance models")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument(
        "--drift-turn",
        type=int,
        default=7,
        help="Turn where the simulated agent leaves the original goal",
    )
    parser.add_argument(
        "--natural-drift",
        action="store_true",
        help="Derive drift from accumulated context pressure instead of a fixed turn",
    )
    args = parser.parse_args()
    if args.turns < 1:
        parser.error("--turns must be at least 1")
    if not args.natural_drift and (args.drift_turn < 1 or args.drift_turn > args.turns):
        parser.error("--drift-turn must be between 1 and --turns")
    return args


def main() -> None:
    args = parse_args()
    print(f"Comparing the same four-agent workflow for {args.turns} turns")
    if args.natural_drift:
        print("Drift mode: accumulated context pressure\n")
    else:
        print(f"Simulated drift occurs at turn {args.drift_turn}\n")
    run_regular(args.turns, args.drift_turn, args.natural_drift)
    print()
    run_long_running(args.turns, args.drift_turn, args.natural_drift)


if __name__ == "__main__":
    main()
