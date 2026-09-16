"""Run a governance-free four-agent healthcare workflow.

This is step 1 of the experiment: establish agent interaction before adding
policy enforcement, drift detection, recovery, or persistence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Handoff:
    """A message passed from one agent to another."""

    sender: str
    recipient: str
    task: str
    turn: int


@dataclass(frozen=True)
class AgentResponse:
    """A response returned by an agent after processing a handoff."""

    sender: str
    recipient: str
    result: str
    turn: int


class RecordsAgent:
    """Handle the records sub-task delegated by the intake agent."""

    name = "records-agent"

    def handle(self, handoff: Handoff) -> AgentResponse:
        result = (
            f"Record review complete for turn {handoff.turn}: "
            "no urgent follow-up was found in the supplied summary."
        )
        return AgentResponse(
            sender=self.name,
            recipient=handoff.sender,
            result=result,
            turn=handoff.turn,
        )


class TriageAgent:
    """Review the intake summary for clinical escalation signals."""

    name = "triage-agent"

    def handle(self, handoff: Handoff) -> AgentResponse:
        return AgentResponse(
            sender=self.name,
            recipient=handoff.sender,
            result=(
                f"Triage review complete for turn {handoff.turn}: "
                "no escalation signal was found in the supplied summary."
            ),
            turn=handoff.turn,
        )


class SchedulingAgent:
    """Find the next operational step for the patient intake."""

    name = "scheduling-agent"

    def handle(self, handoff: Handoff) -> AgentResponse:
        return AgentResponse(
            sender=self.name,
            recipient=handoff.sender,
            result=(
                f"Scheduling review complete for turn {handoff.turn}: "
                "follow-up can be scheduled after intake review."
            ),
            turn=handoff.turn,
        )


class IntakeAgent:
    """Coordinate the patient-intake task and delegate record review."""

    name = "intake-agent"

    def __init__(self, specialist_agents: list[object]) -> None:
        self.specialist_agents = specialist_agents

    def run_turn(self, task: str, turn: int) -> None:
        specialist_tasks = {
            "records-agent": f"Review records relevant to: {task}",
            "triage-agent": f"Check clinical triage signals relevant to: {task}",
            "scheduling-agent": f"Identify scheduling next steps for: {task}",
        }
        for specialist in self.specialist_agents:
            handoff = Handoff(
                sender=self.name,
                recipient=specialist.name,
                task=specialist_tasks[specialist.name],
                turn=turn,
            )
            print(f"[turn {turn}] {handoff.sender} -> {handoff.recipient}: {handoff.task}")
            response = specialist.handle(handoff)
            print(f"[turn {turn}] {response.sender} -> {response.recipient}: {response.result}")
        print(f"[turn {turn}] {self.name}: incorporated all care-team results into the workflow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the four-agent workflow")
    parser.add_argument(
        "--turns",
        type=int,
        default=3,
        help="Number of handoff turns to execute (default: 3)",
    )
    parser.add_argument(
        "--task",
        default="Prepare a safe patient-intake summary",
        help="Shared task for the four agents",
    )
    args = parser.parse_args()
    if args.turns < 1:
        parser.error("--turns must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    intake_agent = IntakeAgent(
        specialist_agents=[RecordsAgent(), TriageAgent(), SchedulingAgent()]
    )

    print(f"Starting four-agent workflow for {args.turns} turn(s)")
    for turn in range(1, args.turns + 1):
        intake_agent.run_turn(task=args.task, turn=turn)
    print("Workflow complete: no governance or policy checks were applied")


if __name__ == "__main__":
    main()