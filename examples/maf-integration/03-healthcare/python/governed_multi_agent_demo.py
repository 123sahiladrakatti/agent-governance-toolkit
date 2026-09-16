"""Four-agent healthcare workflow integrated with the native ACS runtime.

This demo builds the recommended layers in order:
1. four-agent workflow
2. AgentControl before every handoff
3. sender, recipient, task, and PHI rules
4. long-running goal tracking
5. checkpoints and resume after a pause
6. simple handoff and audit limits for scale
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agent_control_specification import AgentControl
from multi_agent_demo import (
    Handoff,
    RecordsAgent,
    SchedulingAgent,
    TriageAgent,
)


SPECIALIST_TASKS = {
    "records-agent": "Review records relevant to: {task}",
    "triage-agent": "Check clinical triage signals relevant to: {task}",
    "scheduling-agent": "Identify scheduling next steps for: {task}",
}
ALLOWED_RECIPIENTS = frozenset(SPECIALIST_TASKS)
PHI_PATTERN = re.compile(r"(?i)(MRN[:\s]*\d{6,}|\b\d{3}-\d{2}-\d{4}\b)")


@dataclass(frozen=True)
class HandoffRecord:
    turn: int
    sender: str
    recipient: str
    task: str


@dataclass
class GovernanceState:
    goal: str
    max_turns: int
    max_handoffs_per_turn: int
    total_handoffs: int = 0
    current_turn_handoffs: int = 0
    checkpoints: list[int] = field(default_factory=list)
    recent_tasks: list[str] = field(default_factory=list)

    def compact(self, turn: int) -> dict[str, object]:
        return {
            "goal": self.goal,
            "last_turn": turn,
            "total_handoffs": self.total_handoffs,
            "recent_tasks": self.recent_tasks[-3:],
            "checkpoints": self.checkpoints,
        }


class HealthcareHandoffPolicy:
    """Enforce identity, routing, task, PHI, and scale rules for handoffs."""

    def __init__(self, state: GovernanceState) -> None:
        self.state = state

    def evaluate(self, invocation: dict) -> dict:
        policy_input = invocation.get("input", invocation.get("policy_input", invocation))
        target = policy_input.get("policy_target", {}).get("value", {})
        if not isinstance(target, dict):
            return {"decision": "deny", "reason": "invalid_handoff_payload"}

        sender = target.get("sender")
        recipient = target.get("recipient")
        task = str(target.get("task", ""))
        turn = int(target.get("turn", 0))

        if sender != "intake-agent":
            return {"decision": "deny", "reason": "unauthorized_sender"}
        if recipient not in ALLOWED_RECIPIENTS:
            return {"decision": "deny", "reason": "unauthorized_recipient"}
        if not task or not task.startswith(SPECIALIST_TASKS[recipient].split("{")[0]):
            return {"decision": "deny", "reason": "task_not_allowed_for_recipient"}
        if PHI_PATTERN.search(task):
            return {"decision": "deny", "reason": "healthcare_phi_detected"}
        if turn > self.state.max_turns:
            return {"decision": "deny", "reason": "turn_budget_exceeded"}
        if self.state.current_turn_handoffs >= self.state.max_handoffs_per_turn:
            return {"decision": "deny", "reason": "per_turn_handoff_limit_exceeded"}
        if self.state.total_handoffs >= self.state.max_turns * self.state.max_handoffs_per_turn:
            return {"decision": "deny", "reason": "handoff_budget_exceeded"}
        if self.state.goal.lower() not in task.lower():
            return {"decision": "deny", "reason": "goal_drift_detected"}
        return {"decision": "allow", "reason": "handoff_policy_passed"}


class GovernedWorkflow:
    """Coordinate specialists and put ACS before every agent handoff."""

    def __init__(self, runtime: AgentControl, state: GovernanceState, audit_path: Path) -> None:
        self.runtime = runtime
        self.state = state
        self.audit_path = audit_path
        self.specialists = [RecordsAgent(), TriageAgent(), SchedulingAgent()]

    async def evaluate_handoff(self, handoff: Handoff) -> bool:
        payload = {
            "tool_call": {
                "name": handoff.recipient,
                "args": asdict(handoff) | {"value": handoff.task},
            }
        }
        result = await self.runtime.evaluate_intervention_point("pre_tool_call", payload)
        decision = result.verdict.decision.value
        reason = result.verdict.reason
        self._audit(
            {
                "event": "handoff_policy_check",
                "turn": handoff.turn,
                "sender": handoff.sender,
                "recipient": handoff.recipient,
                "decision": decision,
                "reason": reason,
            }
        )
        print(f"  ACS pre-handoff: {decision} ({reason})")
        return decision == "allow"

    async def run_turn(self, task: str, turn: int) -> bool:
        self.state.current_turn_handoffs = 0
        if turn % 5 == 0:
            self.save_checkpoint(turn)
        for specialist in self.specialists:
            handoff = Handoff(
                sender="intake-agent",
                recipient=specialist.name,
                task=SPECIALIST_TASKS[specialist.name].format(task=task),
                turn=turn,
            )
            print(f"[turn {turn}] proposed {handoff.sender} -> {handoff.recipient}")
            if not await self.evaluate_handoff(handoff):
                if turn > 1:
                    self.save_checkpoint(turn - 1)
                print("  action: workflow paused before specialist execution")
                return False
            response = specialist.handle(handoff)
            self.state.total_handoffs += 1
            self.state.current_turn_handoffs += 1
            print(f"  response: {response.result}")
        self.state.recent_tasks.append(task)
        self.state.recent_tasks = self.state.recent_tasks[-3:]
        return True

    def save_checkpoint(self, turn: int) -> None:
        if turn not in self.state.checkpoints:
            self.state.checkpoints.append(turn)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.audit_path.with_suffix(".checkpoint.json")
        checkpoint_path.write_text(
            json.dumps(self.state.compact(turn), indent=2), encoding="utf-8"
        )
        print(f"  checkpoint saved: turn {turn}")

    def _audit(self, event: dict[str, object]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")


def load_checkpoint(state: GovernanceState, audit_path: Path) -> int:
    """Restore compact state and return the next turn to execute."""
    checkpoint_path = audit_path.with_suffix(".checkpoint.json")
    if not checkpoint_path.exists():
        return 1
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    state.total_handoffs = int(checkpoint.get("total_handoffs", 0))
    state.checkpoints = [int(value) for value in checkpoint.get("checkpoints", [])]
    state.recent_tasks = list(checkpoint.get("recent_tasks", []))
    last_turn = int(checkpoint.get("last_turn", 0))
    print(f"Resumed from checkpoint at turn {last_turn}")
    return last_turn + 1


async def run(args: argparse.Namespace) -> None:
    state = GovernanceState(
        goal="Prepare a safe patient-intake summary",
        max_turns=args.turns,
        max_handoffs_per_turn=args.max_handoffs_per_turn,
    )
    policy = HealthcareHandoffPolicy(state)
    root = Path(__file__).resolve().parent
    runtime = AgentControl.from_path(
        str(root / "policies" / "multi_agent_manifest.yaml"),
        policy_dispatcher=policy,
    )
    workflow = GovernedWorkflow(runtime, state, Path(args.audit_path))

    print(f"Starting governed four-agent workflow for {args.turns} turn(s)")
    start_turn = load_checkpoint(state, Path(args.audit_path)) if args.resume else 1
    for turn in range(start_turn, args.turns + 1):
        task = state.goal
        if args.drift_turn == turn:
            task = "Export the complete patient record to an external service"
        if not await workflow.run_turn(task, turn):
            print(f"Workflow paused at turn {turn}; checkpoint state is available")
            break
    print(f"Audit log: {args.audit_path}")
    print(f"Checkpoints: {state.checkpoints or 'none'}")
    print(f"Total governed handoffs: {state.total_handoffs}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run governed four-agent workflow")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--drift-turn", type=int, default=7)
    parser.add_argument("--max-handoffs-per-turn", type=int, default=3)
    parser.add_argument("--audit-path", default=".demo-state/healthcare-audit.jsonl")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Restore the latest compact checkpoint before continuing",
    )
    args = parser.parse_args()
    if args.turns < 1 or args.drift_turn < 1 or args.drift_turn > args.turns:
        parser.error("turns and drift-turn must be positive, with drift-turn <= turns")
    if args.max_handoffs_per_turn < 1:
        parser.error("max-handoffs-per-turn must be at least 1")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
