"""Tiny in-process trace so the multi-agent interaction is visible in the terminal.

This is observability only -- it records and prints what happened. It does not
inspect, allow, deny, or modify anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Kinds of event we record, in the order they read best in a log.
TOOL = "TOOL"
RETURN = "RETURN"
HANDOFF = "HANDOFF"  # intake -> records: the outbound delegation
REPLY = "REPLY"  # records -> intake: the answer coming back
GOV = "GOV"  # a governance verdict at an ACS intervention point
NOTE = "NOTE"


@dataclass
class Event:
    turn: int
    agent: str
    kind: str
    label: str
    detail: str = ""


@dataclass
class Trace:
    events: list[Event] = field(default_factory=list)
    turn: int = 0

    def start_turn(self, turn: int) -> None:
        self.turn = turn

    def record(self, agent: str, kind: str, label: str, detail: str = "") -> Event:
        event = Event(self.turn, agent, kind, label, detail)
        self.events.append(event)
        print(_render(event), flush=True)
        return event

    def for_turn(self, turn: int) -> list[Event]:
        return [event for event in self.events if event.turn == turn]


TRACE = Trace()


def truncate(text: object, limit: int = 400) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _render(event: Event) -> str:
    line = f"  [{event.kind:<7}] {event.agent:<14} | {event.label}"
    if event.detail:
        line += f"\n{'':<26}-> {event.detail}"
    return line


def banner(text: str, char: str = "=", width: int = 100) -> str:
    return f"\n{char * width}\n{text}\n{char * width}"
