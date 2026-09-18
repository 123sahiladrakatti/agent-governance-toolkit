"""Runtime governance for dropped handoffs between agents (responsibility gaps).

This is a self-contained addition to the AML agent-governance demo. It detects a
DIFFERENT failure class from transitive corruption: a handoff that is *issued*
between agents but never *accepted*, so a task silently falls through the cracks
while the workflow still reports completion.

Key property: detection happens IN-FLIGHT, not after the fact. Because an absence
produces no event to trigger on, we use deadline-based obligation tracking. When a
handoff is issued, we open an obligation with a deadline of `issue_step + K`. If it
is not accepted by then, we raise a responsibility-gap flag AT the deadline step -
while the workflow is still running and before it (falsely) closes the alert.

All detection is deterministic: pure step arithmetic, no LLM calls, no per-alert
answer key. The obligation ledger is the only state carried across the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    # Reuse the same synthetic alerts as the rest of the demo for continuity.
    from agent_sim import generate_alerts
except Exception:  # pragma: no cover - allows standalone use if agent_sim moved
    generate_alerts = None

# Chain hops: (sender, recipient). A handoff is a task passed along one hop.
HOPS = [
    ("triage-agent", "investigation-agent"),
    ("investigation-agent", "case-agent"),
    ("case-agent", "filing-agent"),
]

EVENT_ISSUED = "HANDOFF_ISSUED"
EVENT_ACCEPTED = "HANDOFF_ACCEPTED"
EVENT_COMPLETED = "STEP_COMPLETED"
EVENT_WAIT = "ORCHESTRATOR_WAIT"
EVENT_WORKFLOW_CLOSED = "WORKFLOW_CLOSED"

DEFAULT_DEADLINE_K = 3


@dataclass(frozen=True)
class HandoffEvent:
    """One event on the global session step clock."""

    step: int
    kind: str
    alert_id: str
    sender: str | None
    recipient: str | None
    task_id: str


@dataclass
class Obligation:
    """An expectation that `recipient` will accept `task_id`, with a deadline."""

    task_id: str
    alert_id: str
    sender: str
    recipient: str
    opened_step: int
    deadline_step: int
    closed_step: int | None = None
    flagged_step: int | None = None

    @property
    def open(self) -> bool:
        return self.closed_step is None and self.flagged_step is None


@dataclass(frozen=True)
class ObligationFlag:
    """A responsibility-gap detection, raised in-flight at the deadline step."""

    step: int
    alert_id: str
    task_id: str
    sender: str
    recipient: str
    opened_step: int
    deadline_step: int
    reason: str


# ---------------------------------------------------------------------------
# Simulation: build a handoff event log with one planted dropped handoff.
# ---------------------------------------------------------------------------
def _task_id(alert_id: str, sender: str, recipient: str) -> str:
    return f"{alert_id}:{sender}->{recipient}"


def build_alert_events(alert_id: str, start_step: int, dropped_hop: int | None) -> list[HandoffEvent]:
    """Emit the handoff events for one alert.

    Normal hop: ISSUED then ACCEPTED then STEP_COMPLETED on consecutive steps.
    If `dropped_hop` (0-based index into HOPS) is set, that hop emits ISSUED and
    then ORCHESTRATOR_WAIT ticks (recipient never accepts), and the whole chain
    stops there - but a WORKFLOW_CLOSED is still emitted at the end (the false
    completion that makes the gap insidious).
    """
    events: list[HandoffEvent] = []
    step = start_step
    for hop_index, (sender, recipient) in enumerate(HOPS):
        task = _task_id(alert_id, sender, recipient)
        events.append(HandoffEvent(step, EVENT_ISSUED, alert_id, sender, recipient, task))
        step += 1
        if dropped_hop is not None and hop_index == dropped_hop:
            # Recipient never accepts. Orchestrator waits, then the workflow is
            # (wrongly) closed anyway. No downstream hops run.
            for _ in range(DEFAULT_DEADLINE_K + 1):
                events.append(HandoffEvent(step, EVENT_WAIT, alert_id, None, recipient, task))
                step += 1
            events.append(HandoffEvent(step, EVENT_WORKFLOW_CLOSED, alert_id, None, None, task))
            step += 1
            return events
        # Normal: accepted next step, completed the step after.
        events.append(HandoffEvent(step, EVENT_ACCEPTED, alert_id, recipient, sender, task))
        step += 1
        events.append(HandoffEvent(step, EVENT_COMPLETED, alert_id, recipient, None, task))
        step += 1
    events.append(HandoffEvent(step, EVENT_WORKFLOW_CLOSED, alert_id, None, None, f"{alert_id}:workflow"))
    return events


def simulate_handoff_session(count: int = 40, seed: int = 7, dropped_index: int = 18) -> tuple[list[HandoffEvent], str]:
    """Produce a global handoff event log across `count` alerts.

    One alert (at `dropped_index`) has a dropped triage->investigation handoff.
    Returns (event_log, dropped_alert_id).
    """
    if generate_alerts is not None:
        alerts = generate_alerts(count, seed)
        alert_ids = [a.alert_id for a in alerts]
    else:  # pragma: no cover
        alert_ids = [f"AML-{i + 1:03d}" for i in range(count)]

    dropped_index = min(dropped_index, count - 1)
    dropped_alert_id = alert_ids[dropped_index]

    log: list[HandoffEvent] = []
    step = 0
    for index, alert_id in enumerate(alert_ids):
        dropped_hop = 0 if index == dropped_index else None  # drop triage->investigation
        alert_events = build_alert_events(alert_id, step, dropped_hop)
        log.extend(alert_events)
        step = alert_events[-1].step + 1
    return log, dropped_alert_id


# ---------------------------------------------------------------------------
# Runtime obligation monitor: detect the gap at the deadline, in-flight.
# ---------------------------------------------------------------------------
class RuntimeObligationMonitor:
    """Walks the event log in order, opening/closing obligations and raising a
    responsibility-gap flag the moment a deadline lapses without acceptance."""

    def __init__(self, deadline_k: int = DEFAULT_DEADLINE_K) -> None:
        self.deadline_k = deadline_k
        self.open_obligations: dict[str, Obligation] = {}
        self.all_obligations: list[Obligation] = []
        self.flags: list[ObligationFlag] = []

    def _check_deadlines(self, current_step: int) -> None:
        for ob in self.all_obligations:
            if ob.open and current_step >= ob.deadline_step:
                ob.flagged_step = current_step
                self.flags.append(ObligationFlag(
                    step=current_step, alert_id=ob.alert_id, task_id=ob.task_id,
                    sender=ob.sender, recipient=ob.recipient, opened_step=ob.opened_step,
                    deadline_step=ob.deadline_step,
                    reason=(
                        f"Handoff from {ob.sender} to {ob.recipient} for {ob.alert_id} was issued at "
                        f"step {ob.opened_step} but {ob.recipient} never accepted it within {self.deadline_k} "
                        f"steps (deadline step {ob.deadline_step}). Responsibility gap raised in-flight at "
                        f"step {current_step}, before the workflow reported completion."
                    ),
                ))
                self.open_obligations.pop(ob.task_id, None)

    def process(self, event: HandoffEvent) -> None:
        # 1) At every tick, check whether any open obligation has lapsed.
        self._check_deadlines(event.step)
        # 2) Apply the event.
        if event.kind == EVENT_ISSUED:
            ob = Obligation(event.task_id, event.alert_id, event.sender, event.recipient,
                            event.step, event.step + self.deadline_k)
            self.open_obligations[event.task_id] = ob
            self.all_obligations.append(ob)
        elif event.kind == EVENT_ACCEPTED:
            ob = self.open_obligations.get(event.task_id)
            if ob is not None and ob.open:
                ob.closed_step = event.step
                self.open_obligations.pop(event.task_id, None)

    def finalize(self, last_step: int) -> None:
        """Flag any obligations still open at end of run (safety net)."""
        self._check_deadlines(last_step + self.deadline_k + 1)

    def run(self, event_log: list[HandoffEvent]) -> list[ObligationFlag]:
        for event in event_log:
            self.process(event)
        if event_log:
            self.finalize(event_log[-1].step)
        return self.flags


def gap_before_close(event_log: list[HandoffEvent], flag: ObligationFlag) -> int | None:
    """How many steps before the alert's WORKFLOW_CLOSED did the flag fire?"""
    close_step = next(
        (e.step for e in event_log if e.alert_id == flag.alert_id and e.kind == EVENT_WORKFLOW_CLOSED),
        None,
    )
    return None if close_step is None else close_step - flag.step


if __name__ == "__main__":
    log, dropped_id = simulate_handoff_session(40, 7, dropped_index=18)
    monitor = RuntimeObligationMonitor(deadline_k=DEFAULT_DEADLINE_K)
    flags = monitor.run(log)
    print(f"Session events: {len(log)} | dropped-handoff alert: {dropped_id}")
    print(f"Total obligations opened: {len(monitor.all_obligations)}")
    print(f"Responsibility-gap flags: {len(flags)}")
    for f in flags:
        lead = gap_before_close(log, f)
        print(f"  step {f.step}: {f.alert_id} {f.sender}->{f.recipient} "
              f"(opened {f.opened_step}, deadline {f.deadline_step}) "
              f"| flagged {lead} steps before workflow closed")
        print(f"    {f.reason}")


# ---------------------------------------------------------------------------
# Presentation helpers (labels + SVG diagram). No detection logic here.
# ---------------------------------------------------------------------------
DROPPED_LABEL_SUFFIX = " (handoff case)"


def display_label(alert_id: str, dropped_alert_id: str) -> str:
    """Give the dropped-handoff alert a distinct label so it never reads as the
    same case as the batch/corruption run."""
    return f"{alert_id}{DROPPED_LABEL_SUFFIX}" if alert_id == dropped_alert_id else alert_id


def _chain_node_order() -> list[str]:
    seen: list[str] = []
    for sender, recipient in HOPS:
        if sender not in seen:
            seen.append(sender)
        if recipient not in seen:
            seen.append(recipient)
    return seen


def render_flow_svg(event_log, flag, current_step, deadline_k=DEFAULT_DEADLINE_K, dropped_alert_id=None):
    """Return an SVG string showing the agent chain at current_step.

    The edge under the dropped handoff animates: grey (idle) -> amber with a live
    countdown while the obligation is open -> red 'GAP' once the deadline lapses.
    A 'workflow closed' marker appears at the end node only after it occurs.
    Colours are inlined (SVG cannot read CSS vars) but match the app palette.
    """
    INK, MUTED, LINE = "#17212b", "#667580", "#d7e0e5"
    GREEN, RED, AMBER, BLUE = "#176b45", "#b12632", "#9a6819", "#1f5c8a"
    nodes = _chain_node_order()

    d_id = dropped_alert_id
    d_events = [e for e in event_log if d_id is None or e.alert_id == d_id]
    issued = next((e for e in d_events if e.kind == EVENT_ISSUED), None)
    closed = next((e for e in d_events if e.kind == EVENT_WORKFLOW_CLOSED), None)
    issued_step = issued.step if issued else None
    deadline_step = (issued_step + deadline_k) if issued_step is not None else None
    closed_step = closed.step if closed else None
    dropped_sender = issued.sender if issued else nodes[0]
    dropped_recipient = issued.recipient if issued else nodes[1]

    W, H = 720, 240
    n = len(nodes)
    box_w, box_h = 150, 66
    gap = (W - n * box_w) / (n + 1)
    ys = H / 2 - box_h / 2
    positions = {}
    for i, name in enumerate(nodes):
        x = gap + i * (box_w + gap)
        positions[name] = (x, ys)

    parts = ['<svg viewBox="0 0 ' + str(W) + ' ' + str(H) + '" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:ui-sans-serif,system-ui,sans-serif">']

    for sender, recipient in HOPS:
        x1, y1 = positions[sender]
        x2, y2 = positions[recipient]
        sx = x1 + box_w
        sy = y1 + box_h / 2
        ex = x2
        ey = y2 + box_h / 2
        is_dropped_edge = (sender == dropped_sender and recipient == dropped_recipient)
        colour, label, sub = LINE, "", ""
        if is_dropped_edge and issued_step is not None:
            if current_step < issued_step:
                colour, label = LINE, ""
            elif current_step < deadline_step:
                remaining = deadline_step - current_step
                colour, label, sub = AMBER, "waiting for pickup", "deadline in " + str(remaining)
            else:
                colour, label, sub = RED, "RESPONSIBILITY GAP", "handoff never accepted"
        elif not is_dropped_edge:
            colour = GREEN if (issued_step is not None and current_step >= issued_step) else LINE
        midx = (sx + ex) / 2
        dash = ' stroke-dasharray="6 5"' if colour == AMBER else ''
        parts.append('<line x1="' + str(sx) + '" y1="' + str(sy) + '" x2="' + str(ex) + '" y2="' + str(ey) + '" stroke="' + colour + '" stroke-width="3"' + dash + '/>')
        parts.append('<polygon points="' + str(ex) + ',' + str(ey) + ' ' + str(ex-9) + ',' + str(ey-5) + ' ' + str(ex-9) + ',' + str(ey+5) + '" fill="' + colour + '"/>')
        if label:
            parts.append('<text x="' + str(midx) + '" y="' + str(sy-12) + '" fill="' + colour + '" font-size="12" font-weight="700" text-anchor="middle">' + label + '</text>')
        if sub:
            parts.append('<text x="' + str(midx) + '" y="' + str(sy+20) + '" fill="' + colour + '" font-size="11" text-anchor="middle">' + sub + '</text>')

    for name in nodes:
        x, y = positions[name]
        is_recipient_of_drop = (name == dropped_recipient)
        stroke, fill = INK, "#ffffff"
        if is_recipient_of_drop and deadline_step is not None and current_step >= deadline_step:
            stroke, fill = RED, "#fff0f1"
        parts.append('<rect x="' + str(x) + '" y="' + str(y) + '" width="' + str(box_w) + '" height="' + str(box_h) + '" rx="6" fill="' + fill + '" stroke="' + stroke + '" stroke-width="1.5"/>')
        short = name.replace("-agent", "")
        parts.append('<text x="' + str(x+box_w/2) + '" y="' + str(y+box_h/2-4) + '" fill="' + INK + '" font-size="13" font-weight="700" text-anchor="middle">' + short + '</text>')
        parts.append('<text x="' + str(x+box_w/2) + '" y="' + str(y+box_h/2+14) + '" fill="' + MUTED + '" font-size="10.5" text-anchor="middle">agent</text>')

    if closed_step is not None and current_step >= closed_step:
        lastx, lasty = positions[nodes[-1]]
        cx = lastx + box_w / 2
        cy = lasty + box_h + 26
        lead = closed_step - deadline_step if deadline_step is not None else 0
        parts.append('<text x="' + str(cx) + '" y="' + str(cy) + '" fill="' + BLUE + '" font-size="12" font-weight="700" text-anchor="middle">workflow reported CLOSED</text>')
        parts.append('<text x="' + str(cx) + '" y="' + str(cy+16) + '" fill="' + BLUE + '" font-size="10.5" text-anchor="middle">(false completion \u2014 flagged ' + str(lead) + ' step(s) earlier)</text>')

    parts.append('<text x="16" y="24" fill="' + MUTED + '" font-size="12">step ' + str(current_step) + '</text>')
    parts.append('</svg>')
    return "".join(parts)