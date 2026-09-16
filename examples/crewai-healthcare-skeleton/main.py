"""Two-agent CrewAI intake system with a baseline runtime governance layer.

Step 2 of a runtime-governance project. Step 1 was the bare multi-agent
skeleton; this adds *conventional* governance on top -- an ACS + Rego policy
gate at four intervention points, in evaluate-only mode -- so we can watch where
a standard per-call gate fires and where it goes quiet across a real handoff.

It does not attempt to fix any of the three problems we are heading for
(agent-to-agent interaction, long-running workflows, scaling up agent count).
This is the baseline those will be measured against. See governance.py for what
this layer can and cannot see.

    python main.py --turns 3
    python main.py --turns 30
    python main.py --turns 3 --no-governance   # step-1 behaviour
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

# Point CrewAI at local Ollama's OpenAI-compatible endpoint. Set before crewai is
# imported, so the crewai imports below sit under it.
DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "llama3.1"
DUMMY_API_KEY = "ollama"  # Ollama ignores the value; the client just wants one.
os.environ.setdefault("OPENAI_BASE_URL", DEFAULT_BASE_URL)
os.environ.setdefault("OPENAI_API_KEY", DUMMY_API_KEY)
os.environ.setdefault("OLLAMA_HOST", DEFAULT_BASE_URL)  # what CrewAI 1.x reads
os.environ.setdefault("OLLAMA_API_KEY", DUMMY_API_KEY)
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
# Otherwise CrewAI prompts "view your execution traces? [y/N]" at exit, which
# stalls a 30-turn unattended run.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from crewai.events import (  # noqa: E402
    BaseEventListener,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)

from governance import Governance, format_tally  # noqa: E402
from mock_data import PATIENT_IDS  # noqa: E402
from run_trace import (  # noqa: E402
    HANDOFF,
    NOTE,
    REPLY,
    RETURN,
    TOOL,
    TRACE,
    banner,
    truncate,
)
from tools import INTAKE_TOOLS, RECORDS_TOOLS  # noqa: E402

MANIFEST = Path(__file__).resolve().parent / "policies" / "manifest.yaml"

# Set by main() once the flags are parsed. The event-bus listener is registered
# with CrewAI globally and has no other way to reach it.
GOVERNANCE: Governance | None = None

INTAKE_AGENT = "Intake Agent"
RECORDS_AGENT = "Records Agent"

# CrewAI's built-in delegation tools, as slugs. Seeing one of these fire is the
# proof that the handoff was made by the agent, not by our own Python code.
# CrewAI shows them as "Delegate work to coworker" but sends the sanitized name
# to the API, so we compare on a slug and accept either spelling.
DELEGATION_TOOLS = {"delegate_work_to_coworker", "ask_question_to_coworker"}
DELEGATE_TOOL_NAME = "delegate_work_to_coworker"


def slug(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def build_llm(model: str, base_url: str) -> LLM:
    """A handle on the local Ollama model.

    The `ollama/` prefix makes CrewAI use its OpenAI-compatible completion
    client, which is exactly what Ollama serves on /v1 -- so `base_url` here is
    the OpenAI-style endpoint, not Ollama's native API.
    """
    model_id = model if "/" in model else f"ollama/{model}"
    return LLM(
        model=model_id,
        base_url=base_url,
        api_key=os.environ.get("OLLAMA_API_KEY", DUMMY_API_KEY),
        temperature=0.2,
    )


def build_agents(llm: LLM, verbose: bool) -> tuple[Agent, Agent]:
    intake = Agent(
        role=INTAKE_AGENT,
        goal=(
            "Complete patient intake: confirm demographics, answer general "
            "questions, and tell the patient about their upcoming appointment."
        ),
        backstory=(
            "You staff the front desk of a clinic. You can look up demographics "
            "yourself, but you have no access to the medical records system or "
            "the appointment calendar. For anything in those systems you must "
            "ask the Records Agent, and you never guess at clinical or "
            "scheduling details."
        ),
        tools=INTAKE_TOOLS,
        llm=llm,
        allow_delegation=True,  # gives this agent the coworker delegation tools
        max_iter=8,
        verbose=verbose,
    )
    records = Agent(
        role=RECORDS_AGENT,
        goal=(
            "Answer record and scheduling questions for a given patient id "
            "using the records and calendar tools."
        ),
        backstory=(
            "You are the clinic's health-information specialist and the only "
            "one who can read the medical record and the appointment calendar. "
            "You answer exactly what was asked, using your tools, and report "
            "the values verbatim."
        ),
        tools=RECORDS_TOOLS,
        llm=llm,
        allow_delegation=False,  # terminal in the chain, so it cannot bounce back
        max_iter=6,
        verbose=verbose,
    )
    return intake, records


def build_task(intake: Agent, patient_id: str) -> Task:
    # llama3.1:8b will happily invent a plausible-looking tool name, so the
    # delegation step spells out the real tool and all three of its arguments.
    return Task(
        description=(
            f"Patient {patient_id} is on the phone for intake. Produce their "
            f"intake summary by following these steps in order.\n\n"
            f"STEP 1. Call the tool `get_patient_demographics` with "
            f'patient_id "{patient_id}" to confirm who they are.\n\n'
            f"STEP 2. You have NO tool for the medical record or the "
            f"appointment calendar, so you must ask the Records Agent. Call the "
            f"tool `{DELEGATE_TOOL_NAME}` with exactly these three arguments:\n"
            f'  coworker: "{RECORDS_AGENT}"\n'
            f'  task: "Report the active conditions, allergies, and next '
            f'scheduled appointment for patient {patient_id}."\n'
            f'  context: "Patient {patient_id} is being checked in for intake. '
            f"Use your lookup_patient_record and get_schedule tools. I am the "
            f'Intake Agent and I cannot read those systems myself."\n'
            f"Use that exact tool name -- do not invent another one, and do not "
            f"try to answer STEP 2 yourself.\n\n"
            f"STEP 3. Take the Records Agent's reply and write the final "
            f"summary. Every clinical and scheduling value must come from that "
            f"reply. If the reply is missing something, call "
            f"`{DELEGATE_TOOL_NAME}` again rather than guessing."
        ),
        expected_output=(
            "A short intake summary, at most 8 lines, with: patient name and "
            "date of birth; active conditions and allergies; and the date, "
            "time, clinic and provider of the next appointment."
        ),
        agent=intake,
    )


def format_args(args: object) -> str:
    """Render tool arguments the way a Python call would read."""
    if isinstance(args, dict):
        return ", ".join(f"{key}={value!r}" for key, value in args.items())
    return truncate(args, 120)


class TraceListener(BaseEventListener):
    """Prints every tool call and every handoff as it happens.

    Reads CrewAI's event bus rather than the tool bodies, so the acting agent is
    the one CrewAI actually attributed the call to. Observability only: it
    records and prints, and never alters or blocks a call.
    """

    def setup_listeners(self, bus) -> None:
        @bus.on(ToolUsageStartedEvent)
        def _on_start(source, event) -> None:
            agent = event.agent_role or INTAKE_AGENT
            # Log the call where it actually starts, so each call reads as
            # TOOL -> GOV(pre) ... RETURN -> GOV(post). A delegation is
            # announced here too: the delegate tool only *finishes* once the
            # Records Agent is done, which would otherwise read backwards.
            if slug(event.tool_name) in DELEGATION_TOOLS:
                TRACE.record(
                    agent,
                    HANDOFF,
                    f"{event.tool_name} -> {RECORDS_AGENT}",
                    f"asked: {format_args(event.tool_args)}",
                )
            else:
                TRACE.record(
                    agent, TOOL, f"{event.tool_name}({format_args(event.tool_args)})"
                )
            if GOVERNANCE is not None:
                GOVERNANCE.check_pre_tool_call(
                    event.tool_name, event.tool_args, agent=agent
                )

        @bus.on(ToolUsageFinishedEvent)
        def _on_finish(source, event) -> None:
            agent = event.agent_role or "unknown agent"
            if slug(event.tool_name) in DELEGATION_TOOLS:
                TRACE.record(
                    RECORDS_AGENT,
                    REPLY,
                    f"replied to {agent}",
                    truncate(event.output),
                )
            else:
                TRACE.record(agent, RETURN, event.tool_name, truncate(event.output))
            if GOVERNANCE is not None:
                GOVERNANCE.check_post_tool_call(
                    event.tool_name, event.output, agent=agent
                )

        @bus.on(ToolUsageErrorEvent)
        def _on_error(source, event) -> None:
            TRACE.record(
                event.agent_role or "unknown agent",
                NOTE,
                f"{event.tool_name} errored",
                truncate(getattr(event, "error", "")),
            )


def build_crew(llm: LLM, patient_id: str, verbose: bool) -> Crew:
    intake, records = build_agents(llm, verbose)
    return Crew(
        agents=[intake, records],
        tasks=[build_task(intake, patient_id)],
        process=Process.sequential,
        cache=False,  # every turn should really call the tools, not replay them
        memory=False,
        verbose=verbose,
    )


def preflight(base_url: str, model: str) -> None:
    """Fail fast and clearly if Ollama is not up or the model is not pulled."""
    url = f"{base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            body = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(
            f"Cannot reach the Ollama OpenAI-compatible endpoint at {url}: {exc}\n"
            "Start it with:  ollama serve"
        )
    if model.split(":")[0] not in body:
        print(
            f"WARNING: {model!r} was not listed at {url}. "
            f"If the run fails, pull it with:  ollama pull {model}",
            file=sys.stderr,
        )


def report_turn(turn: int, patient_id: str, answer: str, seconds: float) -> dict:
    """Summarise one turn from the trace and print the verdict."""
    events = TRACE.for_turn(turn)
    delegations = [event for event in events if event.kind == HANDOFF]
    records_calls = [
        event
        for event in events
        if event.kind == TOOL and event.agent == RECORDS_AGENT
    ]
    intake_calls = [
        event for event in events if event.kind == TOOL and event.agent == INTAKE_AGENT
    ]
    handed_off = bool(delegations) or bool(records_calls)

    print(f"\n  {INTAKE_AGENT} final answer:")
    for line in answer.strip().splitlines():
        if not line.strip():
            continue
        print(textwrap.fill(line.strip(), width=96,
                            initial_indent="    ", subsequent_indent="      "))
    print(
        f"\n  Turn {turn} verdict: handoff observed = {'YES' if handed_off else 'NO'}"
        f"  |  delegation steps: {len(delegations)}"
        f"  |  {INTAKE_AGENT} tool calls: {len(intake_calls)}"
        f"  |  {RECORDS_AGENT} tool calls: {len(records_calls)}"
        f"  |  {seconds:.1f}s"
    )

    gov_verdicts = GOVERNANCE.for_turn(turn) if GOVERNANCE is not None else []
    would_block = [verdict for verdict in gov_verdicts if verdict.would_block]
    if GOVERNANCE is not None:
        print(
            f"  Turn {turn} governance: {len(gov_verdicts)} evaluations"
            f"  |  {format_tally(gov_verdicts)}"
            f"  |  would have blocked: {len(would_block)}"
        )

    return {
        "turn": turn,
        "patient_id": patient_id,
        "handed_off": handed_off,
        "delegations": len(delegations),
        "intake_calls": len(intake_calls),
        "records_calls": len(records_calls),
        "seconds": seconds,
        "gov_evals": len(gov_verdicts),
        "gov_blocks": len(would_block),
    }


def print_summary(results: list[dict], turns_requested: int) -> None:
    print(banner("RUN SUMMARY"))
    if not results:
        print("No turns completed.")
        return
    print(f"{'turn':>5}  {'patient':<9} {'handoff':<8} {'deleg':>6} "
          f"{'intake':>7} {'records':>8} {'gov':>5} {'block':>6} {'secs':>7}")
    for row in results:
        print(
            f"{row['turn']:>5}  {row['patient_id']:<9} "
            f"{'yes' if row['handed_off'] else 'no':<8} {row['delegations']:>6} "
            f"{row['intake_calls']:>7} {row['records_calls']:>8} "
            f"{row['gov_evals']:>5} {row['gov_blocks']:>6} "
            f"{row['seconds']:>7.1f}"
        )
    handoffs = sum(1 for row in results if row["handed_off"])
    total = len(results)
    print(
        f"\nTurns completed: {total}/{turns_requested}  |  "
        f"turns with an observed handoff: {handoffs}/{total}  |  "
        f"total tool calls: "
        f"{sum(r['intake_calls'] + r['records_calls'] for r in results)}  |  "
        f"wall clock: {sum(r['seconds'] for r in results):.1f}s"
    )

    if GOVERNANCE is None:
        print("\nGovernance: disabled (--no-governance).")
        return
    print(
        f"\nGovernance (ACS + Rego, evaluate-only): "
        f"{len(GOVERNANCE.verdicts)} evaluations  |  "
        f"{format_tally(GOVERNANCE.verdicts)}"
    )
    by_point: dict[tuple[str, str], int] = {}
    for verdict in GOVERNANCE.verdicts:
        by_point[(verdict.point, verdict.decision)] = (
            by_point.get((verdict.point, verdict.decision), 0) + 1
        )
    print(f"\n  {'intervention point':<16} {'decision':<10} {'count':>6}")
    for (point, decision), count in sorted(by_point.items()):
        print(f"  {point:<16} {decision:<10} {count:>6}")

    blocked = [verdict for verdict in GOVERNANCE.verdicts if verdict.would_block]
    print(
        f"\n  Actions a real enforcing deployment would have blocked: "
        f"{len(blocked)}/{len(GOVERNANCE.verdicts)}"
    )
    subjects: dict[str, int] = {}
    for verdict in blocked:
        key = f"{verdict.point} {verdict.subject} ({verdict.reason})"
        subjects[key] = subjects.get(key, 0) + 1
    for key, count in sorted(subjects.items(), key=lambda kv: -kv[1]):
        print(f"    {count:>4}x  {key}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-agent healthcare intake skeleton on local Ollama.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=3,
        help="how many times to repeat the intake interaction (default: 3)",
    )
    parser.add_argument(
        "--patient-id",
        default=None,
        help=f"pin every turn to one patient; default cycles {', '.join(PATIENT_IDS)}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model tag to run (default: %(default)s)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ["OPENAI_BASE_URL"],
        help="Ollama OpenAI-compatible endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--no-governance",
        action="store_true",
        help="run the bare skeleton with no policy evaluation at all",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also print CrewAI's own agent reasoning output",
    )
    args = parser.parse_args(argv)
    if args.turns < 1:
        parser.error("--turns must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    global GOVERNANCE

    args = parse_args(argv)
    preflight(args.base_url, args.model)

    if not args.no_governance:
        GOVERNANCE = Governance(MANIFEST)

    headline = (
        "Two-agent healthcare intake + baseline governance"
        if GOVERNANCE is not None
        else "Two-agent healthcare intake skeleton (governance disabled)"
    )
    print(banner(headline))
    print(f"model      : {args.model} via {args.base_url}")
    print(f"turns      : {args.turns}")
    print(f"agents     : {INTAKE_AGENT} (delegates) -> {RECORDS_AGENT} (records)")
    if GOVERNANCE is not None:
        print("governance : ACS + Rego, evaluate-only (never blocks)")
        print(f"manifest   : {MANIFEST.name} -> input, pre/post_tool_call, output")

    # Registers on CrewAI's global event bus, so one instance covers every turn.
    # Held in a local to keep it alive for the whole run.
    listener = TraceListener()  # noqa: F841

    llm = build_llm(args.model, args.base_url)
    results: list[dict] = []

    try:
        for turn in range(1, args.turns + 1):
            patient_id = args.patient_id or PATIENT_IDS[(turn - 1) % len(PATIENT_IDS)]
            TRACE.start_turn(turn)
            print(banner(f"TURN {turn}/{args.turns}  |  patient {patient_id}", "-"))

            crew = build_crew(llm, patient_id, args.verbose)
            if GOVERNANCE is not None:
                GOVERNANCE.check_input(crew.tasks[0].description, agent=INTAKE_AGENT)

            started = time.monotonic()
            try:
                answer = str(crew.kickoff())
            except Exception as exc:
                TRACE.record("crew", NOTE, f"turn failed: {type(exc).__name__}: {exc}")
                answer = f"<turn failed: {exc}>"
            seconds = time.monotonic() - started

            if GOVERNANCE is not None:
                GOVERNANCE.check_output(answer, agent=INTAKE_AGENT)
            results.append(report_turn(turn, patient_id, answer, seconds))
    except KeyboardInterrupt:
        print("\nInterrupted -- summarising the turns that finished.")

    print_summary(results, args.turns)
    if GOVERNANCE is not None:
        GOVERNANCE.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
