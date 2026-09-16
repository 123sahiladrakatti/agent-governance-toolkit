"""Mock tools for the two agents. All data is hardcoded in mock_data.py.

Ownership matters for the demo: the Intake Agent can only see demographics, and
the Records Agent is the only one that can read records or the calendar. That
asymmetry is what forces a real handoff instead of two independent agents.

The tools do no printing -- main.py logs every call from CrewAI's event bus, so
the trace reports the agent CrewAI actually attributed the call to.
"""

from __future__ import annotations

import json

from crewai.tools import tool

from mock_data import DEMOGRAPHICS, RECORDS, SCHEDULE


def _normalize(patient_id: object) -> str:
    """Small models quote, pad, or lowercase ids. Accept all of that."""
    return str(patient_id or "").strip().strip("'\"").upper()


def _lookup(table: dict[str, dict], patient_id: object) -> str:
    pid = _normalize(patient_id)
    payload = table.get(pid) or {
        "error": f"no such patient id {pid!r}",
        "known_patient_ids": ", ".join(table),
    }
    return json.dumps(payload, default=str)


@tool("get_patient_demographics")
def get_patient_demographics(patient_id: str) -> str:
    """Look up a patient's demographics by patient id (for example 'P-1001').

    Returns name, date of birth, age, phone, insurance and preferred language.
    This tool does NOT return medical records, medications, or appointments.
    """
    return _lookup(DEMOGRAPHICS, patient_id)


@tool("lookup_patient_record")
def lookup_patient_record(patient_id: str) -> str:
    """Read a patient's medical record by patient id (for example 'P-1001').

    Returns primary care provider, active conditions, medications, allergies and
    the date of the last visit.
    """
    return _lookup(RECORDS, patient_id)


@tool("get_schedule")
def get_schedule(patient_id: str) -> str:
    """Read a patient's next scheduled appointment by patient id (e.g. 'P-1001').

    Returns the appointment date and time, clinic, provider and visit type.
    """
    return _lookup(SCHEDULE, patient_id)


INTAKE_TOOLS = [get_patient_demographics]
RECORDS_TOOLS = [lookup_patient_record, get_schedule]
