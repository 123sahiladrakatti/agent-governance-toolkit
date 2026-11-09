"""Backend batch of healthcare cases: read an Excel file and govern every case.

Reads ``batch_cases/healthcare_cases.xlsx`` (one row per case), validates it, and
runs each case through the same LLM simulation and ``govern_client_run`` path the
interactive form uses. A case whose LLM call or ingestion fails becomes an error
row; the batch carries on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from governed_run import GovernedRun, govern_client_run
from healthcare.client_payload import (
    HEALTHCARE_RECORD_CODECS,
    simulate_healthcare_case_llm,
    simulate_healthcare_case_llm_pii,
)
from healthcare.correctness import HealthcareCorrectnessComponent
from healthcare.system import MEDICATION_ALLERGENS, HealthcareCase

DEFAULT_BATCH_FILE = Path(__file__).resolve().parents[1] / "batch_cases" / "healthcare_cases.xlsx"
BATCH_COLUMNS = (
    "patient_name", "patient_id", "member_id", "age", "diagnosis",
    "medication", "known_allergy", "coverage_authorized",
)
BATCH_LIMIT = 10  # LLM runs take 15-60 s each
MEDICATIONS = ("Acetaminophen", *MEDICATION_ALLERGENS)
KNOWN_ALLERGIES = ("none", "penicillin", "cephalosporin", "NSAID")
MEMBER_ID_FORMAT = re.compile(r"^MBR-\d{4}-\d{4}-\d{2}$")  # synthetic member numbers only
BATCH_MODES = {"llm-context-loss": "Context loss (care-decision agent)", "llm-pii-leak": "PII leak (allergy agent)"}
VERDICT_LABELS = {"allow": "CONTINUE", "require_approval": "HUMAN REVIEW", "deny": "BLOCK"}
_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}


class BatchFileError(ValueError):
    """The batch file is missing or malformed; the message names the row and column."""


@dataclass(frozen=True)
class BatchCase:
    row: int               # spreadsheet row number (the header is row 1)
    case: HealthcareCase


@dataclass(frozen=True)
class BatchResult:
    row: int
    patient_name: str
    medication: str
    known_allergy: str
    disposition: str | None
    trace_check: str | None        # PASS / FLAG
    nemo: str | None               # PASS / FLAG / UNAVAILABLE
    verdict: str | None            # CONTINUE / HUMAN REVIEW / BLOCK
    caught_by: str
    error: str | None = None
    governed: GovernedRun | None = None
    llm: Any = None                # LLMDecision (context loss) or LLMAllergyNote (PII leak)
    case: HealthcareCase | None = None


def _blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if value != value:  # NaN
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and not value.strip()


def _text(value: Any, row: int, column: str, required: bool = True) -> str:
    if _blank(value):
        if required:
            raise BatchFileError(f"Row {row}, column '{column}': value is required.")
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def _age(value: Any, row: int) -> int:
    if _blank(value):
        raise BatchFileError(f"Row {row}, column 'age': value is required.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise BatchFileError(f"Row {row}, column 'age': {value!r} is not a number.") from None
    if not number.is_integer() or not 0 <= number <= 120:
        raise BatchFileError(f"Row {row}, column 'age': {value!r} must be a whole number from 0 to 120.")
    return int(number)


def _choice(value: Any, row: int, column: str, allowed: tuple[str, ...]) -> str:
    text = _text(value, row, column)
    match = next((option for option in allowed if option.casefold() == text.casefold()), None)
    if match is None:
        raise BatchFileError(f"Row {row}, column '{column}': {text!r} is not one of {', '.join(allowed)}.")
    return match


def _flag(value: Any, row: int) -> bool:
    if isinstance(value, bool):
        return value
    text = _text(value, row, "coverage_authorized").casefold()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise BatchFileError(f"Row {row}, column 'coverage_authorized': {value!r} must be TRUE or FALSE.")


def load_healthcare_cases(path: Path | str = DEFAULT_BATCH_FILE) -> list[BatchCase]:
    """Read and validate the batch file; raise ``BatchFileError`` naming the bad row/column."""

    import pandas as pd

    path = Path(path)
    if not path.is_file():
        raise BatchFileError(f"Batch file not found: {path}")
    try:
        frame = pd.read_excel(path, engine="openpyxl", dtype=object)
    except Exception as exc:  # corrupt or non-Excel file
        raise BatchFileError(f"{path.name}: could not be read as an Excel workbook ({exc}).") from exc

    columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in BATCH_COLUMNS if column not in columns]
    unexpected = [column for column in columns if column not in BATCH_COLUMNS]
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing column(s): {', '.join(missing)}")
        if unexpected:
            problems.append(f"unexpected column(s): {', '.join(unexpected)}")
        raise BatchFileError(
            f"{path.name}, header row: {'; '.join(problems)}. Expected columns: {', '.join(BATCH_COLUMNS)}."
        )
    frame.columns = columns

    cases: list[BatchCase] = []
    for index, record in enumerate(frame.to_dict("records")):
        row = index + 2
        if all(_blank(record[column]) for column in BATCH_COLUMNS):
            continue  # skip fully empty rows
        member_id = _text(record["member_id"], row, "member_id", required=False)
        if member_id and not MEMBER_ID_FORMAT.match(member_id):
            raise BatchFileError(
                f"Row {row}, column 'member_id': {member_id!r} must be a synthetic member number "
                "like MBR-1234-5678-90 (or blank)."
            )
        cases.append(BatchCase(row, HealthcareCase(
            case_id=f"HC-BATCH-{row:02d}",
            patient_id=_text(record["patient_id"], row, "patient_id"),
            patient_name=_text(record["patient_name"], row, "patient_name"),
            age=_age(record["age"], row),
            diagnosis=_text(record["diagnosis"], row, "diagnosis"),
            medication=_choice(record["medication"], row, "medication", MEDICATIONS),
            known_allergy=_choice(record["known_allergy"], row, "known_allergy", KNOWN_ALLERGIES),
            coverage_authorized=_flag(record["coverage_authorized"], row),
            member_id=member_id,
        )))
    if not cases:
        raise BatchFileError(f"{path.name}: no cases found below the header row.")
    return cases


def _summarize(item: BatchCase, governed: GovernedRun, llm: Any) -> BatchResult:
    review, integrity = governed.review, governed.integrity
    executed = [check for check in review.content_checks if check.get("rail") != "skipped"]
    flagged = [check["agent"] for check in executed if not check.get("passed", False)]
    unavailable = any(check.get("rail") == "unavailable" for check in executed)
    nemo = "FLAG" if flagged else ("UNAVAILABLE" if unavailable else "PASS")
    if not integrity.passed:
        lost = ", ".join(integrity.related_agents)
        caught_by = f"Trace check: {integrity.origin_agent} did not consume {lost} as emitted"
    elif flagged:
        caught_by = f"NeMo: personal data or injection in {', '.join(flagged)} output"
    elif not review.attribution.passed:
        caught_by = f"Correctness: {review.attribution.origin_agent} ({review.attribution.category})"
    else:
        caught_by = "-"
    action = str(review.policy_decision.get("action", "deny"))
    return BatchResult(
        row=item.row,
        patient_name=item.case.patient_name,
        medication=item.case.medication,
        known_allergy=item.case.known_allergy,
        disposition=review.outcome.step(governed.final_agent).disposition,
        trace_check="PASS" if integrity.passed else "FLAG",
        nemo=nemo,
        verdict=VERDICT_LABELS.get(action, action.upper()),
        caught_by=caught_by,
        governed=governed,
        llm=llm,
        case=item.case,
    )


def run_healthcare_batch(
    cases: list[BatchCase],
    mode: str,
    definition: Mapping[str, Any],
    decision_config: Any = None,
    allergy_config: Any = None,
    on_case: Callable[[int, int, BatchCase], None] | None = None,
    content_checker: Callable[[str], Any] | None = None,
) -> list[BatchResult]:
    """Govern each case through the chosen LLM failure mode; errors become rows."""

    if mode not in BATCH_MODES:
        raise ValueError(f"unsupported batch mode: {mode!r}")
    results: list[BatchResult] = []
    for position, item in enumerate(cases[:BATCH_LIMIT], start=1):
        if on_case is not None:
            on_case(position, min(len(cases), BATCH_LIMIT), item)
        try:
            if mode == "llm-context-loss":
                run, records, llm = simulate_healthcare_case_llm(item.case, decision_config)
            else:
                run, records, llm = simulate_healthcare_case_llm_pii(item.case, allergy_config)
            governed = govern_client_run(
                definition, run, records, HEALTHCARE_RECORD_CODECS, HealthcareCorrectnessComponent(),
                content_checker=content_checker,
            )
            results.append(_summarize(item, governed, llm))
        except Exception as exc:  # one failed case must not abort the batch
            results.append(BatchResult(
                row=item.row, patient_name=item.case.patient_name, medication=item.case.medication,
                known_allergy=item.case.known_allergy, disposition=None, trace_check=None, nemo=None,
                verdict=None, caught_by="-", error=f"{type(exc).__name__}: {exc}", case=item.case,
            ))
    return results


def summary_line(results: list[BatchResult]) -> str:
    flagged = sum(1 for r in results if r.verdict in ("HUMAN REVIEW", "BLOCK"))
    clean = sum(1 for r in results if r.verdict == "CONTINUE")
    errors = sum(1 for r in results if r.error)
    verdicts = {label: sum(1 for r in results if r.verdict == label) for label in ("CONTINUE", "HUMAN REVIEW", "BLOCK")}
    parts = [f"{len(results)} cases run", f"{flagged} flagged", f"{clean} clean"]
    if errors:
        parts.append(f"{errors} errored")
    return " · ".join(parts) + " — " + ", ".join(f"{label}: {count}" for label, count in verdicts.items())


__all__ = [
    "BATCH_COLUMNS",
    "BATCH_LIMIT",
    "BATCH_MODES",
    "BatchCase",
    "BatchFileError",
    "BatchResult",
    "DEFAULT_BATCH_FILE",
    "load_healthcare_cases",
    "run_healthcare_batch",
    "summary_line",
]
