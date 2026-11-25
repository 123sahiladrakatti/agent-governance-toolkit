"""Persistent handoff between offline batch governance and the Streamlit UI.

The batch worker owns expensive ingestion, agent execution, and governance.  It
writes one materialized result per case.  The UI performs only indexed reads of
summary columns and fetches the full JSON document for a selected case.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

DEFAULT_BATCH_DB = Path(__file__).parent / "runtime_data" / "governance_batches.sqlite3"


@dataclass(frozen=True)
class StoredBatch:
    batch_id: str
    domain: str
    mode: str
    source_name: str
    source_sha256: str
    status: str
    started_at: str
    completed_at: str | None
    total_cases: int
    processed_cases: int
    continue_count: int
    review_count: int
    block_count: int
    error_count: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def source_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert governance dataclasses and runtime values into stable JSON data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(child) for child in value]
    if hasattr(value, "__dict__"):
        return {str(key): _jsonable(child) for key, child in vars(value).items()}
    return str(value)


def healthcare_result_document(result: Any, mode: str) -> dict[str, Any]:
    """Materialize all case and governance information for durable storage."""

    governed = result.governed
    governance = None
    if governed is not None:
        try:
            normalized_payload = json.loads(governed.payload_json)
        except (TypeError, json.JSONDecodeError):
            normalized_payload = governed.payload_json
        governance = {
            "normalized_payload": normalized_payload,
            "outcome": _jsonable(governed.outcome),
            "review": _jsonable(governed.review),
            "integrity": _jsonable(governed.integrity),
            "agt_live": governed.agt_live,
            "final_agent": governed.final_agent,
        }
    return {
        "schema": "governance-batch-result/v1",
        "domain": "Healthcare",
        "mode": mode,
        "summary": {
            "row": result.row,
            "case_id": getattr(result.case, "case_id", None),
            "subject": result.patient_name,
            "medication": result.medication,
            "known_allergy": result.known_allergy,
            "disposition": result.disposition,
            "trace_check": result.trace_check,
            "nemo": result.nemo,
            "verdict": result.verdict,
            "caught_by": result.caught_by,
            "error": result.error,
        },
        "records": _jsonable(result.case),
        "model_run": _jsonable(result.llm),
        "governance": governance,
    }


class BatchStore:
    """Small SQLite repository optimized for one writer and fast UI readers."""

    def __init__(self, path: Path | str = DEFAULT_BATCH_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    total_cases INTEGER NOT NULL,
                    processed_cases INTEGER NOT NULL DEFAULT 0,
                    continue_count INTEGER NOT NULL DEFAULT 0,
                    review_count INTEGER NOT NULL DEFAULT 0,
                    block_count INTEGER NOT NULL DEFAULT 0,
                    error_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS batches_domain_started
                    ON batches(domain, started_at DESC);
                CREATE TABLE IF NOT EXISTS case_results (
                    batch_id TEXT NOT NULL REFERENCES batches(batch_id),
                    sequence INTEGER NOT NULL,
                    source_row INTEGER NOT NULL,
                    case_id TEXT,
                    subject TEXT NOT NULL,
                    disposition TEXT,
                    trace_check TEXT,
                    nemo TEXT,
                    verdict TEXT,
                    caught_by TEXT NOT NULL,
                    error TEXT,
                    result_json TEXT NOT NULL,
                    stored_at TEXT NOT NULL,
                    PRIMARY KEY(batch_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS results_batch_verdict
                    ON case_results(batch_id, verdict, sequence);
            """)

    def begin_batch(
        self, *, domain: str, mode: str, source_name: str, source_hash: str, total_cases: int
    ) -> str:
        batch_id = str(uuid.uuid4())
        with self._connect() as db:
            db.execute(
                """INSERT INTO batches(
                       batch_id, domain, mode, source_name, source_sha256, status, started_at, total_cases
                   ) VALUES (?, ?, ?, ?, ?, 'PROCESSING', ?, ?)""",
                (batch_id, domain, mode, source_name, source_hash, _utc_now(), total_cases),
            )
        return batch_id

    def save_case(self, batch_id: str, sequence: int, result: Any, document: dict[str, Any]) -> None:
        summary = document["summary"]
        with self._connect() as db:
            db.execute(
                """INSERT OR REPLACE INTO case_results(
                       batch_id, sequence, source_row, case_id, subject, disposition, trace_check,
                       nemo, verdict, caught_by, error, result_json, stored_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    batch_id, sequence, int(summary["row"]), summary.get("case_id"), summary["subject"],
                    summary.get("disposition"), summary.get("trace_check"), summary.get("nemo"),
                    summary.get("verdict"), summary.get("caught_by") or "-", summary.get("error"),
                    json.dumps(document, separators=(",", ":"), ensure_ascii=False), _utc_now(),
                ),
            )
            db.execute(
                """UPDATE batches SET
                       processed_cases=(SELECT COUNT(*) FROM case_results WHERE batch_id=?),
                       continue_count=(SELECT COUNT(*) FROM case_results WHERE batch_id=? AND verdict='CONTINUE'),
                       review_count=(SELECT COUNT(*) FROM case_results WHERE batch_id=? AND verdict='HUMAN REVIEW'),
                       block_count=(SELECT COUNT(*) FROM case_results WHERE batch_id=? AND verdict='BLOCK'),
                       error_count=(SELECT COUNT(*) FROM case_results WHERE batch_id=? AND error IS NOT NULL)
                   WHERE batch_id=?""",
                (batch_id, batch_id, batch_id, batch_id, batch_id, batch_id),
            )

    def finish_batch(self, batch_id: str, status: str = "COMPLETE") -> None:
        with self._connect() as db:
            db.execute(
                """UPDATE batches SET status=?, completed_at=?,
                       processed_cases=(SELECT COUNT(*) FROM case_results WHERE batch_id=?),
                       continue_count=(SELECT COUNT(*) FROM case_results WHERE batch_id=? AND verdict='CONTINUE'),
                       review_count=(SELECT COUNT(*) FROM case_results WHERE batch_id=? AND verdict='HUMAN REVIEW'),
                       block_count=(SELECT COUNT(*) FROM case_results WHERE batch_id=? AND verdict='BLOCK'),
                       error_count=(SELECT COUNT(*) FROM case_results WHERE batch_id=? AND error IS NOT NULL)
                   WHERE batch_id=?""",
                (status, _utc_now(), batch_id, batch_id, batch_id, batch_id, batch_id, batch_id),
            )

    def latest_batch(self, domain: str, include_processing: bool = True) -> StoredBatch | None:
        condition = "" if include_processing else " AND status='COMPLETE'"
        with self._connect() as db:
            row = db.execute(
                f"SELECT * FROM batches WHERE domain=?{condition} ORDER BY started_at DESC LIMIT 1",
                (domain,),
            ).fetchone()
        return StoredBatch(**dict(row)) if row is not None else None

    def case_summaries(self, batch_id: str, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT sequence, source_row, case_id, subject, disposition, trace_check, nemo,
                          verdict, caught_by, error, stored_at
                   FROM case_results WHERE batch_id=? ORDER BY sequence LIMIT ?""",
                (batch_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def case_document(self, batch_id: str, sequence: int) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT result_json FROM case_results WHERE batch_id=? AND sequence=?",
                (batch_id, sequence),
            ).fetchone()
        return json.loads(row["result_json"]) if row is not None else None


__all__ = [
    "BatchStore",
    "DEFAULT_BATCH_DB",
    "StoredBatch",
    "healthcare_result_document",
    "source_sha256",
]
