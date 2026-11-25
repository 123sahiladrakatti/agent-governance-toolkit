"""Offline backend worker for materializing governed batch results.

Run this independently from Streamlit.  The UI never executes the expensive
pipeline; it reads the result rows this worker commits to SQLite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from batch_store import BatchStore, DEFAULT_BATCH_DB, healthcare_result_document, source_sha256
from client_pipeline import load_pipeline_definition
from healthcare.batch import BATCH_LIMIT, BATCH_MODES, DEFAULT_BATCH_FILE, load_healthcare_cases, run_healthcare_batch

HERE = Path(__file__).parent
HEALTHCARE_PIPELINE = HERE / "client_pipelines" / "healthcare_pipeline.json"


def process_healthcare_batch(
    *,
    source: Path | str = DEFAULT_BATCH_FILE,
    database: Path | str = DEFAULT_BATCH_DB,
    mode: str = "deterministic",
    content_checker=None,
) -> str:
    """Ingest, govern, and durably store one healthcare batch."""

    source = Path(source)
    cases = load_healthcare_cases(source)
    selected = cases[:BATCH_LIMIT]
    store = BatchStore(database)
    batch_id = store.begin_batch(
        domain="Healthcare",
        mode=mode,
        source_name=source.name,
        source_hash=source_sha256(source),
        total_cases=len(selected),
    )
    definition = load_pipeline_definition(HEALTHCARE_PIPELINE)

    def persist(position, result):
        store.save_case(batch_id, position, result, healthcare_result_document(result, mode))
        print(f"stored {position}/{len(selected)}: {result.patient_name} -> {result.verdict or 'ERROR'}", flush=True)

    try:
        run_healthcare_batch(
            selected,
            mode,
            definition,
            on_result=persist,
            content_checker=content_checker,
        )
    except BaseException:
        store.finish_batch(batch_id, "FAILED")
        raise
    store.finish_batch(batch_id)
    return batch_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Govern a backend batch and materialize UI-ready results.")
    parser.add_argument("--source", type=Path, default=DEFAULT_BATCH_FILE)
    parser.add_argument("--database", type=Path, default=DEFAULT_BATCH_DB)
    parser.add_argument("--mode", choices=tuple(BATCH_MODES), default="deterministic")
    args = parser.parse_args()
    batch_id = process_healthcare_batch(source=args.source, database=args.database, mode=args.mode)
    print(f"batch complete: {batch_id}\ndatabase: {args.database}")


if __name__ == "__main__":
    main()


__all__ = ["process_healthcare_batch"]
