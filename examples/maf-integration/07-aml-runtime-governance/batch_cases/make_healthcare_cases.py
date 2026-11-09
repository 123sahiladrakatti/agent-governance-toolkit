"""Generate batch_cases/healthcare_cases.xlsx, the backend sample for the healthcare batch.

    python batch_cases/make_healthcare_cases.py

All values are synthetic. Member IDs use an obviously fake MBR-####-####-## format.
The output is byte-identical on every run: openpyxl stamps the save time into the
workbook properties and every zip entry, so both are pinned to a fixed timestamp.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

from openpyxl import Workbook

OUTPUT = Path(__file__).resolve().parent / "healthcare_cases.xlsx"
COLUMNS = (
    "patient_name", "patient_id", "member_id", "age", "diagnosis",
    "medication", "known_allergy", "coverage_authorized",
)
# A mix: two medication/allergy matches (contraindication), clean cases, and one
# clean case without prior authorization.
CASES = (
    ("Pat Example", "PAT-DEMO-001", "MBR-8847-2291-05", 45, "bacterial infection", "Amoxicillin", "penicillin", True),
    ("Maria Lopez", "PAT-88213", "MBR-5190-3364-12", 62, "post-operative pain", "Ibuprofen", "NSAID", True),
    ("Alex Morgan", "PAT-40517", "MBR-2276-8051-47", 34, "strep throat", "Acetaminophen", "none", True),
    ("Jordan Lee", "PAT-73190", "MBR-3318-9045-26", 28, "sinus infection", "Amoxicillin", "none", True),
    ("Sam Okafor", "PAT-12077", "MBR-6623-1408-90", 51, "skin infection", "Acetaminophen", "none", False),
)
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
FIXED_ISO_TIME = "2026-01-01T00:00:00Z"


def build_workbook_bytes() -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "cases"
    sheet.append(COLUMNS)
    for case in CASES:
        sheet.append(case)
    for column, width in zip("ABCDEFGH", (16, 14, 20, 6, 22, 16, 15, 20)):
        sheet.column_dimensions[column].width = width
    buffer = io.BytesIO()
    workbook.save(buffer)
    return _normalize(buffer.getvalue())


def _normalize(data: bytes) -> bytes:
    """Rewrite the archive with fixed timestamps so the bytes are reproducible."""
    source = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            content = source.read(info.filename)
            if info.filename == "docProps/core.xml":
                content = re.sub(
                    rb"(<dcterms:(created|modified)[^>]*>)[^<]*(</dcterms:\2>)",
                    lambda m: m.group(1) + FIXED_ISO_TIME.encode() + m.group(3),
                    content,
                )
            entry = zipfile.ZipInfo(info.filename, date_time=FIXED_ZIP_TIME)
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            target.writestr(entry, content)
    return out.getvalue()


def main() -> None:
    OUTPUT.write_bytes(build_workbook_bytes())
    print(f"wrote {OUTPUT} ({len(CASES)} cases)")


if __name__ == "__main__":
    main()
