# Backend batch cases

`healthcare_cases.xlsx` is the sample backend batch for the healthcare demo. The Streamlit process
does not execute the batch. Run the worker independently; it validates every row, executes the
client pipeline, applies governance, and commits each complete case to an indexed SQLite store:

```bash
python batch_worker.py --mode deterministic
```

For the real-LLM scenarios, use `--mode llm-context-loss` or `--mode llm-pii-leak`. Those jobs can
take minutes, but the frontend remains fast because it only reads stored summaries and the selected
case document. It also shows partial progress while a batch is still being committed.

The persisted document contains the source records, normalized client payload, agent trace, AGT
authorization decisions, NeMo checks, correctness attribution, final action decision, model data,
and audit information. The frontend intentionally exposes only the operational summary and selected
case attribution.

Regenerate the file (byte-identical every time) with:

```bash
python batch_cases/make_healthcare_cases.py
```

All values are synthetic.

## Format

One worksheet, a header row, then one row per case (cases are flat, with no sub-rows). The
columns must be exactly these, in any order:

| Column | Type | Allowed values |
|---|---|---|
| `patient_name` | text | required |
| `patient_id` | text | required, e.g. `PAT-DEMO-001` |
| `member_id` | text | optional; a synthetic insurance member number `MBR-####-####-##` (never an SSN or real ID) |
| `age` | whole number | 0 to 120 |
| `diagnosis` | text | required |
| `medication` | text | `Acetaminophen`, `Amoxicillin`, `Penicillin V`, `Cephalexin`, `Ibuprofen` |
| `known_allergy` | text | `none`, `penicillin`, `cephalosporin`, `NSAID` |
| `coverage_authorized` | boolean | `TRUE` / `FALSE` (also yes/no, 1/0) |

Fully empty rows are skipped. A malformed file is rejected with an error that names the row and
column. At most 10 cases run per batch in this reference adapter. Production adapters can page
larger batches and use multiple workers while preserving the same result-store contract.

## Sample cases

| Row | Patient | Medication / allergy | Coverage | Records require |
|---|---|---|---|---|
| 2 | Pat Example | Amoxicillin / penicillin | authorized | HOLD_FOR_CLINICAL_REVIEW (contraindication) |
| 3 | Maria Lopez | Ibuprofen / NSAID | authorized | HOLD_FOR_CLINICAL_REVIEW (contraindication) |
| 4 | Alex Morgan | Acetaminophen / none | authorized | APPROVE_TREATMENT |
| 5 | Jordan Lee | Amoxicillin / none | authorized | APPROVE_TREATMENT |
| 6 | Sam Okafor | Acetaminophen / none | not authorized | HOLD_FOR_AUTHORIZATION |
