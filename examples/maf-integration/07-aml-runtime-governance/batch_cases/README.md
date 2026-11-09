# Backend batch cases

`healthcare_cases.xlsx` is the sample batch for the healthcare demo. In the app, choose the
Healthcare pipeline, load it, and use **Load & govern sample cases**. Every row runs through the
chosen real-LLM failure mode and the same ingestion and governance path as the single-case form.
There is no upload widget; the app reads this file from the repo.

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
column. At most 10 cases run per batch, because each real-LLM case takes 15-60 seconds.

## Sample cases

| Row | Patient | Medication / allergy | Coverage | Records require |
|---|---|---|---|---|
| 2 | Pat Example | Amoxicillin / penicillin | authorized | HOLD_FOR_CLINICAL_REVIEW (contraindication) |
| 3 | Maria Lopez | Ibuprofen / NSAID | authorized | HOLD_FOR_CLINICAL_REVIEW (contraindication) |
| 4 | Alex Morgan | Acetaminophen / none | authorized | APPROVE_TREATMENT |
| 5 | Jordan Lee | Amoxicillin / none | authorized | APPROVE_TREATMENT |
| 6 | Sam Okafor | Acetaminophen / none | not authorized | HOLD_FOR_AUTHORIZATION |
