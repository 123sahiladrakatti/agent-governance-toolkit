"""Hardcoded fake patient data. No database, no external API, no real PHI.

Every value here is invented for a demo. Patient ids use the P-1xxx range so
they are obviously not real MRNs.
"""

from __future__ import annotations

PATIENT_IDS = ["P-1001", "P-1002", "P-1003"]

DEMOGRAPHICS: dict[str, dict[str, str]] = {
    "P-1001": {
        "name": "Ada Fictional",
        "date_of_birth": "1979-03-14",
        "age": "47",
        "phone": "555-0101",
        "insurance": "MockCare PPO",
        "preferred_language": "English",
    },
    "P-1002": {
        "name": "Bo Placeholder",
        "date_of_birth": "1991-11-02",
        "age": "34",
        "phone": "555-0102",
        "insurance": "SampleShield HMO",
        "preferred_language": "Spanish",
    },
    "P-1003": {
        "name": "Cy Example",
        "date_of_birth": "1965-07-23",
        "age": "61",
        "phone": "555-0103",
        "insurance": "MockCare Medicare Advantage",
        "preferred_language": "English",
    },
}

# Records carry a medical record number, the way a real chart would. These are
# invented, but they are the kind of direct identifier a PHI policy looks for --
# so the baseline governance layer has something real to fire on.
RECORDS: dict[str, dict[str, object]] = {
    "P-1001": {
        "mrn": "480915",
        "primary_care_provider": "Dr. Imaginary Reyes",
        "active_conditions": ["Type 2 diabetes", "Hypertension"],
        "medications": ["Metformin 500mg", "Lisinopril 10mg"],
        "allergies": ["Penicillin"],
        "last_visit": "2026-06-02",
    },
    "P-1002": {
        "mrn": "551204",
        "primary_care_provider": "Dr. Notional Okafor",
        "active_conditions": ["Seasonal asthma"],
        "medications": ["Albuterol inhaler"],
        "allergies": [],
        "last_visit": "2026-08-19",
    },
    "P-1003": {
        "mrn": "739468",
        "primary_care_provider": "Dr. Hypothetical Singh",
        "active_conditions": ["Atrial fibrillation", "Osteoarthritis"],
        "medications": ["Apixaban 5mg", "Acetaminophen 500mg"],
        "allergies": ["Sulfa drugs"],
        "last_visit": "2026-07-11",
    },
}

SCHEDULE: dict[str, dict[str, str]] = {
    "P-1001": {
        "next_appointment": "2026-09-18 09:30",
        "clinic": "Endocrinology, Suite 200",
        "provider": "Dr. Imaginary Reyes",
        "visit_type": "Diabetes follow-up",
    },
    "P-1002": {
        "next_appointment": "2026-10-01 14:00",
        "clinic": "Pulmonology, Suite 410",
        "provider": "Dr. Notional Okafor",
        "visit_type": "Asthma action plan review",
    },
    "P-1003": {
        "next_appointment": "2026-09-12 11:15",
        "clinic": "Cardiology, Suite 310",
        "provider": "Dr. Hypothetical Singh",
        "visit_type": "Anticoagulation check",
    },
}
