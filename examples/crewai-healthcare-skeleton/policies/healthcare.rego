# Baseline "regular governance" for the two-agent intake scenario.
#
# Deliberately conventional: per-call pattern matching over whatever payload the
# current intervention point hands it. It has no memory of earlier calls, no
# notion of which agent is acting, and no view of the handoff between them --
# those are the limits this baseline exists to expose.

package agt.examples.crewai_healthcare

import rego.v1

default result := {"decision": "allow", "reason": "no_phi_detected"}

target := lower(sprintf("%v", [input.policy_target.value]))

# --- Direct patient identifiers (classic DLP) -----------------------------
has_mrn if regex.match(`mrn["':\s]*[0-9]{6,}`, target)

has_ssn if regex.match(`[0-9]{3}-[0-9]{2}-[0-9]{4}`, target)

has_identifier if has_mrn

has_identifier if has_ssn

# --- Clinical PHI ---------------------------------------------------------
has_clinical if contains(target, "active_conditions")

has_clinical if contains(target, "medications")

has_clinical if contains(target, "allergies")

# --- Verdicts -------------------------------------------------------------
result := {
	"decision": "deny",
	"reason": "phi_identifier_exposed",
	"message": "Payload contains a direct patient identifier (MRN or SSN).",
} if has_identifier

result := {
	"decision": "warn",
	"reason": "clinical_phi_in_payload",
	"message": "Payload carries clinical PHI (conditions, medications or allergies).",
} if {
	not has_identifier
	has_clinical
}
