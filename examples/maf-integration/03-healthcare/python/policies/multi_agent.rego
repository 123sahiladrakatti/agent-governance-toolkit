# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
package agt.examples.maf.multi_agent

import rego.v1

# The Python dispatcher adds stateful identity, routing, task, PHI, and budget
# checks. This bundle documents the same intervention contract for ACS.
blocked if regex.match(`(?i)(MRN[:\s]*\d{6,}|\b\d{3}-\d{2}-\d{4}\b)`, sprintf("%v", [input.policy_target.value]))

result := {"decision": "deny", "reason": "healthcare_phi_detected"} if blocked
result := {"decision": "allow", "reason": "handoff_policy_passed"} if not blocked
