# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Run the Healthcare MAF adapter example with native ACS."""

from pathlib import Path
import asyncio
import re
from agent_control_specification import AgentControl


class HealthcarePolicy:
    """Evaluate the healthcare safety rule for the ACS dispatcher."""

    def evaluate(self, invocation: dict) -> dict:
        value = str(invocation["input"]["policy_target"]["value"])
        blocked = re.search(
            r"(?i)(MRN[:\s]*\d{6,}|\b\d{3}-\d{2}-\d{4}\b)",
            value,
        )
        if blocked:
            return {"decision": "deny", "reason": "healthcare_safety"}
        return {"decision": "allow", "reason": "safe"}


def main() -> None:
    root = Path(__file__).resolve().parent
    runtime = AgentControl.from_path(
        str(root / "policies" / "manifest.yaml"),
        policy_dispatcher=HealthcarePolicy(),
    )
    async def evaluate_requests() -> None:
        for prompt in ("Review the current request", "Patient MRN: 123456"):
            result = await runtime.evaluate_intervention_point(
                "input",
                {"input": {"body": prompt}},
            )
            print(prompt, "->", result.verdict.decision.value, result.verdict.reason)

    asyncio.run(evaluate_requests())


if __name__ == "__main__":
    main()
