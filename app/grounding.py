from __future__ import annotations

import re
from dataclasses import dataclass


CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class GroundingResult:
    accepted: bool
    violations: list[str]
    citations: list[int]

    def as_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "violations": self.violations,
            "citations": self.citations,
        }


def validate_grounded_answer(answer: str | None, source_count: int) -> GroundingResult:
    if not answer or not answer.strip():
        return GroundingResult(False, ["empty_answer"], [])

    citations = [int(match.group(1)) for match in CITATION_PATTERN.finditer(answer)]
    violations: list[str] = []
    if source_count > 0 and not citations:
        violations.append("missing_citation")
    invalid = [index for index in citations if index < 1 or index > source_count]
    if invalid:
        violations.append("citation_out_of_range")

    return GroundingResult(not violations, violations, citations)
