from __future__ import annotations

from typing import Any


TRIAL_TEXT_FIELDS = [
    "title",
    "brief_title",
    "official_title",
    "brief_summary",
    "detailed_description",
    "condition",
    "conditions",
    "intervention_name",
    "interventions",
    "eligibility_criteria",
    "primary_outcome",
    "keywords",
    "mesh_terms_conditions",
    "mesh_terms_interventions",
    "sites_text",
    "general_keywords",
    "drug_name",
    "drug_keywords",
    "text",
]


def build_trial_text(source: dict[str, Any], max_chars: int = 8000) -> str:
    parts: list[str] = []
    for field in TRIAL_TEXT_FIELDS:
        value = source.get(field)
        if value is None:
            continue
        if isinstance(value, list):
            text = "; ".join(str(v) for v in value if v)
        else:
            text = str(value)
        text = " ".join(text.split())
        if text:
            parts.append(f"{field}: {text}")
    return "\n".join(parts)[:max_chars]
