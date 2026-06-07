from __future__ import annotations

from typing import Any


TRIAL_TEXT_FIELDS = [
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
