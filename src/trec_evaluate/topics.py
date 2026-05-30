from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Topic:
    number: str
    template: str
    fields: dict[str, str]

    def to_query_text(self, include_template: bool = False) -> str:
        lines: list[str] = []
        if include_template and self.template.strip():
            lines.append(self.template.strip())
        for key, value in self.fields.items():
            if value and value.strip():
                lines.append(f"{key}: {value.strip()}")
        return "\n".join(lines).strip()


def parse_topics(path: str | Path) -> list[Topic]:
    root = ET.parse(path).getroot()
    topics: list[Topic] = []
    for topic in root.findall("topic"):
        fields: dict[str, str] = {}
        for field in topic.findall("field"):
            name = field.get("name", "").strip()
            if not name:
                continue
            fields[name] = (field.text or "").strip()
        topics.append(
            Topic(
                number=(topic.get("number") or "").strip(),
                template=(topic.get("template") or "").strip(),
                fields=fields,
            )
        )
    return topics

