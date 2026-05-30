from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunEntry:
    topic: str
    doc_id: str
    rank: int
    score: float
    run_name: str


def write_run(path: str | Path, entries_by_topic: dict[str, list[tuple[str, float]]], run_name: str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for topic in sorted(entries_by_topic.keys(), key=lambda x: int(x) if x.isdigit() else x):
        for rank, (doc_id, score) in enumerate(entries_by_topic[topic], start=1):
            lines.append(f"{topic} Q0 {doc_id} {rank} {float(score):.6f} {run_name}")
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output


def parse_run(path: str | Path) -> dict[str, list[RunEntry]]:
    entries: dict[str, list[RunEntry]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 6:
                raise ValueError(f"Invalid TREC run line {lineno}: expected 6 columns")
            topic, q0, doc_id, rank, score, run_name = parts
            if q0 != "Q0":
                raise ValueError(f"Invalid TREC run line {lineno}: second column must be Q0")
            entries[topic].append(RunEntry(topic, doc_id, int(rank), float(score), run_name))

    for topic, topic_entries in entries.items():
        ranks = [entry.rank for entry in topic_entries]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"Ranks for topic {topic} are not monotonic from 1")
    return dict(entries)


def doc_ids_by_topic(path: str | Path) -> dict[str, list[str]]:
    return {topic: [entry.doc_id for entry in entries] for topic, entries in parse_run(path).items()}

