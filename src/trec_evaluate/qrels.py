from __future__ import annotations

from collections import defaultdict
from pathlib import Path


Qrels = dict[str, dict[str, int]]


def parse_qrels(path: str | Path) -> Qrels:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            topic, _, doc_id, rel = parts[:4]
            qrels[topic][doc_id] = int(rel)
    return dict(qrels)


def recall_at_k(run_by_topic: dict[str, list[str]], qrels: Qrels, k: int = 100, relevant_labels: set[int] | None = None) -> float:
    labels = relevant_labels or {1, 2}
    scores: list[float] = []
    for topic, judgments in qrels.items():
        relevant = {doc_id for doc_id, rel in judgments.items() if rel in labels}
        if not relevant:
            continue
        retrieved = set(run_by_topic.get(topic, [])[:k])
        scores.append(len(retrieved & relevant) / len(relevant))
    return sum(scores) / len(scores) if scores else 0.0

