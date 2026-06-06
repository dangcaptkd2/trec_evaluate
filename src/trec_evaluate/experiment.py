from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any

from .cache import key_for
from .es import ElasticsearchHttpClient, discover_text_fields
from .llm import LlmReranker, llm_rerank_candidates
from .query_expansion import QueryExpander
from .rerank import Candidate, CrossEncoderReranker, rerank_candidates
from .runfile import write_run
from .topics import Topic, parse_topics
from .trial_text import build_trial_text


ALL_CONFIGS = (
    "bm25_only",
    "bm25_expanded",
    "bm25_minilm_l6",
    "bm25_medcpt",
    "bm25_minilm_l12",
    "bm25_llm",
    "bm25_minilm_l12_llm",
)

CONFIGS = {
    "bm25_only",
    "bm25_expanded",
    "bm25_minilm_l6",
    "bm25_minilm_l12",
    "bm25_medcpt",
    "bm25_llm",
    "bm25_minilm_l12_llm",
}


@dataclass
class ExperimentResult:
    run_dir: Path
    run_files: list[Path]
    latency_rows: list[dict[str, str]]


def make_run_dir(output_dir: str | Path, requested: str | Path | None = None) -> Path:
    if requested:
        path = Path(requested)
    else:
        path = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    latest = Path(output_dir) / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(path.resolve(), target_is_directory=True)
    except OSError:
        pass
    return path


def run_experiment(
    config: dict[str, Any],
    config_name: str,
    es_index: str | None = None,
    es_url: str | None = None,
    run_dir: str | Path | None = None,
    limit_topics: int | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> ExperimentResult:
    names = list(ALL_CONFIGS) if config_name == "all" else [config_name]
    unknown = [name for name in names if name not in CONFIGS]
    if unknown:
        raise ValueError(f"Unknown config(s): {', '.join(unknown)}")

    data_cfg = config["data"]
    exp_cfg = config["experiment"]
    es_cfg = config["elasticsearch"]
    model_cfg = config["models"]
    llm_cfg = config.get("llm", {})

    output_dir = exp_cfg.get("output_dir", "runs")
    output = make_run_dir(output_dir, run_dir)
    topics = parse_topics(data_cfg["topics_path"])
    if limit_topics is not None:
        topics = topics[:limit_topics]

    client = ElasticsearchHttpClient(es_url or es_cfg.get("url", "http://localhost:9200"))
    index = es_index or es_cfg.get("index")
    mapping = client.mapping(index)
    fields = _apply_field_boosts(discover_text_fields(mapping, es_cfg.get("fields", ["text"])), es_cfg)
    query_mode = es_cfg.get("query_mode", "multi_match")

    cache_dir = Path(exp_cfg.get("cache_dir", "cache"))
    top_k = int(exp_cfg.get("top_k_bm25", 1000))
    neural_window = int(exp_cfg.get("neural_window", 1000))
    neural_keep = int(exp_cfg.get("neural_keep", 100))
    llm_direct_window = int(exp_cfg.get("llm_direct_window", 100))
    llm_final_window = int(exp_cfg.get("llm_final_window", 100))
    llm_keep = int(exp_cfg.get("llm_keep", 10))
    expansion_cfg = config.get("query_expansion", {})

    run_files: list[Path] = []
    latency_rows: list[dict[str, str]] = []
    bm25_cache: dict[str, list[Candidate]] = {}

    for name in names:
        entries_by_topic: dict[str, list[tuple[str, float]]] = {}
        query_expander = (
            QueryExpander(
                provider=expansion_cfg.get("provider", llm_cfg.get("provider", "openai")),
                model=expansion_cfg.get("model", llm_cfg.get("model", "gpt-4.1-nano")),
                cache_dir=cache_dir / "query_expansion",
                temperature=float(expansion_cfg.get("temperature", 0)),
                max_retries=int(expansion_cfg.get("max_retries", llm_cfg.get("max_retries", 3))),
                max_terms_per_category=int(expansion_cfg.get("max_terms_per_category", 8)),
                use_related_concepts=bool(expansion_cfg.get("use_related_concepts", True)),
            )
            if _uses_query_expansion(name, expansion_cfg)
            else None
        )
        neural_model = _neural_model_for_config(name, model_cfg)
        neural_reranker = (
            CrossEncoderReranker(
                neural_model,
                cache_dir=cache_dir / "cross_encoder",
                batch_size=int(exp_cfg.get("cross_encoder_batch_size", 16)),
                max_length=int(exp_cfg.get("cross_encoder_max_length", 512)),
            )
            if neural_model
            else None
        )
        llm_reranker = (
            LlmReranker(
                provider=llm_provider or llm_cfg.get("provider", "openai"),
                model=llm_model or llm_cfg.get("model", "gpt-4.1-nano"),
                cache_dir=cache_dir / "llm",
                temperature=float(llm_cfg.get("temperature", 0)),
                max_retries=int(llm_cfg.get("max_retries", 3)),
            )
            if name in {"bm25_llm", "bm25_minilm_l12_llm"}
            else None
        )

        for topic in topics:
            started = time.perf_counter()
            query = topic.to_query_text(include_template=False)
            if query_expander is not None:
                expansion_started = time.perf_counter()
                query = query_expander.expand(topic.number, query)
                latency_rows.append(_latency_row(name, topic.number, "query_expansion", expansion_started, 1))
            bm25_key = key_for(topic.number, query)
            candidates = bm25_cache.get(bm25_key)
            if candidates is None:
                search_started = time.perf_counter()
                candidates = _retrieve_candidates(client, index, fields, query, top_k, query_mode)
                latency_rows.append(_latency_row(name, topic.number, "bm25", search_started, len(candidates)))
                bm25_cache[bm25_key] = candidates

            ranked = candidates
            if neural_reranker is not None:
                neural_started = time.perf_counter()
                ranked = rerank_candidates(query, ranked, neural_reranker, window=neural_window, keep=neural_keep)
                latency_rows.append(_latency_row(name, topic.number, "neural", neural_started, min(neural_keep, len(ranked))))

            if llm_reranker is not None:
                llm_started = time.perf_counter()
                window = llm_direct_window if name == "bm25_llm" else llm_final_window
                ranked = llm_rerank_candidates(query, ranked, llm_reranker, window=window, keep=llm_keep)
                latency_rows.append(_latency_row(name, topic.number, "llm", llm_started, min(llm_keep, len(ranked))))

            if name == "bm25_only":
                entries_by_topic[topic.number] = [(candidate.doc_id, candidate.score) for candidate in ranked]
            else:
                entries_by_topic[topic.number] = [
                    (candidate.doc_id, 1000.0 - idx)
                    for idx, candidate in enumerate(ranked)
                ]
            latency_rows.append(_latency_row(name, topic.number, "total", started, len(ranked)))

        run_path = output / f"{name}.run"
        write_run(run_path, entries_by_topic, run_name=name)
        run_files.append(run_path)

    _write_latency(output / "latency.csv", latency_rows)
    return ExperimentResult(output, run_files, latency_rows)


def _retrieve_candidates(
    client: ElasticsearchHttpClient,
    index: str,
    fields: list[str],
    query: str,
    top_k: int,
    query_mode: str,
) -> list[Candidate]:
    hits = client.search(index=index, query_text=query, fields=fields, size=top_k, query_mode=query_mode)
    candidates: list[Candidate] = []
    for hit in hits:
        doc_id = str(hit.get("_id"))
        source = hit.get("_source") or {}
        candidates.append(Candidate(doc_id=doc_id, score=float(hit.get("_score", 0.0)), text=build_trial_text(source)))
    return candidates


def _apply_field_boosts(fields: list[str], es_cfg: dict[str, Any]) -> list[str]:
    if es_cfg.get("query_mode") == "text_match":
        return [field.split("^", 1)[0] for field in fields]
    boosts = es_cfg.get("bm25", {}).get("field_boosts", {})
    boosted_fields: list[str] = []
    for field in fields:
        base = field.split("^", 1)[0]
        boost = boosts.get(base)
        boosted_fields.append(f"{base}^{boost}" if boost and boost != 1 else base)
    return boosted_fields


def _uses_query_expansion(name: str, expansion_cfg: dict[str, Any]) -> bool:
    if name == "bm25_expanded":
        return bool(expansion_cfg.get("enabled", True))
    return False


def _neural_model_for_config(name: str, model_cfg: dict[str, Any]) -> str | None:
    return {
        "bm25_minilm_l6": model_cfg.get("minilm_l6"),
        "bm25_minilm_l12": model_cfg.get("minilm_l12"),
        "bm25_medcpt": model_cfg.get("medcpt"),
        "bm25_minilm_l12_llm": model_cfg.get("minilm_l12"),
    }.get(name)


def _latency_row(config: str, topic: str, stage: str, started: float, candidates: int) -> dict[str, str]:
    return {
        "config": config,
        "topic": topic,
        "stage": stage,
        "candidates": str(candidates),
        "seconds": f"{time.perf_counter() - started:.6f}",
    }


def _write_latency(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "topic", "stage", "candidates", "seconds"])
        writer.writeheader()
        writer.writerows(rows)
