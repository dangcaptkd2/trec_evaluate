from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


DISPLAY_NAMES = {
    "bm25_only": "BM25 only",
    "bm25_expanded": "BM25 + LLM query expansion",
    "bm25_minilm_l6": "BM25 + MiniLM-L6",
    "bm25_expanded_minilm_l6": "BM25 expanded + MiniLM-L6",
    "bm25_minilm_l12": "BM25 + MiniLM-L12",
    "bm25_expanded_minilm_l12": "BM25 expanded + MiniLM-L12",
    "bm25_medcpt": "BM25 + MedCPT Cross-Encoder",
    "bm25_llm": "BM25 + LLM reranker",
    "bm25_minilm_l12_llm": "BM25 + MiniLM-L12 + LLM",
}


def export_tables(run_dir: str | Path) -> list[Path]:
    base = Path(run_dir)
    metrics = _read_csv(base / "metrics.csv")
    latency = _read_csv(base / "latency.csv")
    latency_by_config = _latency_by_config(latency)
    metrics_by_config = {row["config"]: row for row in metrics}
    table_dir = base / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    outputs.extend(
        _write_all_formats(
            table_dir / "table_trec_eval",
            _trec_eval_rows(metrics_by_config),
            ["Mô hình", "nDCG@10", "P@10", "RR"],
        )
    )
    outputs.extend(
        _write_all_formats(
            table_dir / "table_reranker_selection",
            _reranker_rows(metrics_by_config, latency_by_config),
            ["Cấu hình", "Luồng ứng viên", "nDCG@10", "P@10", "RR", "Recall@100", "Latency/query"],
        )
    )
    outputs.extend(
        _write_all_formats(
            table_dir / "table_ablation",
            _ablation_rows(metrics_by_config),
            ["Cấu hình hệ thống", "nDCG@10", "P@10", "RR"],
        )
    )
    return outputs


def _trec_eval_rows(metrics: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    full = metrics.get("bm25_minilm_l12_llm", {})
    return [
        {"Mô hình": "BM25 (baseline) [Rybinski et al.]", "nDCG@10": "0.6190", "P@10": "0.3300", "RR": "0.5630"},
        {
            "Mô hình": "BM25 (q&d enr.) + TCRR + GPT_t [Rybinski et al.]",
            "nDCG@10": "0.7770",
            "P@10": "0.6970",
            "RR": "0.7830",
        },
        {
            "Mô hình": "Hệ thống của luận văn (BM25 + neural reranker + LLM)",
            "nDCG@10": full.get("nDCG@10", "PENDING"),
            "P@10": full.get("P@10", "PENDING"),
            "RR": full.get("RR", "PENDING"),
        },
    ]


def _reranker_rows(metrics: dict[str, dict[str, str]], latency: dict[str, float]) -> list[dict[str, str]]:
    flows = {
        "bm25_only": "Top 1000 -> Top 10",
        "bm25_expanded": "LLM expansion -> Top 1000 -> Top 10",
        "bm25_minilm_l6": "Top 1000 -> Top 100",
        "bm25_expanded_minilm_l6": "LLM expansion -> Top 1000 -> Top 100",
        "bm25_medcpt": "Top 1000 -> Top 100",
        "bm25_minilm_l12": "Top 1000 -> Top 100",
        "bm25_expanded_minilm_l12": "LLM expansion -> Top 1000 -> Top 100",
        "bm25_minilm_l12_llm": "Top 1000 -> Top 100 -> Top 10",
    }
    rows: list[dict[str, str]] = []
    for config in [
        "bm25_only",
        "bm25_expanded",
        "bm25_minilm_l6",
        "bm25_expanded_minilm_l6",
        "bm25_medcpt",
        "bm25_minilm_l12",
        "bm25_expanded_minilm_l12",
        "bm25_minilm_l12_llm",
    ]:
        row = metrics.get(config, {})
        rows.append(
            {
                "Cấu hình": DISPLAY_NAMES[config],
                "Luồng ứng viên": flows[config],
                "nDCG@10": row.get("nDCG@10", "PENDING"),
                "P@10": row.get("P@10", "PENDING"),
                "RR": row.get("RR", "PENDING"),
                "Recall@100": row.get("Recall@100", "PENDING"),
                "Latency/query": _format_latency(latency.get(config)),
            }
        )
    return rows


def _ablation_rows(metrics: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    configs = [
        ("(1) BM25 only", "bm25_only"),
        ("(2) BM25 + neural reranker tốt nhất (MiniLM-L12)", "bm25_minilm_l12"),
        ("(3) BM25 + LLM reranker", "bm25_llm"),
        ("(4) BM25 + MiniLM-L12 + LLM reranker", "bm25_minilm_l12_llm"),
    ]
    return [
        {
            "Cấu hình hệ thống": label,
            "nDCG@10": metrics.get(config, {}).get("nDCG@10", "PENDING"),
            "P@10": metrics.get(config, {}).get("P@10", "PENDING"),
            "RR": metrics.get(config, {}).get("RR", "PENDING"),
        }
        for label, config in configs
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _latency_by_config(rows: list[dict[str, str]]) -> dict[str, float]:
    totals: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("stage") == "total":
            totals[row["config"]].append(float(row["seconds"]))
    return {config: sum(values) / len(values) for config, values in totals.items() if values}


def _format_latency(value: float | None) -> str:
    return "PENDING" if value is None else f"{value:.2f}s"


def _write_all_formats(prefix: Path, rows: list[dict[str, str]], headers: list[str]) -> list[Path]:
    return [
        _write_csv(prefix.with_suffix(".csv"), rows, headers),
        _write_md(prefix.with_suffix(".md"), rows, headers),
        _write_tex(prefix.with_suffix(".tex"), rows, headers),
    ]


def _write_csv(path: Path, rows: list[dict[str, str]], headers: list[str]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_md(path: Path, rows: list[dict[str, str]], headers: list[str]) -> Path:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(header, "") for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_tex(path: Path, rows: list[dict[str, str]], headers: list[str]) -> Path:
    colspec = "|" + "|".join(["l"] + ["c"] * (len(headers) - 1)) + "|"
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\hline", " & ".join(_tex_escape(h) for h in headers) + r" \\", "\\hline"]
    for row in rows:
        lines.append(" & ".join(_tex_escape(row.get(header, "")) for header in headers) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _tex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )
