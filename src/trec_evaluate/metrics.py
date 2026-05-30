from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from .qrels import parse_qrels, recall_at_k
from .runfile import doc_ids_by_topic


def run_trec_eval(trec_eval_path: str | Path, qrels_path: str | Path, run_path: str | Path, output_path: str | Path) -> Path:
    cmd = [
        str(trec_eval_path),
        "-q",
        "-c",
        "-M1000",
        "-m",
        "ndcg_cut.10,1000",
        "-m",
        "P.10",
        "-m",
        "recip_rank",
        str(qrels_path),
        str(run_path),
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    output = Path(output_path)
    output.write_text(result.stdout, encoding="utf-8")
    return output


def parse_trec_eval_output(path: str | Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            metric, topic, value = parts
            if topic == "all":
                metrics[metric] = float(value)
    return metrics


def collect_metrics(run_dir: str | Path, qrels_path: str | Path) -> list[dict[str, str]]:
    run_path = Path(run_dir)
    qrels = parse_qrels(qrels_path)
    rows: list[dict[str, str]] = []
    for eval_file in sorted(run_path.glob("*_eval.txt")):
        config = eval_file.name.removesuffix("_eval.txt")
        run_file = run_path / f"{config}.run"
        parsed = parse_trec_eval_output(eval_file)
        recall100 = recall_at_k(doc_ids_by_topic(run_file), qrels, k=100) if run_file.exists() else 0.0
        rows.append(
            {
                "config": config,
                "nDCG@10": f"{parsed.get('ndcg_cut_10', 0.0):.4f}",
                "P@10": f"{parsed.get('P_10', 0.0):.4f}",
                "RR": f"{parsed.get('recip_rank', 0.0):.4f}",
                "nDCG@1000": f"{parsed.get('ndcg_cut_1000', 0.0):.4f}",
                "Recall@100": f"{recall100:.4f}",
            }
        )
    return rows


def write_csv(path: str | Path, rows: list[dict[str, str]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return output
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output

