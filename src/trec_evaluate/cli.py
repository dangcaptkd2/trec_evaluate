from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import load_config
from .download import download_file
from .es import ElasticsearchHttpClient, discover_text_fields
from .experiment import CONFIGS, run_experiment
from .export import export_tables
from .metrics import collect_metrics, run_trec_eval, write_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trec-evaluate")
    parser.add_argument("--config-file", default="configs/trec2023.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_download = subparsers.add_parser("download-data")
    p_download.add_argument("--overwrite", action="store_true")

    p_inspect = subparsers.add_parser("inspect-es")
    p_inspect.add_argument("--es-url")
    p_inspect.add_argument("--es-index")

    p_run = subparsers.add_parser("run-experiment")
    p_run.add_argument("--config", choices=["all", *sorted(CONFIGS)], required=True)
    p_run.add_argument("--es-url")
    p_run.add_argument("--es-index")
    p_run.add_argument("--run-dir")
    p_run.add_argument("--limit-topics", type=int)
    p_run.add_argument("--llm-provider")
    p_run.add_argument("--llm-model")
    p_run.add_argument("--llm-workers", type=int, help="Number of parallel LLM scoring calls for reranking")
    p_run.add_argument("--no-progress", action="store_true", help="Disable experiment progress logs")

    p_eval = subparsers.add_parser("eval-runs")
    p_eval.add_argument("--run-dir", default="runs/latest")
    p_eval.add_argument("--trec-eval")
    p_eval.add_argument("--qrels")

    p_export = subparsers.add_parser("export-tables")
    p_export.add_argument("--run-dir", required=True)

    args = parser.parse_args(argv)
    config = load_config(args.config_file)

    try:
        if args.command == "download-data":
            return _download(config, overwrite=args.overwrite)
        if args.command == "inspect-es":
            return _inspect_es(config, es_url=args.es_url, es_index=args.es_index)
        if args.command == "run-experiment":
            result = run_experiment(
                config,
                config_name=args.config,
                es_index=args.es_index,
                es_url=args.es_url,
                run_dir=args.run_dir,
                limit_topics=args.limit_topics,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                llm_workers=args.llm_workers,
                progress=not args.no_progress,
            )
            print(f"Run directory: {result.run_dir}")
            for run_file in result.run_files:
                print(run_file)
            return 0
        if args.command == "eval-runs":
            return _eval_runs(config, run_dir=args.run_dir, trec_eval=args.trec_eval, qrels=args.qrels)
        if args.command == "export-tables":
            outputs = export_tables(args.run_dir)
            for output in outputs:
                print(output)
            return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 1


def _download(config: dict, overwrite: bool = False) -> int:
    data = config["data"]
    topics = download_file(data["topics_url"], data["topics_path"], overwrite=overwrite)
    qrels = download_file(data["qrels_url"], data["qrels_path"], overwrite=overwrite)
    print(f"Downloaded/available: {topics}")
    print(f"Downloaded/available: {qrels}")
    return 0


def _inspect_es(config: dict, es_url: str | None, es_index: str | None) -> int:
    es_cfg = config["elasticsearch"]
    client = ElasticsearchHttpClient(es_url or es_cfg.get("url", "http://localhost:9200"))
    index = es_index or es_cfg.get("index")
    try:
        print(client.indices())
    except RuntimeError as e:
        print(f"Index listing unavailable: {e}")
    mapping = client.mapping(index)
    fields = discover_text_fields(mapping, es_cfg.get("fields", []))
    print(f"Index: {index}")
    print(f"Query mode: {es_cfg.get('query_mode', 'multi_match')}")
    print(f"Document count: {client.count(index)}")
    print(f"Search fields: {', '.join(fields)}")
    sample = client.sample(index)
    source = sample.get("_source", {})
    print(f"Sample ID: {sample.get('_id', '')}")
    print(f"Sample fields: {', '.join(sorted(source.keys())[:40])}")
    return 0


def _eval_runs(config: dict, run_dir: str, trec_eval: str | None, qrels: str | None) -> int:
    run_path = Path(run_dir)
    qrels_path = Path(qrels or config["data"]["qrels_path"])
    trec_eval_path = trec_eval or config.get("trec_eval", {}).get("path", "trec_eval")
    run_files = sorted(run_path.glob("*.run"))
    if not run_files:
        raise RuntimeError(f"No .run files found in {run_path}")
    for run_file in run_files:
        output = run_file.with_name(f"{run_file.stem}_eval.txt")
        run_trec_eval(trec_eval_path, qrels_path, run_file, output)
        print(output)
    rows = collect_metrics(run_path, qrels_path)
    metrics_path = write_csv(run_path / "metrics.csv", rows)
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
