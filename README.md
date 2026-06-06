# TREC 2023 Clinical Trial Matching Experiment Pipeline

This repository builds TREC-formatted runs for Clinical Trials 2023, evaluates them
with `trec_eval`, and exports thesis-ready result tables.

## Quick Start

```bash
uv sync --extra test
uv run trec-evaluate download-data
uv run trec-evaluate inspect-es --es-url http://localhost:9200
uv run trec-evaluate run-experiment --config bm25_only --es-index aact
uv run trec-evaluate eval-runs --trec-eval /path/to/trec_eval
uv run trec-evaluate export-tables --run-dir runs/latest
```

On this machine the detected Elasticsearch index is `aact`, so the default
configuration uses that index. Override it with `--es-index` if you rebuild the
data into a different index.

To install the official NIST evaluator locally:

```bash
scripts/install_trec_eval.sh
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
```

Use `--config all` to run all thesis configurations. Neural rerankers require
`uv sync --extra rerank`; LLM reranking requires `uv sync --extra llm` and API keys
in `.env`.

## One-Topic Smoke Tests

BM25 only:

```bash
uv run trec-evaluate run-experiment --config bm25_only --limit-topics 1
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
```

BM25 with cached LLM query expansion. This calls the LLM once per topic and
saves expanded terms under `cache/query_expansion/`:

```bash
uv sync --extra llm
uv run trec-evaluate run-experiment --config bm25_expanded --limit-topics 1
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
```

BM25 plus one neural reranker. Use `bm25_minilm_l12` for the thesis default:

```bash
uv sync --extra rerank
uv run trec-evaluate run-experiment --config bm25_minilm_l12 --limit-topics 1
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
```

BM25 plus MiniLM-L12 reranker plus OpenAI LLM reranker. The default LLM model is
`gpt-4.1-nano`:

```bash
uv sync --extra rerank --extra llm
uv run trec-evaluate run-experiment --config bm25_minilm_l12_llm --limit-topics 1
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
```

Available experiment configs:

- `bm25_expanded`: BM25 after cached LLM synonym/related-concept expansion
- `bm25_minilm_l6`: `cross-encoder/ms-marco-MiniLM-L6-v2`
- `bm25_medcpt`: `ncbi/MedCPT-Cross-Encoder`
- `bm25_minilm_l12`: `cross-encoder/ms-marco-MiniLM-L12-v2`

## Outputs

Runs are written under `runs/<timestamp>/`:

- `<config>.run`
- `<config>_eval.txt`
- `metrics.csv`
- `latency.csv`
- `tables/table_trec_eval.{csv,md,tex}`
- `tables/table_reranker_selection.{csv,md,tex}`
- `tables/table_ablation.{csv,md,tex}`
