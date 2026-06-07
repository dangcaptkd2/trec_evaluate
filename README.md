# TREC 2023 Clinical Trial Matching Experiment Pipeline

This repository builds TREC-formatted runs for Clinical Trials 2023, evaluates them
with `trec_eval`, and exports thesis-ready result tables.

## Quick Start

```bash
uv sync --extra test
uv run trec-evaluate download-data
uv run trec-evaluate inspect-es
uv run trec-evaluate run-experiment --config bm25_only
uv run trec-evaluate eval-runs --trec-eval /path/to/trec_eval
uv run trec-evaluate export-tables --run-dir runs/latest
```

The default configuration uses the CTnlp-built Elasticsearch index
`trec2023_ctnlp` at the configured RunPod URL. BM25 uses a single-field
`match` query over the trial `text` field, and neural rerankers receive the same
`text` field as the trial representation. Override `--es-url` or `--es-index`
only if you rebuild the data into a different index.

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

BM25 with cached LLM query expansion, then MiniLM reranking. If the expanded
query already exists under `cache/query_expansion/`, the LLM is not called again:

```bash
uv sync --extra rerank --extra llm
uv run --extra rerank --extra llm trec-evaluate run-experiment --config bm25_expanded_minilm_l6 --limit-topics 1
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
```

BM25 with cached LLM query expansion, then MiniLM-L12 reranker plus OpenAI LLM
reranker. The default LLM model is `gpt-4.1-nano`:

```bash
uv sync --extra rerank --extra llm
uv run --extra rerank --extra llm trec-evaluate run-experiment --config bm25_expanded_minilm_l12_llm --limit-topics 1
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
```

Available experiment configs:

- `bm25_expanded`: BM25 after cached LLM synonym/related-concept expansion
- `bm25_expanded_minilm_l6`: cached LLM query expansion + MiniLM-L6 reranker
- `bm25_expanded_medcpt`: cached LLM query expansion + `ncbi/MedCPT-Cross-Encoder`
- `bm25_expanded_bge_reranker_base`: cached LLM query expansion + `BAAI/bge-reranker-base`
- `bm25_expanded_bge_reranker_large`: cached LLM query expansion + `BAAI/bge-reranker-large`
- `bm25_expanded_bge_reranker_v2_m3`: cached LLM query expansion + `BAAI/bge-reranker-v2-m3`
- `bm25_expanded_mxbai_rerank_xsmall`: cached LLM query expansion + `mixedbread-ai/mxbai-rerank-xsmall-v1`
- `bm25_expanded_mxbai_rerank_base`: cached LLM query expansion + `mixedbread-ai/mxbai-rerank-base-v1`
- `bm25_expanded_mxbai_rerank_large`: cached LLM query expansion + `mixedbread-ai/mxbai-rerank-large-v1`
- `bm25_expanded_minilm_l12`: cached LLM query expansion + MiniLM-L12 reranker
- `bm25_expanded_llm`: cached LLM query expansion + LLM reranker
- `bm25_expanded_minilm_l12_llm`: cached LLM query expansion + MiniLM-L12 + LLM reranker

## Outputs

Runs are written under `runs/<timestamp>/`:

- `<config>.run`
- `<config>_eval.txt`
- `metrics.csv`
- `latency.csv`
- `tables/table_trec_eval.{csv,md,tex}`
- `tables/table_reranker_selection.{csv,md,tex}`
- `tables/table_ablation.{csv,md,tex}`
