# Hướng Dẫn Chạy TREC 2023 Experiment

File này gom các lệnh chính để chạy pipeline trong repo:

```bash
cd /Users/phamhoangyen/thinhquyen/trec_evaluate
```

## 1. Cài môi trường

Chạy tối thiểu cho BM25, eval, export table:

```bash
uv sync --extra test
```

Nếu cần chạy neural reranker:

```bash
uv sync --extra rerank
```

Nếu cần chạy neural reranker + LLM reranker:

```bash
uv sync --extra rerank --extra llm
```

## 2. Tải dữ liệu TREC 2023

```bash
uv run trec-evaluate download-data
```

Dữ liệu sẽ nằm ở:

```text
data/trec2023/topics2023.xml
data/trec2023/qrels2023.txt
```

## 3. Cài trec_eval

```bash
scripts/install_trec_eval.sh
```

Binary sau khi build:

```text
tools/trec_eval/trec_eval
```

## 4. Kiểm tra Elasticsearch

Default index hiện tại trong `configs/trec2023.yaml` là `aact`.

```bash
uv run trec-evaluate inspect-es --es-url http://localhost:9200
```

Nếu muốn chỉ rõ index:

```bash
uv run trec-evaluate inspect-es --es-url http://localhost:9200 --es-index aact
```

## 5. Chạy thử 1 topic

BM25 only:

```bash
uv run trec-evaluate run-experiment --config bm25_only --limit-topics 1
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_evaltrec_eval
```

BM25 + một neural reranker, khuyến nghị test `bm25_minilm` trước:

```bash
uv run trec-evaluate run-experiment --config bm25_minilm --limit-topics 1
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_evaltrec_eval
```

BM25 + Jina reranker + OpenAI LLM reranker:

```bash
uv run trec-evaluate run-experiment --config bm25_jina_llm --limit-topics 1
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
```

LLM default đang là:

```text
provider: openai
model: gpt-4.1-nano
```

Nhớ đặt `OPENAI_API_KEY` trong `.env` trước khi chạy config có LLM.

## 6. Các config reranker hiện có

```text
bm25_minilm  -> cross-encoder/ms-marco-MiniLM-L6-v2
bm25_medcpt  -> ncbi/MedCPT-Cross-Encoder
bm25_bge     -> BAAI/bge-reranker-v2-m3
bm25_jina    -> jinaai/jina-reranker-v2-base-multilingual
```

Chạy thử một model bất kỳ:

```bash
uv run trec-evaluate run-experiment --config bm25_bge --limit-topics 1
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
```

## 7. Chạy full BM25 official

Không dùng `--limit-topics`:

```bash
uv run trec-evaluate run-experiment --config bm25_only
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
uv run trec-evaluate export-tables --run-dir runs/latest
```

## 8. Chạy full một neural reranker

Ví dụ Jina:

```bash
uv run trec-evaluate run-experiment --config bm25_jina
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
uv run trec-evaluate export-tables --run-dir runs/latest
```

## 9. Chạy full toàn bộ thí nghiệm

Lệnh này chạy tất cả config:

```bash
uv run trec-evaluate run-experiment --config all
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
uv run trec-evaluate export-tables --run-dir runs/latest
```

Các config trong `all`:

```text
bm25_only
bm25_minilm
bm25_medcpt
bm25_bge
bm25_jina
bm25_llm
bm25_jina_llm
```

## 10. Output cần xem

Sau mỗi lần chạy, output mới nhất nằm ở:

```text
runs/latest/
```

Các file chính:

```text
runs/latest/<config>.run
runs/latest/<config>_eval.txt
runs/latest/metrics.csv
runs/latest/latency.csv
runs/latest/tables/table_trec_eval.csv
runs/latest/tables/table_reranker_selection.csv
runs/latest/tables/table_ablation.csv
```

File `.tex` để copy vào luận văn:

```text
runs/latest/tables/table_trec_eval.tex
runs/latest/tables/table_reranker_selection.tex
runs/latest/tables/table_ablation.tex
```

