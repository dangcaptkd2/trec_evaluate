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
uv pip install --reinstall "transformers>=4.40.0,<5"
```

Nếu chỉ cần chạy BM25 với mở rộng truy vấn bằng LLM:

```bash
uv sync --extra llm
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

Default index hiện tại trong `configs/trec2023.yaml` là `trec2023_ctnlp` trên RunPod Elasticsearch. Index này không có field `text` đơn như bản cũ, nên BM25 dùng `query_mode: cross_fields` trên các field CTnlp để mô phỏng một tài liệu lâm sàng gộp.

```bash
uv run trec-evaluate inspect-es
```

Nếu muốn chỉ rõ URL/index khác:

```bash
uv run trec-evaluate inspect-es --es-url https://8t2muxa4qo42wo-9200.proxy.runpod.net --es-index trec2023_ctnlp
```

## 5. Chạy thử 1 topic

BM25 only:

```bash
uv run trec-evaluate run-experiment --config bm25_only --limit-topics 1
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
```

BM25 + mở rộng truy vấn bằng LLM. Bước này gọi LLM một lần cho mỗi topic và cache kết quả ở `cache/query_expansion/`:

```bash
uv run trec-evaluate run-experiment --config bm25_expanded --limit-topics 1
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
```

BM25 + một neural reranker, khuyến nghị test `bm25_minilm_l12` vì đây là reranker mặc định của luận văn:

```bash
uv run trec-evaluate run-experiment --config bm25_minilm_l12 --limit-topics 1
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
```

BM25 + MiniLM-L12 reranker + OpenAI LLM reranker:

```bash
uv run trec-evaluate run-experiment --config bm25_minilm_l12_llm --limit-topics 1
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
```

LLM default đang là:

```text
provider: openai
model: gpt-4.1-nano
```

Nhớ đặt `OPENAI_API_KEY` trong `.env` trước khi chạy config có LLM.

## 6. Các config thí nghiệm hiện có

```text
bm25_expanded  -> BM25 + cached LLM query expansion
bm25_minilm_l6  -> cross-encoder/ms-marco-MiniLM-L6-v2
bm25_medcpt  -> ncbi/MedCPT-Cross-Encoder
bm25_minilm_l12 -> cross-encoder/ms-marco-MiniLM-L12-v2
```

Chạy thử một model bất kỳ:

```bash
uv run trec-evaluate run-experiment --config bm25_medcpt --limit-topics 1
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
```

## 7. Chạy full BM25 official

Không dùng `--limit-topics`:

```bash
uv run trec-evaluate run-experiment --config bm25_only
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
uv run trec-evaluate export-tables --run-dir runs/latest
```

## 8. Chạy từng config cần cho thí nghiệm

Khuyến nghị chạy từng config vào cùng một thư mục để có thể dừng/chạy tiếp nếu máy yếu. Các lệnh dưới đây dùng chung `RUN_DIR`, sau mỗi config sẽ chạy `eval-runs` để cập nhật `metrics.csv`.

```bash
RUN_DIR=runs/thesis_final
TREC_EVAL=tools/trec_eval/trec_eval

uv run trec-evaluate run-experiment --config bm25_only --run-dir $RUN_DIR
uv run trec-evaluate eval-runs --run-dir $RUN_DIR --trec-eval $TREC_EVAL

uv run trec-evaluate run-experiment --config bm25_expanded --run-dir $RUN_DIR
uv run trec-evaluate eval-runs --run-dir $RUN_DIR --trec-eval $TREC_EVAL

uv run trec-evaluate run-experiment --config bm25_minilm_l6 --run-dir $RUN_DIR
uv run trec-evaluate eval-runs --run-dir $RUN_DIR --trec-eval $TREC_EVAL

uv run trec-evaluate run-experiment --config bm25_medcpt --run-dir $RUN_DIR
uv run trec-evaluate eval-runs --run-dir $RUN_DIR --trec-eval $TREC_EVAL

uv run trec-evaluate run-experiment --config bm25_minilm_l12 --run-dir $RUN_DIR
uv run trec-evaluate eval-runs --run-dir $RUN_DIR --trec-eval $TREC_EVAL

uv run trec-evaluate run-experiment --config bm25_llm --run-dir $RUN_DIR
uv run trec-evaluate eval-runs --run-dir $RUN_DIR --trec-eval $TREC_EVAL

uv run trec-evaluate run-experiment --config bm25_minilm_l12_llm --run-dir $RUN_DIR
uv run trec-evaluate eval-runs --run-dir $RUN_DIR --trec-eval $TREC_EVAL
```

## 9. Tạo bảng kết quả cuối cùng

Sau khi đã chạy xong các config cần thiết, xuất bảng `.csv`, `.md`, `.tex`:

```bash
RUN_DIR=runs/thesis_final

uv run trec-evaluate export-tables --run-dir $RUN_DIR
```

Kết quả sẽ nằm ở:

```text
runs/thesis_final/tables/table_trec_eval.csv
runs/thesis_final/tables/table_trec_eval.md
runs/thesis_final/tables/table_trec_eval.tex
runs/thesis_final/tables/table_reranker_selection.csv
runs/thesis_final/tables/table_reranker_selection.md
runs/thesis_final/tables/table_reranker_selection.tex
runs/thesis_final/tables/table_ablation.csv
runs/thesis_final/tables/table_ablation.md
runs/thesis_final/tables/table_ablation.tex
```

## 10. Chạy full toàn bộ thí nghiệm bằng một lệnh

Lệnh này chạy tất cả config:

```bash
uv run trec-evaluate run-experiment --config all
uv run trec-evaluate eval-runs --run-dir runs/latest --trec-eval tools/trec_eval/trec_eval
uv run trec-evaluate export-tables --run-dir runs/latest
```

Các config trong `all`:

```text
bm25_only
bm25_expanded
bm25_minilm_l6
bm25_medcpt
bm25_minilm_l12
bm25_llm
bm25_minilm_l12_llm
```

## 11. Output cần xem

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
