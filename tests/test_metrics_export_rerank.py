from pathlib import Path

from trec_evaluate.export import export_tables
from trec_evaluate.metrics import parse_trec_eval_output, write_csv
from trec_evaluate.query_expansion import QueryExpander
from trec_evaluate.rerank import Candidate


def test_parse_trec_eval_output(tmp_path: Path):
    path = tmp_path / "eval.txt"
    path.write_text(
        "ndcg_cut_10 all 0.1234\nP_10 all 0.5000\nrecip_rank all 1.0000\n",
        encoding="utf-8",
    )
    metrics = parse_trec_eval_output(path)
    assert metrics["ndcg_cut_10"] == 0.1234
    assert metrics["P_10"] == 0.5
    assert metrics["recip_rank"] == 1.0


def test_export_tables_with_pending_rows(tmp_path: Path):
    run_dir = tmp_path / "run"
    write_csv(
        run_dir / "metrics.csv",
        [
            {
                "config": "bm25_only",
                "nDCG@10": "0.1000",
                "P@10": "0.2000",
                "RR": "0.3000",
                "nDCG@1000": "0.4000",
                "Recall@100": "0.5000",
            }
        ],
    )
    (run_dir / "latency.csv").write_text("config,topic,stage,candidates,seconds\nbm25_only,1,total,1000,1.0\n")
    outputs = export_tables(run_dir)
    assert len(outputs) == 9
    assert (run_dir / "tables" / "table_ablation.md").exists()


class FakeReranker:
    def score(self, query: str, candidates: list[Candidate]) -> list[float]:
        return [10.0 if c.doc_id == "B" else 1.0 for c in candidates]


def test_rerank_preserves_tail():
    from trec_evaluate.rerank import rerank_candidates

    candidates = [
        Candidate("A", 3.0, "a"),
        Candidate("B", 2.0, "b"),
        Candidate("C", 1.0, "c"),
    ]
    ranked = rerank_candidates("query", candidates, FakeReranker(), window=2)
    assert [c.doc_id for c in ranked] == ["B", "A", "C"]


def test_query_expansion_builds_bm25_query_without_inferred_facts():
    expander = QueryExpander(max_terms_per_category=2, use_related_concepts=False)
    terms = expander._terms_from_response(
        {
            "diseases": ["non-small cell lung cancer", "NSCLC", "extra ignored"],
            "biomarkers": ["EGFR"],
            "related_concepts": ["lung neoplasm"],
        }
    )
    query = expander._build_query("disease: lung cancer", terms)
    assert "disease: lung cancer" in query
    assert "non-small cell lung cancer NSCLC" in query
    assert "EGFR" in query
    assert "lung neoplasm" not in query
