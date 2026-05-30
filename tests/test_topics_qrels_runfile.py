from pathlib import Path

from trec_evaluate.qrels import parse_qrels, recall_at_k
from trec_evaluate.runfile import doc_ids_by_topic, parse_run, write_run
from trec_evaluate.topics import parse_topics


def test_parse_topics(tmp_path: Path):
    path = tmp_path / "topics.xml"
    path.write_text(
        """<topics task="clinical">
        <topic number="1" template="x">
          <field name="disease">lung cancer</field>
          <field name="age">65</field>
        </topic>
        </topics>""",
        encoding="utf-8",
    )
    topics = parse_topics(path)
    assert len(topics) == 1
    assert topics[0].number == "1"
    assert "disease: lung cancer" in topics[0].to_query_text()


def test_qrels_recall_at_100(tmp_path: Path):
    path = tmp_path / "qrels.txt"
    path.write_text("1 0 A 2\n1 0 B 1\n1 0 C 0\n2 0 D 2\n", encoding="utf-8")
    qrels = parse_qrels(path)
    recall = recall_at_k({"1": ["A", "C"], "2": ["X"]}, qrels, k=100)
    assert recall == 0.25


def test_runfile_round_trip_and_doc_ids(tmp_path: Path):
    run = tmp_path / "test.run"
    write_run(run, {"1": [("A", 3.0), ("B", 2.0)]}, "unit")
    parsed = parse_run(run)
    assert parsed["1"][0].doc_id == "A"
    assert doc_ids_by_topic(run) == {"1": ["A", "B"]}

