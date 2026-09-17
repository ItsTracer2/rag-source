"""Tests des métriques d'évaluation.

Des métriques fausses sont pires que pas de métriques : elles donnent une
confiance injustifiée. Elles sont donc vérifiées sur des cas où le résultat
attendu se calcule à la main.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_source.domain import SectionKind
from rag_source.eval.retrieval import EvalCase, evaluate, load_cases
from rag_source.retrieval.search import Passage, SearchMode, SearchResult


class FakeService:
    """Renvoie les sources décidées à l'avance pour chaque question."""

    def __init__(self, answers: dict[str, list[str]]) -> None:
        self.answers = answers

    def search(self, query: str, **kwargs: object) -> SearchResult:
        sources = self.answers.get(query, [])
        passages = [
            Passage(
                id=f"id-{i}",
                text="…",
                source=source,
                heading_path=(),
                kind=SectionKind.PROSE,
                page_start=None,
                page_end=None,
                score=1.0,
                token_count=10,
                metadata={},
            )
            for i, source in enumerate(sources)
        ]
        return SearchResult(
            query=query, passages=passages, candidates=len(passages), mode=SearchMode.HYBRID
        )


def run(answers: dict[str, list[str]], cases: list[EvalCase]):  # type: ignore[no-untyped-def]
    return evaluate(FakeService(answers), cases, mode=SearchMode.HYBRID, top_k=6)  # type: ignore[arg-type]


CASE = EvalCase(id="c1", question="q1", expected_sources=("a.pdf",))
MULTI = EvalCase(id="c2", question="q2", expected_sources=("a.pdf", "b.pdf"))
OUT_OF_CORPUS = EvalCase(id="c3", question="q3", expected_sources=())


class TestMetrics:
    def test_perfect_hit(self) -> None:
        report = run({"q1": ["a.pdf"]}, [CASE])
        assert (report.hit_rate, report.recall, report.mrr) == (1.0, 1.0, 1.0)

    def test_miss(self) -> None:
        report = run({"q1": ["z.pdf"]}, [CASE])
        assert (report.hit_rate, report.recall, report.mrr) == (0.0, 0.0, 0.0)

    def test_reciprocal_rank_follows_position(self) -> None:
        assert run({"q1": ["z.pdf", "a.pdf"]}, [CASE]).mrr == pytest.approx(0.5)
        assert run({"q1": ["z.pdf", "y.pdf", "a.pdf"]}, [CASE]).mrr == pytest.approx(1 / 3)

    def test_partial_recall(self) -> None:
        report = run({"q2": ["a.pdf", "z.pdf"]}, [MULTI])
        assert report.recall == pytest.approx(0.5)
        assert report.hit_rate == 1.0  # une source trouvée suffit pour un « hit »

    def test_duplicate_sources_count_once(self) -> None:
        """Trois passages du même document ne valent pas trois sources trouvées."""
        report = run({"q2": ["a.pdf", "a.pdf", "a.pdf"]}, [MULTI])
        assert report.recall == pytest.approx(0.5)


class TestAbstention:
    def test_abstaining_on_out_of_corpus_is_correct(self) -> None:
        report = run({"q3": []}, [OUT_OF_CORPUS])
        assert report.abstention_rate == 1.0
        assert report.hit_rate == 0.0  # aucune question « répondable » dans ce lot

    def test_answering_out_of_corpus_is_counted(self) -> None:
        report = run({"q3": ["a.pdf"]}, [OUT_OF_CORPUS])
        assert report.abstention_rate == 0.0

    def test_false_abstention_is_tracked(self) -> None:
        """Se taire alors que la réponse existe est une erreur, et doit se voir."""
        report = run({"q1": []}, [CASE])
        assert report.false_abstention_rate == 1.0
        assert report.hit_rate == 0.0

    def test_out_of_corpus_cases_excluded_from_hit_rate(self) -> None:
        report = run({"q1": ["a.pdf"], "q3": []}, [CASE, OUT_OF_CORPUS])
        assert report.hit_rate == 1.0
        assert report.abstention_rate == 1.0


class TestFailures:
    def test_lists_misses_and_wrong_answers(self) -> None:
        report = run({"q1": ["z.pdf"], "q3": ["a.pdf"]}, [CASE, OUT_OF_CORPUS])
        assert {outcome.case.id for outcome in report.failures()} == {"c1", "c3"}

    def test_empty_when_everything_is_right(self) -> None:
        report = run({"q1": ["a.pdf"], "q3": []}, [CASE, OUT_OF_CORPUS])
        assert report.failures() == []


class TestDataset:
    def test_reads_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.jsonl"
        path.write_text(
            '{"id": "a", "question": "q ?", "expected_sources": ["x.pdf"], "tags": ["t"]}\n'
            "\n"
            '{"id": "b", "question": "q2 ?", "expected_sources": []}\n',
            encoding="utf-8",
        )
        cases = load_cases(path)
        assert [case.id for case in cases] == ["a", "b"]
        assert cases[0].tags == ("t",)
        assert cases[1].out_of_corpus

    def test_invalid_line_points_to_its_number(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.jsonl"
        path.write_text('{"id": "a", "question": "q ?"}\n{pas du json}\n', encoding="utf-8")
        with pytest.raises(ValueError, match=":2"):
            load_cases(path)


def test_reference_dataset_is_valid() -> None:
    """Le jeu de référence du dépôt doit rester lisible et cohérent."""
    cases = load_cases(Path("eval/datasets/corpus-reference.jsonl"))
    assert len(cases) >= 20
    assert len({case.id for case in cases}) == len(cases)
    assert any(case.out_of_corpus for case in cases)
    assert all(case.question.strip() for case in cases)
