"""Tests de la chaîne de recherche, avec index, embedder et reranker simulés.

Ce qui est vérifié ici n'est pas la qualité des modèles (c'est le rôle du banc
d'évaluation) mais la logique autour : quels passages sont retenus, dans quel
ordre, et surtout quand le système doit refuser de répondre.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from rag_source.clients.reranker import RerankScore
from rag_source.domain import SectionKind
from rag_source.retrieval.search import Passage, SearchConfig, SearchMode, SearchService


class FakeEmbedder:
    dimension = 3

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeStore:
    """Renvoie des points Qdrant simulés, et retient la requête reçue."""

    def __init__(self, points: list[dict[str, Any]]) -> None:
        self.points = points
        self.calls: list[dict[str, Any]] = []

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return self.points


class FakeReranker:
    def __init__(self, scores: dict[int, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, int]] = []

    def rerank(self, query: str, documents: Sequence[str]) -> list[RerankScore]:
        self.calls.append((query, len(documents)))
        ranked = [
            RerankScore(index=i, score=self.scores.get(i, 0.0)) for i in range(len(documents))
        ]
        return sorted(ranked, key=lambda item: item.score, reverse=True)


def point(
    index: int, source: str = "manuel.pdf", tokens: int = 100, **payload: Any
) -> dict[str, Any]:
    base = {
        "text": f"Passage numéro {index}.",
        "source": source,
        "heading_path": ["Chapitre", f"Section {index}"],
        "kind": "prose",
        "page_start": index,
        "page_end": index,
        "token_count": tokens,
        "metadata": {},
    }
    base.update(payload)
    return {"id": f"id-{index}", "score": 1.0 / (index + 1), "payload": base}


def service(points: list[dict[str, Any]], scores: dict[int, float] | None = None, **config: Any):  # type: ignore[no-untyped-def]
    store = FakeStore(points)
    reranker = FakeReranker(scores) if scores is not None else None
    return (
        SearchService(store, FakeEmbedder(), reranker, config=SearchConfig(**config)),  # type: ignore[arg-type]
        store,
        reranker,
    )


class TestModes:
    def test_hybrid_sends_both_vectors(self) -> None:
        svc, store, _ = service([point(0)])
        svc.search("question", mode=SearchMode.HYBRID)
        call = store.calls[0]
        assert call["dense"] is not None and call["sparse"] is not None

    def test_dense_only(self) -> None:
        svc, store, _ = service([point(0)])
        svc.search("question", mode=SearchMode.DENSE)
        assert store.calls[0]["dense"] is not None
        assert store.calls[0]["sparse"] is None

    def test_sparse_only_skips_embedding(self) -> None:
        svc, store, _ = service([point(0)])
        svc.search("question", mode=SearchMode.SPARSE)
        assert store.calls[0]["dense"] is None
        assert store.calls[0]["sparse"] is not None

    def test_filters_are_forwarded(self) -> None:
        svc, store, _ = service([point(0)])
        svc.search("question", source="manuel.pdf", kind="record")
        assert store.calls[0]["source"] == "manuel.pdf"
        assert store.calls[0]["kind"] == "record"


class TestReranking:
    def test_reorders_passages(self) -> None:
        """Test d'ordre pur : le seuil est mis hors jeu pour ne mesurer qu'une chose."""
        points = [point(0), point(1), point(2)]
        svc, _, reranker = service(points, scores={0: -1.0, 1: 5.0, 2: 2.0}, min_score=-100.0)
        result = svc.search("question", mode=SearchMode.HYBRID_RERANK)
        assert [p.id for p in result.passages] == ["id-1", "id-2", "id-0"]
        assert reranker is not None and reranker.calls == [("question", 3)]

    def test_hybrid_without_rerank_keeps_fusion_order(self) -> None:
        points = [point(0), point(1), point(2)]
        svc, _, _ = service(points, scores={0: -1.0, 1: 5.0, 2: 2.0})
        result = svc.search("question", mode=SearchMode.HYBRID)
        assert [p.id for p in result.passages] == ["id-0", "id-1", "id-2"]


class TestThreshold:
    def test_abstains_when_nothing_is_relevant(self) -> None:
        """Question hors corpus : mieux vaut ne rien rendre que de laisser le LLM broder."""
        points = [point(0), point(1)]
        svc, _, _ = service(points, scores={0: -8.0, 1: -11.0})
        result = svc.search("question hors sujet")
        assert result.passages == []
        assert not result.found

    def test_keeps_only_passages_above_the_threshold(self) -> None:
        points = [point(0), point(1), point(2)]
        svc, _, _ = service(points, scores={0: 4.0, 1: 1.0, 2: -6.0}, min_score=0.0)
        result = svc.search("question")
        assert [p.id for p in result.passages] == ["id-0", "id-1"]

    def test_threshold_is_configurable(self) -> None:
        points = [point(0), point(1)]
        svc, _, _ = service(points, scores={0: 4.0, 1: 1.0}, min_score=2.0)
        assert [p.id for p in svc.search("question").passages] == ["id-0"]


class TestBudget:
    def test_stops_at_the_token_budget(self) -> None:
        """Deux passages de 400 tokens tiennent dans 1000, le troisième non."""
        points = [point(i, tokens=400) for i in range(6)]
        svc, _, _ = service(points, scores=dict.fromkeys(range(6), 5.0), context_tokens=1000)
        assert len(svc.search("question").passages) == 2

    def test_stops_at_top_k(self) -> None:
        points = [point(i, tokens=10) for i in range(10)]
        svc, _, _ = service(points, scores=dict.fromkeys(range(10), 5.0), top_k=4)
        assert len(svc.search("question").passages) == 4

    def test_first_passage_is_kept_even_if_huge(self) -> None:
        """Un passage unique plus gros que le budget vaut mieux que rien du tout."""
        svc, _, _ = service([point(0, tokens=5000)], scores={0: 5.0}, context_tokens=1000)
        assert len(svc.search("question").passages) == 1


class TestPassage:
    def test_maps_payload_fields(self) -> None:
        svc, _, _ = service([point(3, source="dossier/guide.pdf", metadata={"rec_id": "R24"})])
        passage = svc.search("question", mode=SearchMode.HYBRID).passages[0]
        assert passage.source == "dossier/guide.pdf"
        assert passage.heading_path == ("Chapitre", "Section 3")
        assert passage.kind is SectionKind.PROSE
        assert passage.metadata["rec_id"] == "R24"

    @pytest.mark.parametrize(
        ("page_start", "page_end", "expected"),
        [
            (3, 3, "guide.pdf, p. 3 — Entretien"),
            (3, 4, "guide.pdf, p. 3-4 — Entretien"),
            (None, None, "guide.pdf — Entretien"),
        ],
    )
    def test_location_is_readable(
        self, page_start: int | None, page_end: int | None, expected: str
    ) -> None:
        passage = Passage(
            id="x",
            text="…",
            source="guide.pdf",
            heading_path=("Entretien",),
            kind=SectionKind.PROSE,
            page_start=page_start,
            page_end=page_end,
            score=1.0,
            token_count=10,
            metadata={},
        )
        assert passage.location == expected


def test_empty_index_returns_nothing() -> None:
    svc, _, _ = service([], scores={})
    result = svc.search("question")
    assert not result.found and result.candidates == 0
