"""Tests de l'API, avec des services simulés.

Aucun modèle n'est appelé : on vérifie le contrat HTTP — authentification, forme
des réponses, gestion des pannes, format du flux.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_source.api.app import app
from rag_source.api.dependencies import Services
from rag_source.config import Settings
from rag_source.domain import SectionKind
from rag_source.generation.answer import Answer
from rag_source.generation.prompts import Citation
from rag_source.retrieval.search import Passage, SearchMode, SearchResult
from rag_source.store.qdrant import StoreError

TOKEN = "jeton-de-test"

PASSAGE = Passage(
    id="id-1",
    text="Nettoyer le plateau avec un chiffon humide.",
    source="manuel.pdf",
    heading_path=("Entretien",),
    kind=SectionKind.PROSE,
    page_start=4,
    page_end=4,
    score=3.2,
    token_count=42,
    metadata={"rec_id": "R24"},
)


class FakeSearch:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def search(self, query: str, **kwargs: Any) -> SearchResult:
        if self.error:
            raise self.error
        return SearchResult(
            query=query, passages=[PASSAGE], candidates=12, mode=SearchMode.HYBRID_RERANK
        )


class FakeAnswer:
    def __init__(self, answer: Answer | None = None, error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def ask(self, question: str, **kwargs: Any) -> Answer:
        self.calls.append({"question": question, **kwargs})
        if self.error:
            raise self.error
        return self.answer or Answer(
            question=question,
            search_query=question,
            text="La réponse [1].",
            passages=[PASSAGE],
            citations=[Citation(number=1, passage=PASSAGE)],
        )

    def stream(self, question: str, **kwargs: Any) -> tuple[list[Passage], Iterator[str]]:
        self.calls.append({"question": question, **kwargs})
        return [PASSAGE], iter(["La ", "réponse ", "[1]."])


class FakeStore:
    def __init__(self, count: int = 7, error: Exception | None = None) -> None:
        self._count = count
        self._error = error

    def count(self) -> int:
        if self._error:
            raise self._error
        return self._count

    def sources(self) -> dict[str, int]:
        if self._error:
            raise self._error
        return {"manuel.pdf": 5, "guide.md": 2}


class FakeModelClient:
    def embed(self, texts: Any) -> list[list[float]]:
        return [[0.1]]

    def rerank(self, query: str, documents: Any) -> list[Any]:
        return []


def make_client(
    *,
    search: Any = None,
    answer: Any = None,
    store: Any = None,
    token: str | None = TOKEN,
) -> TestClient:
    settings = Settings(_env_file=None, api_token=token)
    services = Services(
        settings=settings,
        store=store or FakeStore(),  # type: ignore[arg-type]
        embedder=FakeModelClient(),  # type: ignore[arg-type]
        reranker=FakeModelClient(),  # type: ignore[arg-type]
        llm=None,  # type: ignore[arg-type]
        search=search or FakeSearch(),  # type: ignore[arg-type]
        answer=answer or FakeAnswer(),  # type: ignore[arg-type]
    )
    client = TestClient(app)
    client.app.state.services = services  # type: ignore[attr-defined]
    client.headers.update({"Authorization": f"Bearer {TOKEN}"})
    return client


class TestAuthentication:
    def test_missing_token_is_refused(self) -> None:
        client = make_client()
        client.headers.pop("Authorization")
        assert client.get("/health").status_code == 401

    def test_wrong_token_is_refused(self) -> None:
        client = make_client()
        client.headers["Authorization"] = "Bearer mauvais-jeton"
        assert client.get("/health").status_code == 401

    def test_malformed_header_is_refused(self) -> None:
        client = make_client()
        client.headers["Authorization"] = TOKEN  # sans le préfixe « Bearer »
        assert client.get("/health").status_code == 401

    def test_unconfigured_token_blocks_the_api(self) -> None:
        """Mieux vaut une API inutilisable qu'une API ouverte à tous."""
        client = make_client(token=None)
        assert client.get("/health").status_code == 503


class TestHealth:
    def test_reports_services_and_sovereignty(self) -> None:
        body = make_client().get("/health").json()
        assert body["status"] == "ok"
        assert body["sovereignty"] == "local"
        assert body["indexed_chunks"] == 7
        assert {service["name"] for service in body["services"]} == {"qdrant", "embed", "rerank"}

    def test_degraded_when_a_service_fails(self) -> None:
        body = make_client(store=FakeStore(error=StoreError("hors service"))).get("/health").json()
        assert body["status"] == "degraded"
        assert body["indexed_chunks"] is None
        qdrant = next(s for s in body["services"] if s["name"] == "qdrant")
        assert qdrant["reachable"] is False and "hors service" in qdrant["detail"]


class TestDocuments:
    def test_lists_indexed_documents(self) -> None:
        body = make_client().get("/v1/documents").json()
        assert body["total_chunks"] == 7
        assert body["documents"] == [
            {"source": "manuel.pdf", "chunks": 5},
            {"source": "guide.md", "chunks": 2},
        ]

    def test_store_failure_returns_503(self) -> None:
        client = make_client(store=FakeStore(error=StoreError("base injoignable")))
        assert client.get("/v1/documents").status_code == 503


class TestSearch:
    def test_returns_passages(self) -> None:
        body = make_client().post("/v1/search", json={"query": "plateau"}).json()
        assert body["candidates"] == 12
        passage = body["passages"][0]
        assert passage["source"] == "manuel.pdf"
        assert passage["location"] == "manuel.pdf, p. 4 — Entretien"

    def test_empty_query_is_rejected(self) -> None:
        assert make_client().post("/v1/search", json={"query": ""}).status_code == 422

    def test_unknown_mode_is_rejected(self) -> None:
        response = make_client().post("/v1/search", json={"query": "q", "mode": "magique"})
        assert response.status_code == 422

    def test_store_failure_returns_503(self) -> None:
        client = make_client(search=FakeSearch(error=StoreError("base injoignable")))
        response = client.post("/v1/search", json={"query": "q"})
        assert response.status_code == 503
        assert "injoignable" in response.json()["detail"]


class TestAsk:
    def test_answer_with_citations(self) -> None:
        body = make_client().post("/v1/ask", json={"question": "Comment nettoyer ?"}).json()
        assert body["answer"] == "La réponse [1]."
        assert body["grounded"] is True and body["refused"] is False
        assert body["citations"] == [
            {"number": 1, "source": "manuel.pdf", "location": "manuel.pdf, p. 4 — Entretien"}
        ]

    def test_refusal_is_explicit(self) -> None:
        refused = Answer(
            question="q", search_query="q", text="Je ne trouve pas…", passages=[], citations=[]
        )
        body = (
            make_client(answer=FakeAnswer(refused)).post("/v1/ask", json={"question": "q"}).json()
        )
        assert body["refused"] is True and body["grounded"] is False and body["passages"] == []

    def test_invalid_citations_are_exposed(self) -> None:
        answer = Answer(
            question="q",
            search_query="q",
            text="Affirmation [9].",
            passages=[PASSAGE],
            citations=[],
            invalid_citations=[9],
        )
        body = make_client(answer=FakeAnswer(answer)).post("/v1/ask", json={"question": "q"}).json()
        assert body["invalid_citations"] == [9]
        assert body["grounded"] is False

    def test_history_is_forwarded(self) -> None:
        fake = FakeAnswer()
        client = make_client(answer=fake)
        client.post(
            "/v1/ask",
            json={
                "question": "Et ensuite ?",
                "history": [{"question": "Première ?", "answer": "A."}],
            },
        )
        assert fake.calls[0]["history"] == [("Première ?", "A.")]

    def test_filters_are_forwarded(self) -> None:
        fake = FakeAnswer()
        client = make_client(answer=fake)
        client.post("/v1/ask", json={"question": "q", "source": "manuel.pdf", "top_k": 3})
        assert fake.calls[0]["source"] == "manuel.pdf"
        assert fake.calls[0]["top_k"] == 3

    @pytest.mark.parametrize("payload", [{}, {"question": "x" * 3000}, {"question": None}])
    def test_invalid_payloads_are_rejected(self, payload: dict[str, Any]) -> None:
        assert make_client().post("/v1/ask", json=payload).status_code == 422


class TestStream:
    def test_events_are_ordered(self) -> None:
        with make_client().stream(
            "POST", "/v1/ask/stream", json={"question": "Comment nettoyer ?"}
        ) as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            body = "".join(response.iter_text())

        events = [line[len("event: ") :] for line in body.splitlines() if line.startswith("event:")]
        assert events[0] == "passages"
        assert events[-1] == "done"
        assert events.count("token") == 3

    def test_passages_arrive_before_the_answer(self) -> None:
        """L'interface doit pouvoir afficher les sources pendant la génération."""
        with make_client().stream("POST", "/v1/ask/stream", json={"question": "q"}) as response:
            body = "".join(response.iter_text())
        first_payload = json.loads(body.split("data: ", 1)[1].split("\n", 1)[0])
        assert first_payload["passages"][0]["source"] == "manuel.pdf"

    def test_done_event_carries_citations(self) -> None:
        with make_client().stream("POST", "/v1/ask/stream", json={"question": "q"}) as response:
            body = "".join(response.iter_text())
        done = json.loads(body.rstrip().rsplit("data: ", 1)[1])
        assert [citation["number"] for citation in done["citations"]] == [1]
        assert done["invalid_citations"] == []
