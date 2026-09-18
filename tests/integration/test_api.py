"""API interrogée sur la pile réelle : modèles, index et génération.

docker compose up -d && uv run python -m rag_source.ingest data
uv run pytest -m integration
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from rag_source.api.app import app
from rag_source.api.dependencies import build_services
from rag_source.config import Settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    settings = Settings()
    if settings.api_token is None:  # pragma: no cover - dépend de l'environnement
        pytest.skip("RAG_SOURCE_API_TOKEN absent : lancer ./scripts/init-env.sh")
    try:
        httpx.get(settings.qdrant_url, timeout=5)
    except httpx.ConnectError:  # pragma: no cover
        pytest.skip("Pile injoignable : lancer `docker compose up -d`.")

    services = build_services(settings)
    test_client = TestClient(app)
    test_client.app.state.services = services  # type: ignore[attr-defined]
    test_client.headers.update({"Authorization": f"Bearer {settings.api_token.get_secret_value()}"})
    try:
        yield test_client
    finally:
        services.close()


def test_health_reports_a_populated_index(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok", body
    assert body["sovereignty"] in {"local", "external"}
    assert (body["indexed_chunks"] or 0) > 0, "corpus non indexé : python -m rag_source.ingest data"


def test_search_returns_located_passages(client: TestClient) -> None:
    body = client.post("/v1/search", json={"query": "qualification prestataire", "top_k": 3}).json()
    assert body["passages"], body
    assert all(passage["location"] for passage in body["passages"])


def test_ask_answers_with_verified_citations(client: TestClient) -> None:
    body = client.post(
        "/v1/ask",
        json={"question": "Combien de temps dure la qualification d'un prestataire ?"},
    ).json()
    assert body["refused"] is False
    assert body["invalid_citations"] == [], "le modèle a cité un passage inexistant"
    assert body["grounded"] is True, body["answer"]
    cited = {citation["number"] for citation in body["citations"]}
    assert cited <= set(range(1, len(body["passages"]) + 1))


def test_out_of_corpus_question_is_refused(client: TestClient) -> None:
    """Bout en bout : rien de pertinent, donc pas d'appel au modèle et pas d'invention."""
    body = client.post("/v1/ask", json={"question": "Quelle est la recette du tiramisu ?"}).json()
    assert body["refused"] is True
    assert body["passages"] == []


def test_stream_delivers_passages_then_tokens(client: TestClient) -> None:
    with client.stream(
        "POST", "/v1/ask/stream", json={"question": "Que dit la recommandation R24 ?", "top_k": 3}
    ) as response:
        body = "".join(response.iter_text())
    events = [line[len("event: ") :] for line in body.splitlines() if line.startswith("event:")]
    assert events[0] == "passages"
    assert "token" in events
    assert events[-1] == "done"
