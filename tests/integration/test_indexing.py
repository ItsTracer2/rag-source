"""Indexation de bout en bout : vrai service d'embedding, vraie base Qdrant.

    docker compose up -d
    uv run pytest -m integration

Une collection dédiée est créée puis détruite : la collection de travail n'est
jamais touchée.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from rag_source.clients.embedder import LlamaCppEmbedder
from rag_source.config import Settings
from rag_source.ingest.indexer import index_corpus
from rag_source.ingest.tokenizer import EstimatedTokenCounter
from rag_source.store.qdrant import DENSE, SPARSE, QdrantStore
from rag_source.store.sparse import encode_query

pytestmark = pytest.mark.integration

COLLECTION = "rag_source_integration_index"
MANUAL = """# Micro-ondes MW-900

## Nettoyage

Nettoyer le plateau tournant avec un chiffon humide et un savon doux.
Ne jamais utiliser de produit abrasif sur la paroi intérieure.

## Codes d'erreur

Le code E01 signale une porte restée ouverte pendant la cuisson.
"""


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="module")
def store(settings: Settings) -> Iterator[QdrantStore]:
    api_key = (
        settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key is not None else None
    )
    try:
        httpx.get(settings.qdrant_url, timeout=5, headers={"api-key": api_key or ""})
    except httpx.ConnectError:  # pragma: no cover - dépend de l'environnement
        pytest.skip("Qdrant injoignable : lancer `docker compose up -d`.")
    with QdrantStore(settings.qdrant_url, COLLECTION, api_key=api_key) as opened:
        opened.drop()
        yield opened
        opened.drop()


@pytest.fixture(scope="module")
def embedder(settings: Settings) -> Iterator[LlamaCppEmbedder]:
    with LlamaCppEmbedder(settings.embed_url) as opened:
        try:
            opened.embed(["test"])
        except Exception:  # pragma: no cover - dépend de l'environnement
            pytest.skip("Service d'embedding injoignable.")
        yield opened


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "manuel.md").write_text(MANUAL, encoding="utf-8")
    return tmp_path


def test_full_lifecycle(
    corpus: Path, store: QdrantStore, embedder: LlamaCppEmbedder, settings: Settings
) -> None:
    counter = EstimatedTokenCounter()

    # 1. Première indexation.
    report = index_corpus(corpus, store, embedder, counter, force=True)
    assert report.added == ["manuel.md"]
    assert store.count() == report.chunks_written > 0
    assert store.dimension() == embedder.dimension

    # 2. Rien n'a changé : aucun chunk réécrit.
    again = index_corpus(corpus, store, embedder, counter)
    assert again.unchanged == ["manuel.md"]
    assert again.chunks_written == 0
    assert store.count() == report.chunks_written

    # 3. Le document change : ses chunks sont remplacés, pas ajoutés.
    (corpus / "manuel.md").write_text(MANUAL + "\nLa garantie dure deux ans.\n", encoding="utf-8")
    updated = index_corpus(corpus, store, embedder, counter)
    assert updated.updated == ["manuel.md"]
    assert store.count() == updated.chunks_written

    # 4. Le document disparaît : l'index se vide.
    (corpus / "manuel.md").unlink()
    removed = index_corpus(corpus, store, embedder, counter)
    assert removed.removed == ["manuel.md"]
    assert store.count() == 0


def test_both_vectors_are_searchable(
    corpus: Path, store: QdrantStore, embedder: LlamaCppEmbedder, settings: Settings
) -> None:
    """Les deux représentations doivent retrouver le bon passage, chacune à sa façon.

    Le vecteur dense répond à une question reformulée ; le vecteur creux retrouve un
    identifiant exact, ce que la recherche sémantique seule fait mal.
    """
    index_corpus(corpus, store, embedder, EstimatedTokenCounter(), force=True)
    api_key = (
        settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key is not None else None
    )
    client = httpx.Client(
        base_url=settings.qdrant_url,
        timeout=60,
        headers={"api-key": api_key} if api_key else {},
    )

    question = "comment entretenir le plateau ?"
    dense_hit = client.post(
        f"/collections/{COLLECTION}/points/query",
        json={
            "query": embedder.embed([question])[0],
            "using": DENSE,
            "limit": 1,
            "with_payload": True,
        },
    ).json()["result"]["points"][0]
    assert "plateau tournant" in dense_hit["payload"]["text"]

    sparse_query = encode_query("E01")
    sparse_hit = client.post(
        f"/collections/{COLLECTION}/points/query",
        json={
            "query": {"indices": sparse_query.indices, "values": sparse_query.values},
            "using": SPARSE,
            "limit": 1,
            "with_payload": True,
        },
    ).json()["result"]["points"][0]
    assert "E01" in sparse_hit["payload"]["text"]
    client.close()
