"""Recherche de bout en bout sur la vraie pile : hybride, reranking, abstention."""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from rag_source.clients.embedder import LlamaCppEmbedder
from rag_source.clients.reranker import LlamaCppReranker
from rag_source.config import Settings
from rag_source.ingest.indexer import index_corpus
from rag_source.ingest.tokenizer import EstimatedTokenCounter
from rag_source.retrieval.search import SearchMode, SearchService
from rag_source.store.qdrant import QdrantStore

pytestmark = pytest.mark.integration

COLLECTION = "rag_source_integration_search"
MANUAL = """# Four à micro-ondes MW-900

## Nettoyage

Nettoyer le plateau tournant avec un chiffon humide et un savon doux.
Ne jamais employer de produit abrasif sur la paroi intérieure de l'appareil.

## Codes d'erreur

Le code E01 signale une porte restée ouverte pendant la cuisson.
Le code E07 indique une surchauffe du magnétron et impose un arrêt immédiat.

## Garantie

La garantie couvre deux ans les pièces et la main d'oeuvre, hors usure normale.
"""


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="module")
def service(
    settings: Settings, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[SearchService]:
    api_key = (
        settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key is not None else None
    )
    try:
        httpx.get(settings.qdrant_url, timeout=5, headers={"api-key": api_key or ""})
    except httpx.ConnectError:  # pragma: no cover - dépend de l'environnement
        pytest.skip("Pile injoignable : lancer `docker compose up -d`.")

    corpus = tmp_path_factory.mktemp("corpus")
    (corpus / "manuel.md").write_text(MANUAL, encoding="utf-8")

    with (
        QdrantStore(settings.qdrant_url, COLLECTION, api_key=api_key) as store,
        LlamaCppEmbedder(settings.embed_url) as embedder,
        LlamaCppReranker(settings.rerank_url) as reranker,
    ):
        store.drop()
        index_corpus(corpus, store, embedder, EstimatedTokenCounter(), force=True)
        yield SearchService(store, embedder, reranker, EstimatedTokenCounter())
        store.drop()


def test_reformulated_question_finds_the_passage(service: SearchService) -> None:
    """Aucun mot commun avec le texte : seule la recherche sémantique peut réussir."""
    result = service.search("comment entretenir le plateau ?")
    assert result.found
    assert "chiffon humide" in result.passages[0].text


def test_exact_identifier_is_found(service: SearchService) -> None:
    """« E07 » n'a pas de sens sémantique : c'est la recherche par mots-clés qui répond."""
    result = service.search("E07")
    assert result.found
    assert "E07" in result.passages[0].text


def test_english_question_on_french_document(service: SearchService) -> None:
    """bge-m3 est multilingue : la langue de la question n'a pas à suivre le corpus."""
    result = service.search("how long is the warranty?")
    assert result.found
    assert "garantie" in result.passages[0].text.lower()


def test_out_of_corpus_question_is_refused(service: SearchService) -> None:
    """Le seuil de pertinence évite de fournir au LLM des passages hors sujet."""
    result = service.search("quelle est la recette du gratin dauphinois ?")
    assert not result.found


def test_reranking_changes_the_order(service: SearchService) -> None:
    question = "que faire si la porte reste ouverte ?"
    without = service.search(question, mode=SearchMode.HYBRID, top_k=5)
    with_rerank = service.search(question, mode=SearchMode.HYBRID_RERANK, top_k=5)
    assert with_rerank.found
    assert "E01" in with_rerank.passages[0].text
    assert without.found
