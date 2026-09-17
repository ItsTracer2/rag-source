"""Vérifie que la pile Docker rend bien les services attendus.

    docker compose up -d
    uv run pytest -m integration

Ces tests ne remplacent pas les tests unitaires : ils contrôlent le contrat entre
RAG-Source et les services externes (llama.cpp, Qdrant), c'est-à-dire exactement ce
qu'un test unitaire ne peut pas voir — une option de ligne de commande oubliée, un
modèle mal chargé, une version d'API qui change.
"""

from __future__ import annotations

import math
import os
from typing import Any

import httpx
import pytest

from rag_source.config import Settings

pytestmark = pytest.mark.integration

_settings = Settings()
QDRANT_HEADERS = (
    {"api-key": _settings.qdrant_api_key.get_secret_value()}
    if _settings.qdrant_api_key is not None
    else {}
)

EMBED_URL = os.getenv("RAG_SOURCE_TEST_EMBED_URL", "http://127.0.0.1:8082")
RERANK_URL = os.getenv("RAG_SOURCE_TEST_RERANK_URL", "http://127.0.0.1:8083")
LLM_URL = os.getenv("RAG_SOURCE_TEST_LLM_URL", "http://127.0.0.1:8081")
QDRANT_URL = os.getenv("RAG_SOURCE_TEST_QDRANT_URL", "http://127.0.0.1:6333")

BGE_M3_DIMENSION = 1024


def post(url: str, payload: dict[str, object], timeout: float = 300.0) -> dict[str, Any]:
    try:
        response = httpx.post(url, json=payload, timeout=timeout)
    except httpx.ConnectError:  # pragma: no cover - dépend de l'environnement
        pytest.skip(f"Service injoignable ({url}) : lancer `docker compose up -d`.")
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    return data


def cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm


class TestEmbedding:
    def test_dimension_matches_bge_m3(self) -> None:
        data = post(f"{EMBED_URL}/v1/embeddings", {"input": ["Un texte de test."]})
        vector = data["data"][0]["embedding"]
        assert len(vector) == BGE_M3_DIMENSION

    def test_is_multilingual(self) -> None:
        """Deux formulations du même besoin, en deux langues, doivent se ressembler.

        C'est la propriété qui permet d'interroger en français un manuel rédigé en
        anglais — et la raison du choix de bge-m3.
        """
        data = post(
            f"{EMBED_URL}/v1/embeddings",
            {
                "input": [
                    "Comment nettoyer le plateau tournant ?",
                    "How do I clean the turntable?",
                    "La garantie couvre deux ans de pièces et main d'œuvre.",
                ]
            },
        )
        fr, en, unrelated = (item["embedding"] for item in data["data"])
        assert cosine(fr, en) > 0.6
        assert cosine(fr, en) > cosine(fr, unrelated)


class TestReranking:
    def test_ranks_the_relevant_passage_first(self) -> None:
        data = post(
            f"{RERANK_URL}/v1/rerank",
            {
                "query": "comment nettoyer le plateau ?",
                "documents": [
                    "La puissance maximale de l'appareil est de 900 watts.",
                    "Nettoyer le plateau tournant avec un chiffon humide et du savon doux.",
                    "Le minuteur se règle à l'aide de la molette de droite.",
                ],
            },
        )
        results = sorted(data["results"], key=lambda r: -r["relevance_score"])
        assert results[0]["index"] == 1
        # L'écart doit être franc : c'est ce qui permettra de fixer un seuil de
        # pertinence et de répondre « je ne sais pas » plutôt que d'inventer.
        assert results[0]["relevance_score"] - results[1]["relevance_score"] > 3


class TestLlm:
    def test_answers_in_the_language_of_the_question(self) -> None:
        data = post(
            f"{LLM_URL}/v1/chat/completions",
            {
                "model": "local",
                "messages": [{"role": "user", "content": "Réponds « bonjour » et rien d'autre."}],
                "max_tokens": 20,
                "temperature": 0,
            },
        )
        content = data["choices"][0]["message"]["content"].strip().lower()
        assert "bonjour" in content

    def test_context_window_is_configured(self) -> None:
        try:
            response = httpx.get(f"{LLM_URL}/props", timeout=30)
        except httpx.ConnectError:  # pragma: no cover
            pytest.skip("LLM injoignable.")
        context = response.json()["default_generation_settings"]["n_ctx"]
        # 4096 suffit : 2500 tokens de passages, la question, et la réponse.
        assert context >= 4096


class TestQdrant:
    def test_is_reachable_and_versioned(self) -> None:
        try:
            response = httpx.get(f"{QDRANT_URL}/", timeout=10, headers=QDRANT_HEADERS)
        except httpx.ConnectError:  # pragma: no cover
            pytest.skip("Qdrant injoignable.")
        assert response.json()["title"].startswith("qdrant")

    def test_accepts_collection_lifecycle(self) -> None:
        """Création, écriture, recherche et suppression : le cycle complet."""
        name = "rag_source_integration_check"
        base = f"{QDRANT_URL}/collections/{name}"
        httpx.delete(base, timeout=30, headers=QDRANT_HEADERS)
        httpx.put(
            base,
            json={"vectors": {"size": 4, "distance": "Cosine"}},
            timeout=30,
            headers=QDRANT_HEADERS,
        ).raise_for_status()
        try:
            httpx.put(
                f"{base}/points?wait=true",
                json={
                    "points": [
                        {"id": 1, "vector": [1.0, 0.0, 0.0, 0.0], "payload": {"k": "a"}},
                        {"id": 2, "vector": [0.0, 1.0, 0.0, 0.0], "payload": {"k": "b"}},
                    ]
                },
                timeout=30,
                headers=QDRANT_HEADERS,
            ).raise_for_status()
            found = httpx.post(
                f"{base}/points/search",
                json={"vector": [0.9, 0.1, 0.0, 0.0], "limit": 1, "with_payload": True},
                timeout=30,
                headers=QDRANT_HEADERS,
            ).json()["result"]
            assert found[0]["payload"]["k"] == "a"
        finally:
            httpx.delete(base, timeout=30, headers=QDRANT_HEADERS)
