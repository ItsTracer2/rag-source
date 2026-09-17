"""Client de reranking (llama-server ``--reranking``).

Le reranker est un *cross-encoder* : il lit la question et le passage ensemble,
là où l'embedding les a vectorisés séparément. Il est bien plus juste, mais aussi
bien plus lent — d'où l'ordre de la chaîne : la recherche vectorielle ramène
quelques dizaines de candidats, le reranker n'en juge que ceux-là.

L'implémentation d'origine utilisait ``ms-marco-MiniLM-L-6-v2``, entraîné
uniquement sur de l'anglais, pour reclasser un corpus français : il réordonnait
presque au hasard. Ici, ``bge-reranker-v2-m3`` est multilingue, de la même famille
que le modèle d'embedding.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 300.0


class RerankError(RuntimeError):
    """Le service de reranking est injoignable ou a renvoyé une réponse inattendue."""


@dataclass(frozen=True, slots=True)
class RerankScore:
    index: int
    """Position du document dans la liste envoyée."""
    score: float


class Reranker(Protocol):
    def rerank(self, query: str, documents: Sequence[str]) -> list[RerankScore]: ...


class LlamaCppReranker:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/v1/rerank"
        self._client = client or httpx.Client(timeout=timeout)

    def rerank(self, query: str, documents: Sequence[str]) -> list[RerankScore]:
        """Scores de pertinence, du plus pertinent au moins pertinent.

        Les scores ne sont ni bornés ni centrés sur zéro, et leur échelle dépend du
        modèle : sur bge-reranker-v2-m3, un passage qui répond se situe entre -2,5 et
        0, un passage hors sujet autour de -11. Seul l'écart compte, et c'est lui qui
        permet plus loin de refuser une réponse plutôt que de la fabriquer à partir
        de passages inadaptés.
        """
        if not documents:
            return []
        try:
            response = self._client.post(
                self._url, json={"query": query, "documents": list(documents)}
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise RerankError(f"Service de reranking injoignable ({self._url}) : {exc}") from exc

        try:
            scores = [
                RerankScore(index=int(item["index"]), score=float(item["relevance_score"]))
                for item in payload["results"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankError(f"Réponse de reranking inexploitable : {exc}") from exc

        if len(scores) != len(documents):
            raise RerankError(
                f"{len(scores)} scores reçus pour {len(documents)} documents envoyés."
            )
        return sorted(scores, key=lambda item: item.score, reverse=True)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LlamaCppReranker:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
