"""Client d'embedding (llama-server, API compatible OpenAI).

Le serveur de modèles est joint par HTTP plutôt qu'embarqué dans le processus :
l'API reste légère, le modèle peut vivre sur une autre machine, et on peut le
remplacer sans toucher au code appelant.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 16
DEFAULT_TIMEOUT = 300.0


class EmbeddingError(RuntimeError):
    """Le service d'embedding est injoignable ou a renvoyé une réponse inattendue."""


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class LlamaCppEmbedder:
    """Embeddings via ``llama-server --embeddings``.

    Les textes sont envoyés par lots : un lot trop gros dépasse la taille de batch
    du serveur, un lot trop petit multiplie les allers-retours. La dimension est
    déduite du premier appel, puis vérifiée à chaque réponse — un modèle changé
    sous les pieds de l'index produirait sinon des vecteurs incomparables.
    """

    def __init__(
        self,
        base_url: str,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/v1/embeddings"
        self._batch_size = batch_size
        self._client = client or httpx.Client(timeout=timeout)
        self._dimension = 0

    @property
    def dimension(self) -> int:
        """Dimension des vecteurs, connue après le premier appel."""
        if self._dimension == 0:
            self.embed(["dimension"])
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = self._client.post(self._url, json={"input": batch})
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Service d'embedding injoignable ({self._url}) : {exc}") from exc

        try:
            items = sorted(payload["data"], key=lambda item: item["index"])
            vectors = [list(map(float, item["embedding"])) for item in items]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(f"Réponse d'embedding inexploitable : {exc}") from exc

        if len(vectors) != len(batch):
            raise EmbeddingError(f"{len(vectors)} vecteurs reçus pour {len(batch)} textes envoyés.")
        self._check_dimension(vectors)
        return vectors

    def _check_dimension(self, vectors: list[list[float]]) -> None:
        for vector in vectors:
            if self._dimension == 0:
                self._dimension = len(vector)
                logger.debug("Dimension d'embedding détectée : %d", self._dimension)
            elif len(vector) != self._dimension:
                raise EmbeddingError(
                    f"Dimension incohérente : {len(vector)} au lieu de {self._dimension}. "
                    "Le modèle d'embedding a-t-il changé en cours de route ?"
                )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LlamaCppEmbedder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
