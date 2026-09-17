"""Accès à Qdrant, par son API REST.

Le client officiel n'est pas utilisé : il tire gRPC, numpy et portalocker pour ce
dont nous avons besoin — créer une collection, écrire des points, chercher, filtrer,
supprimer. Une quinzaine d'appels HTTP explicites sont plus faciles à lire, à tester
et à diagnostiquer qu'une couche d'abstraction supplémentaire.

La collection porte deux vecteurs par chunk : ``dense`` (bge-m3, sémantique) et
``sparse`` (BM25, mots-clés). Qdrant sait fusionner les deux classements côté
serveur, en une seule requête (étape suivante).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any

import httpx

from rag_source.domain import Chunk
from rag_source.store.sparse import SparseVector

logger = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"
DEFAULT_TIMEOUT = 120.0
_UPSERT_BATCH = 64
_SCROLL_PAGE = 512


class StoreError(RuntimeError):
    """Qdrant est injoignable, ou a refusé l'opération."""


class QdrantStore:
    def __init__(
        self,
        url: str,
        collection: str,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        headers = {"api-key": api_key} if api_key else {}
        self._collection = collection
        self._client = client or httpx.Client(
            base_url=url.rstrip("/"), timeout=timeout, headers=headers
        )

    # ── Collection ──────────────────────────────────────────────────────────

    def exists(self) -> bool:
        response = self._request("GET", f"/collections/{self._collection}", allow_404=True)
        return response is not None

    def create(self, dimension: int) -> None:
        """Crée la collection si elle n'existe pas, avec ses index de filtrage.

        ``modifier: idf`` demande à Qdrant de pondérer lui-même les termes rares :
        le client n'envoie que des fréquences, lui seul connaît le corpus complet.
        """
        if self.exists():
            return
        self._request(
            "PUT",
            f"/collections/{self._collection}",
            json={
                "vectors": {DENSE: {"size": dimension, "distance": "Cosine"}},
                "sparse_vectors": {SPARSE: {"modifier": "idf"}},
            },
        )
        # Sans index de charge utile, filtrer par source impose un parcours complet.
        for field, schema in (("source", "keyword"), ("rec_id", "keyword"), ("kind", "keyword")):
            self._request(
                "PUT",
                f"/collections/{self._collection}/index?wait=true",
                json={"field_name": field, "field_schema": schema},
            )
        logger.info("Collection %s créée (dimension %d).", self._collection, dimension)

    def drop(self) -> None:
        self._request("DELETE", f"/collections/{self._collection}", allow_404=True)

    def info(self) -> dict[str, Any]:
        result = self._request("GET", f"/collections/{self._collection}")
        assert result is not None
        return result

    def count(self) -> int:
        result = self._request(
            "POST", f"/collections/{self._collection}/points/count", json={"exact": True}
        )
        assert result is not None
        return int(result["count"])

    def dimension(self) -> int | None:
        """Dimension configurée, ou ``None`` si la collection n'existe pas."""
        response = self._request("GET", f"/collections/{self._collection}", allow_404=True)
        if response is None:
            return None
        vectors = response["config"]["params"]["vectors"]
        return int(vectors[DENSE]["size"])

    # ── Écriture ────────────────────────────────────────────────────────────

    def upsert(
        self, chunks: Sequence[Chunk], dense: Sequence[list[float]], sparse: Sequence[SparseVector]
    ) -> None:
        if not (len(chunks) == len(dense) == len(sparse)):
            raise ValueError("chunks, vecteurs denses et creux doivent être de même longueur.")
        for start in range(0, len(chunks), _UPSERT_BATCH):
            stop = start + _UPSERT_BATCH
            points = [
                {
                    "id": chunk.id,
                    "vector": {
                        DENSE: dense_vector,
                        SPARSE: {"indices": sparse_vector.indices, "values": sparse_vector.values},
                    },
                    "payload": chunk.payload(),
                }
                for chunk, dense_vector, sparse_vector in zip(
                    chunks[start:stop], dense[start:stop], sparse[start:stop], strict=True
                )
            ]
            self._request(
                "PUT", f"/collections/{self._collection}/points?wait=true", json={"points": points}
            )

    def query(
        self,
        *,
        dense: list[float] | None = None,
        sparse: SparseVector | None = None,
        limit: int = 30,
        prefetch: int = 40,
        source: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recherche hybride : les deux classements sont fusionnés par Qdrant.

        La fusion RRF (Reciprocal Rank Fusion) combine les *rangs*, pas les scores :
        un score cosinus et un score BM25 ne sont pas comparables, leurs rangs si.
        Un passage bien classé par les deux méthodes remonte ainsi devant un passage
        excellent pour une seule.

        Une seule requête HTTP fait le tout : ni le trafic ni la logique de fusion ne
        remontent côté client.
        """
        if dense is None and sparse is None:
            raise ValueError("Au moins un des deux vecteurs est requis.")

        conditions = [
            {"key": key, "match": {"value": value}}
            for key, value in (("source", source), ("kind", kind))
            if value is not None
        ]
        query_filter = {"must": conditions} if conditions else None

        prefetches: list[dict[str, Any]] = []
        if dense is not None:
            prefetches.append({"query": dense, "using": DENSE, "limit": prefetch})
        if sparse is not None and len(sparse) > 0:
            prefetches.append(
                {
                    "query": {"indices": sparse.indices, "values": sparse.values},
                    "using": SPARSE,
                    "limit": prefetch,
                }
            )

        body: dict[str, Any] = {"limit": limit, "with_payload": True}
        if len(prefetches) == 1:
            body.update(prefetches[0])
        else:
            body["prefetch"] = prefetches
            body["query"] = {"fusion": "rrf"}
        if query_filter is not None:
            body["filter"] = query_filter

        result = self._request("POST", f"/collections/{self._collection}/points/query", json=body)
        assert result is not None
        points: list[dict[str, Any]] = result["points"]
        return points

    def delete_source(self, source: str) -> None:
        """Supprime tous les chunks d'un document."""
        self._request(
            "POST",
            f"/collections/{self._collection}/points/delete?wait=true",
            json={"filter": {"must": [{"key": "source", "match": {"value": source}}]}},
        )

    # ── Lecture ─────────────────────────────────────────────────────────────

    def indexed_versions(self) -> dict[str, str]:
        """Empreinte du fichier indexé, par document.

        C'est la mémoire de ce qui a déjà été traité : l'index est sa propre source
        de vérité, sans fichier d'état annexe qui pourrait diverger.
        """
        versions: dict[str, str] = {}
        for payload in self._scroll(("source", "doc_sha256")):
            source = payload.get("source")
            sha = payload.get("doc_sha256")
            if isinstance(source, str) and isinstance(sha, str):
                versions.setdefault(source, sha)
        return versions

    def _scroll(self, fields: tuple[str, ...]) -> Iterator[dict[str, Any]]:
        offset: Any = None
        while True:
            body: dict[str, Any] = {
                "limit": _SCROLL_PAGE,
                "with_payload": list(fields),
                "with_vector": False,
            }
            if offset is not None:
                body["offset"] = offset
            result = self._request(
                "POST", f"/collections/{self._collection}/points/scroll", json=body
            )
            assert result is not None
            for point in result["points"]:
                yield point.get("payload") or {}
            offset = result.get("next_page_offset")
            if offset is None:
                return

    # ── Transport ───────────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        allow_404: bool = False,
    ) -> dict[str, Any] | None:
        try:
            response = self._client.request(method, path, json=json)
        except httpx.HTTPError as exc:
            raise StoreError(f"Qdrant injoignable : {exc}") from exc
        if response.status_code == 404 and allow_404:
            return None
        if response.status_code >= 400:
            raise StoreError(
                f"Qdrant a refusé {method} {path} ({response.status_code}) : {response.text[:300]}"
            )
        payload = response.json()
        result = payload.get("result")
        return result if isinstance(result, dict) else {"result": result}

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> QdrantStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
