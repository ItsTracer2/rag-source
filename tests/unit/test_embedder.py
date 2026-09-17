from __future__ import annotations

from typing import Any

import httpx
import pytest

from rag_source.clients.embedder import EmbeddingError, LlamaCppEmbedder


def make_embedder(handler: Any, **kwargs: Any) -> LlamaCppEmbedder:
    transport = httpx.MockTransport(handler)
    return LlamaCppEmbedder("http://embed:8080", client=httpx.Client(transport=transport), **kwargs)


def embeddings_response(request: httpx.Request) -> httpx.Response:
    import json

    texts = json.loads(request.content)["input"]
    return httpx.Response(
        200,
        json={
            "data": [
                {"index": i, "embedding": [float(len(text)), 0.5, 0.25]}
                for i, text in enumerate(texts)
            ]
        },
    )


def test_returns_one_vector_per_text() -> None:
    embedder = make_embedder(embeddings_response)
    vectors = embedder.embed(["abc", "de"])
    assert vectors == [[3.0, 0.5, 0.25], [2.0, 0.5, 0.25]]


def test_dimension_is_detected_once() -> None:
    embedder = make_embedder(embeddings_response)
    assert embedder.dimension == 3


def test_batches_large_inputs() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        calls.append(len(json.loads(request.content)["input"]))
        return embeddings_response(request)

    embedder = make_embedder(handler, batch_size=2)
    embedder.embed(["a", "b", "c", "d", "e"])
    assert calls == [2, 2, 1]


def test_preserves_order_even_if_server_shuffles() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [2.0]},
                    {"index": 0, "embedding": [1.0]},
                ]
            },
        )

    embedder = make_embedder(handler)
    assert embedder.embed(["premier", "second"]) == [[1.0], [2.0]]


def test_inconsistent_dimension_is_rejected() -> None:
    """Un modèle changé en cours de route produirait des vecteurs incomparables."""
    responses = iter(
        [
            httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]}),
            httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}]}),
        ]
    )
    embedder = make_embedder(lambda request: next(responses))
    embedder.embed(["premier"])
    with pytest.raises(EmbeddingError, match="Dimension incohérente"):
        embedder.embed(["second"])


def test_unreachable_service_is_explicit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connexion refusée")

    with pytest.raises(EmbeddingError, match="injoignable"):
        make_embedder(handler).embed(["texte"])


def test_malformed_response_is_explicit() -> None:
    embedder = make_embedder(lambda request: httpx.Response(200, json={"oops": True}))
    with pytest.raises(EmbeddingError, match="inexploitable"):
        embedder.embed(["texte"])


def test_missing_vectors_are_detected() -> None:
    embedder = make_embedder(
        lambda request: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})
    )
    with pytest.raises(EmbeddingError, match="1 vecteurs reçus pour 2"):
        embedder.embed(["a", "b"])
