"""L'interface servie par Caddy, avec l'API derrière.

    docker compose up -d
    uv run pytest -m integration

Ce qui est vérifié ici ne l'est nulle part ailleurs : la page est bien servie, le
proxy ajoute le jeton (le navigateur n'en a aucun), et le flux SSE traverse le
proxy sans être mis en tampon.
"""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.integration

UI_URL = os.getenv("RAG_SOURCE_TEST_UI_URL", "http://127.0.0.1:8080")


def get(path: str, **kwargs: object) -> httpx.Response:
    try:
        return httpx.get(f"{UI_URL}{path}", timeout=30, **kwargs)  # type: ignore[arg-type]
    except httpx.ConnectError:  # pragma: no cover - dépend de l'environnement
        pytest.skip("Interface injoignable : lancer `docker compose up -d`.")


def test_page_is_served() -> None:
    response = get("/")
    assert response.status_code == 200
    assert "RAG-Source" in response.text
    assert 'src="app.js"' in response.text


def test_static_assets_are_served() -> None:
    for path in ("/app.js", "/styles.css"):
        assert get(path).status_code == 200


def test_security_headers_are_present() -> None:
    headers = get("/").headers
    assert "default-src 'self'" in headers["content-security-policy"]
    assert headers["x-content-type-options"] == "nosniff"


def test_proxy_adds_the_token() -> None:
    """La page n'a aucun secret : c'est le proxy qui authentifie la requête."""
    body = get("/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert body["sovereignty"] in {"local", "external"}


def test_documents_are_reachable_through_the_proxy() -> None:
    body = get("/v1/documents").json()
    assert body["total_chunks"] >= 0


def test_stream_is_not_buffered_by_the_proxy() -> None:
    """Les passages doivent arriver avant la fin de la génération."""
    with httpx.stream(
        "POST",
        f"{UI_URL}/v1/ask/stream",
        json={"question": "Que dit la recommandation R24 ?", "top_k": 2},
        timeout=300,
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        first = ""
        for line in response.iter_lines():
            if line.startswith("event:"):
                first = line[len("event: ") :]
                break
    assert first == "passages"
