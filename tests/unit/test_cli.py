"""Tests de la ligne de commande.

L'API est simulée par un transport HTTP factice : on vérifie ce que la commande
envoie, ce qu'elle affiche, et surtout son **code de retour** — c'est lui qui rend
la commande utilisable dans un script ou une chaîne d'intégration continue.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from rag_source import cli
from rag_source.config import Settings

SETTINGS = Settings(_env_file=None, api_token="jeton-de-test")

PASSAGE = {
    "id": "id-1",
    "text": "Manuel › Entretien\n\nNettoyer le plateau avec un chiffon humide.",
    "source": "manuel.pdf",
    "location": "manuel.pdf, p. 4 — Entretien",
    "heading_path": ["Entretien"],
    "kind": "prose",
    "page_start": 4,
    "page_end": 4,
    "score": 3.2,
    "token_count": 42,
}

HEALTH = {
    "status": "ok",
    "sovereignty": "local",
    "llm_provider": "local",
    "collection": "rag_source",
    "indexed_chunks": 1114,
    "services": [{"name": "qdrant", "reachable": True, "detail": None}],
}


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Remplace l'API par un transport factice, et retient les requêtes reçues."""
    state: dict[str, Any] = {"requests": [], "responses": {}, "stream": []}

    def handler(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        key = f"{request.method} {request.url.path}"
        if key in state["responses"]:
            status, body = state["responses"][key]
            return httpx.Response(status, json=body)
        if request.url.path.endswith("/stream"):
            payload = "".join(
                f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in state["stream"]
            )
            return httpx.Response(200, text=payload, headers={"content-type": "text/event-stream"})
        return httpx.Response(404, json={"detail": "inconnu"})

    def api_client(settings: Settings, base_url: str) -> httpx.Client:
        return httpx.Client(
            base_url=base_url.rstrip("/"), transport=httpx.MockTransport(handler), timeout=5
        )

    monkeypatch.setattr(cli, "api_client", api_client)
    monkeypatch.setattr(cli, "get_settings", lambda: SETTINGS)
    return state


def run(*argv: str) -> int:
    return cli.main(list(argv))


class TestHealth:
    def test_prints_state_and_sovereignty(
        self, fake_api: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_api["responses"]["GET /health"] = (200, HEALTH)
        assert run("health") == 0
        output = capsys.readouterr().out
        assert "local" in output and "1114" in output

    def test_degraded_service_returns_nonzero(self, fake_api: dict[str, Any]) -> None:
        """Un code de retour non nul rend la commande utilisable en supervision."""
        fake_api["responses"]["GET /health"] = (200, {**HEALTH, "status": "degraded"})
        assert run("health") == 1

    def test_json_output_is_machine_readable(
        self, fake_api: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_api["responses"]["GET /health"] = (200, HEALTH)
        run("health", "--json")
        assert json.loads(capsys.readouterr().out)["collection"] == "rag_source"


class TestDocs:
    def test_lists_documents(
        self, fake_api: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_api["responses"]["GET /v1/documents"] = (
            200,
            {"documents": [{"source": "manuel.pdf", "chunks": 12}], "total_chunks": 12},
        )
        assert run("docs") == 0
        assert "manuel.pdf" in capsys.readouterr().out

    def test_empty_index_is_signalled(self, fake_api: dict[str, Any]) -> None:
        fake_api["responses"]["GET /v1/documents"] = (200, {"documents": [], "total_chunks": 0})
        assert run("docs") == 1


class TestSearch:
    def test_shows_passages(
        self, fake_api: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_api["responses"]["POST /v1/search"] = (
            200,
            {"query": "q", "mode": "hybrid+rerank", "candidates": 12, "passages": [PASSAGE]},
        )
        assert run("search", "plateau") == 0
        output = capsys.readouterr().out
        assert "manuel.pdf, p. 4 — Entretien" in output
        assert "chiffon humide" in output
        # Le fil d'Ariane du chunk n'est pas répété sous la référence.
        assert output.count("Entretien") == 1

    def test_no_result_returns_nonzero(self, fake_api: dict[str, Any]) -> None:
        fake_api["responses"]["POST /v1/search"] = (
            200,
            {"query": "q", "mode": "hybrid", "candidates": 12, "passages": []},
        )
        assert run("search", "gratin") == 1

    def test_options_are_sent(self, fake_api: dict[str, Any]) -> None:
        fake_api["responses"]["POST /v1/search"] = (
            200,
            {"query": "q", "mode": "dense", "candidates": 1, "passages": [PASSAGE]},
        )
        run("search", "q", "--mode", "dense", "--source", "manuel.pdf", "--top-k", "3")
        body = json.loads(fake_api["requests"][0].content)
        assert body == {"query": "q", "mode": "dense", "source": "manuel.pdf", "top_k": 3}


class TestAsk:
    def test_streams_sources_then_answer(
        self, fake_api: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_api["stream"] = [
            ("passages", {"passages": [PASSAGE]}),
            ("token", {"text": "La réponse "}),
            ("token", {"text": "[1]."}),
            (
                "done",
                {
                    "citations": [
                        {"number": 1, "source": "manuel.pdf", "location": "manuel.pdf, p. 4"}
                    ],
                    "invalid_citations": [],
                    "refused": False,
                },
            ),
        ]
        assert run("ask", "Comment nettoyer ?") == 0
        output = capsys.readouterr().out
        assert output.index("Sources retenues") < output.index("La réponse")
        assert "[1] manuel.pdf, p. 4" in output

    def test_refusal_returns_nonzero(self, fake_api: dict[str, Any]) -> None:
        fake_api["stream"] = [
            ("passages", {"passages": []}),
            ("token", {"text": "Je ne trouve pas…"}),
            ("done", {"citations": [], "invalid_citations": [], "refused": True}),
        ]
        assert run("ask", "gratin dauphinois") == 1

    def test_invalid_citations_are_shown(
        self, fake_api: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_api["stream"] = [
            ("passages", {"passages": [PASSAGE]}),
            ("token", {"text": "Affirmation [9]."}),
            ("done", {"citations": [], "invalid_citations": [9], "refused": False}),
        ]
        run("ask", "q")
        assert "Citations inexistantes : 9" in capsys.readouterr().out

    def test_no_stream_uses_the_simple_endpoint(self, fake_api: dict[str, Any]) -> None:
        fake_api["responses"]["POST /v1/ask"] = (
            200,
            {
                "question": "q",
                "search_query": "q",
                "answer": "Réponse [1].",
                "refused": False,
                "grounded": True,
                "citations": [],
                "invalid_citations": [],
                "passages": [PASSAGE],
            },
        )
        assert run("ask", "q", "--no-stream") == 0
        assert fake_api["requests"][0].url.path == "/v1/ask"


class TestErrors:
    def test_unreachable_api_is_explained(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def failing(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connexion refusée")

        monkeypatch.setattr(cli, "get_settings", lambda: SETTINGS)
        monkeypatch.setattr(
            cli,
            "api_client",
            lambda settings, url: httpx.Client(
                base_url=url, transport=httpx.MockTransport(failing)
            ),
        )
        assert run("health") == 2
        assert "docker compose up -d" in capsys.readouterr().err

    def test_rejected_token_is_explained(
        self, fake_api: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        fake_api["responses"]["GET /health"] = (401, {"detail": "non"})
        assert run("health") == 2
        assert "RAG_SOURCE_API_TOKEN" in capsys.readouterr().err


class TestParser:
    def test_unknown_command_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["danser"])

    def test_command_is_required(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([])

    @pytest.mark.parametrize(
        "command", ["health", "docs", "ask", "search", "ingest", "inspect", "eval"]
    )
    def test_every_command_has_a_handler(self, command: str) -> None:
        argv = [command, "x"] if command in {"ask", "search"} else [command]
        assert callable(cli.build_parser().parse_args(argv).handler)
