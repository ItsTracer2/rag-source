"""API HTTP de RAG-Source.

    uv run uvicorn rag_source.api.app:app --port 8000

Trois points d'entrée : interroger (``/v1/ask``), chercher sans générer
(``/v1/search``, précieux pour déboguer la pertinence) et connaître l'état du
système (``/health``, qui affiche notamment où partent les données).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse

from rag_source.api.dependencies import Authenticated, ServicesDep, build_services
from rag_source.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentOut,
    DocumentsResponse,
    HealthResponse,
    PassageOut,
    SearchRequest,
    SearchResponse,
    ServiceHealth,
)
from rag_source.clients.embedder import EmbeddingError
from rag_source.clients.llm import LLMError
from rag_source.clients.reranker import RerankError
from rag_source.config import get_settings
from rag_source.generation.prompts import extract_citations
from rag_source.store.qdrant import StoreError

logger = logging.getLogger(__name__)

router = APIRouter()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.services = build_services(get_settings())
    try:
        yield
    finally:
        app.state.services.close()


app = FastAPI(
    title="RAG-Source",
    version="0.1.0",
    summary="Réponses sourcées et citées sur un corpus documentaire local.",
    lifespan=lifespan,
)


@router.get("/health", response_model=HealthResponse, tags=["système"])
def health(services: ServicesDep) -> HealthResponse:
    """État des services, et mode de souveraineté en vigueur.

    Le champ ``sovereignty`` n'est pas décoratif : il dit si les extraits du corpus
    quittent l'hôte. Une interface doit pouvoir l'afficher en permanence.
    """
    checks: list[ServiceHealth] = []
    chunks: int | None = None
    try:
        chunks = services.store.count()
        checks.append(ServiceHealth(name="qdrant", reachable=True))
    except StoreError as exc:
        checks.append(ServiceHealth(name="qdrant", reachable=False, detail=str(exc)[:200]))

    try:
        services.embedder.embed(["ping"])
        checks.append(ServiceHealth(name="embed", reachable=True))
    except EmbeddingError as exc:
        checks.append(ServiceHealth(name="embed", reachable=False, detail=str(exc)[:200]))

    try:
        services.reranker.rerank("ping", ["ping"])
        checks.append(ServiceHealth(name="rerank", reachable=True))
    except RerankError as exc:
        checks.append(ServiceHealth(name="rerank", reachable=False, detail=str(exc)[:200]))

    settings = services.settings
    return HealthResponse(
        status="ok" if all(check.reachable for check in checks) else "degraded",
        sovereignty=settings.sovereignty,
        llm_provider=settings.llm_provider.value,
        collection=settings.qdrant_collection,
        indexed_chunks=chunks,
        services=checks,
    )


@router.get("/v1/documents", response_model=DocumentsResponse, tags=["recherche"])
def documents(services: ServicesDep) -> DocumentsResponse:
    """Documents indexés, pour filtrer une recherche ou vérifier ce qui est en base."""
    try:
        counts = services.store.sources()
    except StoreError as exc:
        raise _unavailable(exc) from exc
    return DocumentsResponse(
        documents=[DocumentOut(source=source, chunks=chunks) for source, chunks in counts.items()],
        total_chunks=sum(counts.values()),
    )


@router.post("/v1/search", response_model=SearchResponse, tags=["recherche"])
def search(request: SearchRequest, services: ServicesDep) -> SearchResponse:
    """Recherche seule, sans génération.

    Sépare ce qui est trouvé de ce qui est dit : quand une réponse déçoit, c'est ici
    que l'on voit si le problème vient de la recherche ou du modèle.
    """
    try:
        result = services.search.search(
            request.query, mode=request.mode, source=request.source, top_k=request.top_k
        )
    except (StoreError, EmbeddingError, RerankError) as exc:
        raise _unavailable(exc) from exc
    return SearchResponse.of(result)


@router.post("/v1/ask", response_model=AskResponse, tags=["question"])
def ask(request: AskRequest, services: ServicesDep) -> AskResponse:
    """Réponse complète, avec ses citations vérifiées."""
    try:
        answer = services.answer.ask(
            request.question,
            history=request.as_pairs(),
            source=request.source,
            mode=request.mode,
            top_k=request.top_k,
        )
    except (StoreError, EmbeddingError, RerankError, LLMError) as exc:
        raise _unavailable(exc) from exc
    return AskResponse.of(answer)


@router.post("/v1/ask/stream", tags=["question"])
def ask_stream(request: AskRequest, services: ServicesDep) -> StreamingResponse:
    """Même réponse, en flux d'événements (SSE).

    Trois types d'événements : ``passages`` (immédiat, pour afficher les sources),
    ``token`` (au fil de la génération), puis ``done``. Les sources arrivent donc
    avant la réponse : sur CPU, l'attente serait autrement muette.
    """
    try:
        passages, stream = services.answer.stream(
            request.question,
            history=request.as_pairs(),
            source=request.source,
            mode=request.mode,
            top_k=request.top_k,
        )
    except (StoreError, EmbeddingError, RerankError, LLMError) as exc:
        raise _unavailable(exc) from exc

    def events() -> Iterator[str]:
        yield _event("passages", {"passages": [PassageOut.of(p).model_dump() for p in passages]})
        pieces: list[str] = []
        try:
            for piece in stream:
                pieces.append(piece)
                yield _event("token", {"text": piece})
        except LLMError as exc:  # le flux est déjà ouvert : on signale dans le flux
            logger.warning("Flux interrompu : %s", exc)
            yield _event("error", {"detail": str(exc)[:200]})
            return
        citations, invalid = extract_citations("".join(pieces), passages)
        yield _event(
            "done",
            {
                "citations": [
                    {"number": c.number, "source": c.source, "location": c.location}
                    for c in citations
                ],
                "invalid_citations": invalid,
                "refused": not passages,
            },
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _event(name: str, payload: dict[str, object]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _unavailable(exc: Exception) -> HTTPException:
    logger.error("Service indisponible : %s", exc)
    return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)[:300])


app.include_router(router, dependencies=[Authenticated])
