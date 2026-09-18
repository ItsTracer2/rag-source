"""Construction des services et authentification de l'API.

Les clients HTTP vers les modèles et la base vectorielle sont créés une fois au
démarrage, puis partagés : ouvrir une connexion par requête gaspillerait le temps
d'établissement TCP à chaque question.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from rag_source.clients.embedder import LlamaCppEmbedder
from rag_source.clients.llm import ChatClient, OpenAICompatibleClient
from rag_source.clients.reranker import LlamaCppReranker
from rag_source.config import LLMProvider, Settings, Sovereignty
from rag_source.generation.answer import AnswerService
from rag_source.ingest.tokenizer import get_token_counter
from rag_source.retrieval.search import SearchConfig, SearchService
from rag_source.store.qdrant import QdrantStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Services:
    settings: Settings
    store: QdrantStore
    embedder: LlamaCppEmbedder
    reranker: LlamaCppReranker
    llm: ChatClient
    search: SearchService
    answer: AnswerService

    def close(self) -> None:
        self.store.close()
        self.embedder.close()
        self.reranker.close()
        close = getattr(self.llm, "close", None)
        if callable(close):
            close()


def build_services(settings: Settings) -> Services:
    api_key = (
        settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key is not None else None
    )
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, api_key=api_key)
    embedder = LlamaCppEmbedder(settings.embed_url)
    reranker = LlamaCppReranker(settings.rerank_url)
    llm = _build_llm(settings)
    search = SearchService(
        store,
        embedder,
        reranker,
        get_token_counter(settings.tokenizer_path),
        SearchConfig(),
    )
    if settings.sovereignty is Sovereignty.EXTERNAL:
        logger.warning(
            "⚠ LLM externe (%s) : les extraits du corpus quittent cet hôte. "
            "Mode visible dans /health.",
            settings.llm_base_url,
        )
    return Services(
        settings=settings,
        store=store,
        embedder=embedder,
        reranker=reranker,
        llm=llm,
        search=search,
        answer=AnswerService(search, llm),
    )


def _build_llm(settings: Settings) -> ChatClient:
    if settings.llm_provider is not LLMProvider.LOCAL:
        # Les fournisseurs externes arrivent à l'étape dédiée ; d'ici là, tout passe
        # par l'API compatible OpenAI, que la plupart d'entre eux exposent aussi.
        logger.info("Fournisseur LLM : %s", settings.llm_provider)
    return OpenAICompatibleClient(
        settings.llm_base_url,
        model=settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )


def get_services(request: Request) -> Services:
    services = getattr(request.app.state, "services", None)
    if not isinstance(services, Services):  # pragma: no cover - erreur de câblage
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Services non initialisés.")
    return services


def require_token(
    services: Annotated[Services, Depends(get_services)],
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Vérifie le jeton Bearer.

    La comparaison est à temps constant : comparer deux chaînes avec ``==`` laisse
    fuir, par le temps de réponse, le nombre de caractères corrects.
    """
    expected = services.settings.api_token
    if expected is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "RAG_SOURCE_API_TOKEN n'est pas défini : lancer ./scripts/init-env.sh",
        )
    provided = ""
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[len("bearer ") :].strip()
    if not secrets.compare_digest(provided, expected.get_secret_value()):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Jeton manquant ou invalide.",
            headers={"WWW-Authenticate": "Bearer"},
        )


ServicesDep = Annotated[Services, Depends(get_services)]
Authenticated = Depends(require_token)
