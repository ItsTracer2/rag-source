"""Schémas d'entrée et de sortie de l'API.

Ils constituent le contrat public : l'interface web, la ligne de commande et tout
autre client passent par là. La documentation OpenAPI en découle automatiquement.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag_source.config import Sovereignty
from rag_source.domain import SectionKind
from rag_source.generation.answer import Answer
from rag_source.retrieval.search import Passage, SearchMode, SearchResult

MAX_QUESTION_CHARS = 2000


class PassageOut(BaseModel):
    id: str
    text: str
    source: str
    location: str = Field(description="Référence lisible : fichier, page, titres.")
    heading_path: list[str]
    kind: SectionKind
    page_start: int | None = None
    page_end: int | None = None
    score: float
    token_count: int

    @classmethod
    def of(cls, passage: Passage) -> PassageOut:
        return cls(
            id=passage.id,
            text=passage.text,
            source=passage.source,
            location=passage.location,
            heading_path=list(passage.heading_path),
            kind=passage.kind,
            page_start=passage.page_start,
            page_end=passage.page_end,
            score=round(passage.score, 4),
            token_count=passage.token_count,
        )


class CitationOut(BaseModel):
    number: int
    source: str
    location: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    mode: SearchMode = SearchMode.HYBRID_RERANK
    source: str | None = Field(default=None, description="Restreint la recherche à un document.")
    top_k: int | None = Field(default=None, ge=1, le=20)


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    candidates: int
    passages: list[PassageOut]

    @classmethod
    def of(cls, result: SearchResult) -> SearchResponse:
        return cls(
            query=result.query,
            mode=result.mode,
            candidates=result.candidates,
            passages=[PassageOut.of(passage) for passage in result.passages],
        )


class Exchange(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[Exchange] = Field(default_factory=list, max_length=20)
    source: str | None = None
    mode: SearchMode = SearchMode.HYBRID_RERANK
    top_k: int | None = Field(default=None, ge=1, le=20)

    def as_pairs(self) -> list[tuple[str, str]]:
        return [(exchange.question, exchange.answer) for exchange in self.history]


class AskResponse(BaseModel):
    question: str
    search_query: str = Field(description="Question réellement recherchée (reformulée si suivi).")
    answer: str
    refused: bool = Field(description="Aucun passage pertinent : le modèle n'a pas été appelé.")
    grounded: bool = Field(description="La réponse cite au moins une source valide.")
    citations: list[CitationOut]
    invalid_citations: list[int] = Field(
        description="Numéros cités par le modèle ne correspondant à aucun passage fourni."
    )
    passages: list[PassageOut]

    @classmethod
    def of(cls, answer: Answer) -> AskResponse:
        return cls(
            question=answer.question,
            search_query=answer.search_query,
            answer=answer.text,
            refused=answer.refused,
            grounded=answer.grounded,
            citations=[
                CitationOut(
                    number=citation.number,
                    source=citation.source,
                    location=citation.location,
                )
                for citation in answer.citations
            ],
            invalid_citations=answer.invalid_citations,
            passages=[PassageOut.of(passage) for passage in answer.passages],
        )


class DocumentOut(BaseModel):
    source: str
    chunks: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentOut]
    total_chunks: int


class ServiceHealth(BaseModel):
    name: str
    reachable: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str = Field(description="« ok » si tous les services répondent, « degraded » sinon.")
    sovereignty: Sovereignty = Field(
        description="« local » : aucune donnée ne quitte l'hôte. « external » : "
        "les extraits du corpus sont envoyés à un tiers."
    )
    llm_provider: str
    collection: str
    indexed_chunks: int | None = None
    services: list[ServiceHealth]
