"""Recherche : hybride, reclassée, puis filtrée sur un budget de tokens.

La chaîne complète :

1. la question est vectorisée deux fois — densément (sens) et en mots-clés (BM25) ;
2. Qdrant fusionne les deux classements (RRF) et renvoie quelques dizaines de
   candidats ;
3. le reranker les relit un par un, avec la question, et les réordonne ;
4. les passages trop faibles sont écartés, et l'on ne garde que ce qui tient dans
   le budget de contexte alloué au LLM.

Les étapes 3 et 4 sont celles qui manquent le plus souvent. Sans reclassement, la
recherche vectorielle ramène des passages « du bon sujet » mais pas forcément
« qui répondent ». Sans seuil, le système présente toujours quelque chose, même
quand le corpus ne contient pas la réponse — et le LLM, lui, brodera dessus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from rag_source.clients.embedder import Embedder
from rag_source.clients.reranker import Reranker
from rag_source.domain import MetadataValue, SectionKind
from rag_source.ingest.tokenizer import TokenCounter
from rag_source.store.qdrant import QdrantStore
from rag_source.store.sparse import encode_query

logger = logging.getLogger(__name__)


class SearchMode(StrEnum):
    """Modes disponibles, surtout utiles pour comparer les stratégies à l'évaluation."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid+rerank"


@dataclass(frozen=True, slots=True)
class SearchConfig:
    prefetch: int = 40
    """Candidats ramenés par chacune des deux recherches avant fusion."""
    candidates: int = 12
    """Candidats soumis au reranker.

    Valeur choisie par la mesure, pas au jugé : sur le jeu de référence, 12 et 30
    candidats donnent les mêmes hit@6, MRR et abstention, mais 12 divise le temps
    de reranking par 2,3 (12 s contre 28 s par question sur CPU). Le reranking est
    l'étape la plus coûteuse de la chaîne : c'est là que se règle la latence."""
    top_k: int = 6
    """Passages conservés au maximum."""
    min_score: float = -5.0
    """Seuil de pertinence, sur le score du reranker.

    Les scores ne sont pas bornés et ne sont pas centrés sur zéro : mesurés sur
    bge-reranker-v2-m3, un passage qui répond obtient entre -2,5 et 0, un passage
    hors sujet environ -11. La séparation est franche, mais elle ne passe pas par
    zéro — d'où un seuil mesuré plutôt que supposé."""
    context_tokens: int = 2500
    """Budget de tokens pour l'ensemble des passages retenus."""


@dataclass(frozen=True, slots=True)
class Passage:
    """Un extrait retenu, prêt à être cité."""

    id: str
    text: str
    source: str
    heading_path: tuple[str, ...]
    kind: SectionKind
    page_start: int | None
    page_end: int | None
    score: float
    """Score du reranker, ou score de fusion si le reranking est désactivé."""
    token_count: int
    metadata: dict[str, MetadataValue]

    @property
    def location(self) -> str:
        """Référence lisible : « manuel.pdf, p. 12 — Entretien › Plateau »."""
        parts = [self.source]
        if self.page_start is not None:
            pages = (
                f"p. {self.page_start}"
                if self.page_start == self.page_end
                else f"p. {self.page_start}-{self.page_end}"
            )
            parts.append(pages)
        trail = " › ".join(self.heading_path)
        reference = ", ".join(parts)
        return f"{reference} — {trail}" if trail else reference


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    passages: list[Passage]
    candidates: int
    """Nombre de candidats examinés avant reclassement."""
    mode: SearchMode

    @property
    def found(self) -> bool:
        return bool(self.passages)


class SearchService:
    def __init__(
        self,
        store: QdrantStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        counter: TokenCounter | None = None,
        config: SearchConfig | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._counter = counter
        self._config = config or SearchConfig()

    def search(
        self,
        query: str,
        *,
        mode: SearchMode = SearchMode.HYBRID_RERANK,
        source: str | None = None,
        kind: str | None = None,
        top_k: int | None = None,
    ) -> SearchResult:
        config = self._config
        limit = top_k or config.top_k

        dense = None if mode is SearchMode.SPARSE else self._embedder.embed([query])[0]
        sparse = None if mode is SearchMode.DENSE else encode_query(query)

        points = self._store.query(
            dense=dense,
            sparse=sparse,
            limit=config.candidates,
            prefetch=config.prefetch,
            source=source,
            kind=kind,
        )
        passages = [_to_passage(point) for point in points]

        if mode is SearchMode.HYBRID_RERANK and self._reranker is not None and passages:
            passages = self._rerank(query, passages)

        selected = self._select(passages, limit, config)
        logger.debug(
            "Recherche « %s » : %d candidats, %d retenus.", query, len(passages), len(selected)
        )
        return SearchResult(query=query, passages=selected, candidates=len(passages), mode=mode)

    def _rerank(self, query: str, passages: list[Passage]) -> list[Passage]:
        scores = self._reranker.rerank(query, [passage.text for passage in passages])  # type: ignore[union-attr]
        reordered = []
        for score in scores:
            passage = passages[score.index]
            reordered.append(
                Passage(
                    id=passage.id,
                    text=passage.text,
                    source=passage.source,
                    heading_path=passage.heading_path,
                    kind=passage.kind,
                    page_start=passage.page_start,
                    page_end=passage.page_end,
                    score=score.score,
                    token_count=passage.token_count,
                    metadata=passage.metadata,
                )
            )
        return reordered

    def _select(self, passages: list[Passage], limit: int, config: SearchConfig) -> list[Passage]:
        """Garde les passages pertinents qui tiennent dans le budget de contexte.

        Le seuil ne s'applique qu'aux scores du reranker : les scores de fusion RRF
        sont des rangs inversés, sans signification absolue.
        """
        kept: list[Passage] = []
        used = 0
        for passage in passages:
            if len(kept) >= limit:
                break
            if self._reranker is not None and passage.score < config.min_score and kept:
                break  # la liste est triée : tout ce qui suit est encore moins pertinent
            tokens = passage.token_count or self._count(passage.text)
            if kept and used + tokens > config.context_tokens:
                break
            kept.append(passage)
            used += tokens
        if kept and self._reranker is not None and kept[0].score < config.min_score:
            return []  # même le meilleur passage est hors sujet
        return kept

    def _count(self, text: str) -> int:
        return self._counter.count(text) if self._counter is not None else len(text) // 3


def _to_passage(point: dict[str, object]) -> Passage:
    payload = point.get("payload") or {}
    assert isinstance(payload, dict)
    metadata = payload.get("metadata") or {}
    assert isinstance(metadata, dict)
    return Passage(
        id=str(point.get("id", "")),
        text=str(payload.get("text", "")),
        source=str(payload.get("source", "")),
        heading_path=tuple(payload.get("heading_path") or ()),
        kind=SectionKind(payload.get("kind", SectionKind.PROSE)),
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        score=float(point.get("score", 0.0)),  # type: ignore[arg-type]
        token_count=int(payload.get("token_count", 0)),
        metadata=dict(metadata),
    )
