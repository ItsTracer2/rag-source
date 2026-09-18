"""Génération de la réponse à partir des passages retrouvés.

Le service enchaîne : reformulation éventuelle de la question, recherche, puis
génération encadrée. Deux garde-fous encadrent le modèle :

- **si la recherche ne renvoie rien, le LLM n'est pas appelé.** Inutile de lui
  demander de répondre à partir de rien : on répond que l'information n'est pas
  dans les documents. Sur CPU, cela économise aussi plusieurs dizaines de secondes ;
- **les citations sont vérifiées** après génération, et les numéros inventés sont
  signalés au lieu d'être présentés comme des sources.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from rag_source.clients.llm import ChatClient
from rag_source.generation.prompts import (
    NO_CONTEXT_ANSWER,
    Citation,
    build_condense_messages,
    build_messages,
    extract_citations,
)
from rag_source.retrieval.search import Passage, SearchMode, SearchService

logger = logging.getLogger(__name__)

CONDENSE_MAX_TOKENS = 80
MAX_HISTORY_TURNS = 3
HISTORY_ANSWER_CHARS = 400


@dataclass(frozen=True, slots=True)
class Answer:
    question: str
    """Question posée par l'utilisateur."""
    search_query: str
    """Question réellement envoyée à la recherche (reformulée si suivi)."""
    text: str
    passages: list[Passage]
    citations: list[Citation] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    """Numéros cités par le modèle qui ne correspondent à aucun passage fourni."""

    @property
    def grounded(self) -> bool:
        """La réponse s'appuie-t-elle sur au moins une source valide ?"""
        return bool(self.citations)

    @property
    def refused(self) -> bool:
        return not self.passages


class AnswerService:
    def __init__(
        self,
        search: SearchService,
        llm: ChatClient,
        *,
        max_history_turns: int = MAX_HISTORY_TURNS,
    ) -> None:
        self._search = search
        self._llm = llm
        self._max_history_turns = max_history_turns

    def ask(
        self,
        question: str,
        *,
        history: list[tuple[str, str]] | None = None,
        source: str | None = None,
        mode: SearchMode = SearchMode.HYBRID_RERANK,
        top_k: int | None = None,
    ) -> Answer:
        query, passages = self._prepare(question, history, source, mode, top_k)
        if not passages:
            return Answer(
                question=question, search_query=query, text=NO_CONTEXT_ANSWER, passages=[]
            )

        messages = build_messages(question, passages, self._trim(history))
        text = self._llm.complete(messages).strip()
        citations, invalid = extract_citations(text, passages)
        if invalid:
            logger.warning("Citations inventées dans la réponse : %s", invalid)
        return Answer(
            question=question,
            search_query=query,
            text=text,
            passages=passages,
            citations=citations,
            invalid_citations=invalid,
        )

    def stream(
        self,
        question: str,
        *,
        history: list[tuple[str, str]] | None = None,
        source: str | None = None,
        mode: SearchMode = SearchMode.HYBRID_RERANK,
        top_k: int | None = None,
    ) -> tuple[list[Passage], Iterator[str]]:
        """Passages retenus, puis flux de la réponse.

        Les passages sont renvoyés d'abord : l'interface peut afficher les sources
        pendant que la réponse s'écrit, ce qui change tout quand le modèle produit
        quelques tokens par seconde.
        """
        _query, passages = self._prepare(question, history, source, mode, top_k)
        if not passages:
            return [], iter([NO_CONTEXT_ANSWER])
        messages = build_messages(question, passages, self._trim(history))
        return passages, self._llm.stream(messages)

    def _prepare(
        self,
        question: str,
        history: list[tuple[str, str]] | None,
        source: str | None,
        mode: SearchMode,
        top_k: int | None,
    ) -> tuple[str, list[Passage]]:
        query = self._condense(question, history) if history else question
        result = self._search.search(query, mode=mode, source=source, top_k=top_k)
        return query, result.passages

    def _condense(self, question: str, history: list[tuple[str, str]]) -> str:
        """Rend une question de suivi autonome, pour qu'elle soit interrogeable.

        L'appel est borné en tokens : il ne s'agit que de réécrire une phrase, et
        sur CPU chaque token coûte. En cas d'échec, on retombe sur la question
        d'origine plutôt que d'interrompre la réponse.
        """
        try:
            rewritten = self._llm.complete(
                build_condense_messages(question, self._trim(history)),
                max_tokens=CONDENSE_MAX_TOKENS,
                temperature=0.0,
            ).strip()
        except Exception as exc:  # le repli vaut mieux qu'une erreur pour l'utilisateur
            logger.warning("Reformulation impossible, question d'origine conservée : %s", exc)
            return question
        return rewritten.splitlines()[0].strip() if rewritten else question

    def _trim(self, history: list[tuple[str, str]] | None) -> list[tuple[str, str]]:
        """Derniers échanges, réponses tronquées.

        Un historique complet mange le contexte disponible pour les passages, qui
        sont la seule matière factuelle de la réponse.
        """
        if not history:
            return []
        return [
            (question, answer[:HISTORY_ANSWER_CHARS])
            for question, answer in history[-self._max_history_turns :]
        ]
