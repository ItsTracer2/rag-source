"""Tests du prompt, des citations et du service de réponse.

Le LLM est simulé : ce qui est vérifié ici, c'est l'encadrement autour de lui —
ce qu'on lui envoie, ce qu'on accepte de sa réponse, et ce qu'on refuse de lui
demander.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest

from rag_source.clients.llm import Message
from rag_source.domain import SectionKind
from rag_source.generation.answer import AnswerService
from rag_source.generation.prompts import (
    NO_CONTEXT_ANSWER,
    build_condense_messages,
    build_messages,
    extract_citations,
)
from rag_source.retrieval.search import Passage, SearchMode, SearchResult


def passage(index: int, source: str = "manuel.pdf", page: int | None = 3) -> Passage:
    return Passage(
        id=f"id-{index}",
        text=f"Contenu du passage {index}.",
        source=source,
        heading_path=("Entretien", f"Section {index}"),
        kind=SectionKind.PROSE,
        page_start=page,
        page_end=page,
        score=5.0 - index,
        token_count=50,
        metadata={},
    )


class FakeLLM:
    """Renvoie des réponses décidées à l'avance et retient les messages reçus."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses) or ["Réponse. [1]"]
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self.calls.append(list(messages))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    def stream(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        self.calls.append(list(messages))
        yield from self.responses[0].split(" ")


class FakeSearch:
    def __init__(self, passages: list[Passage]) -> None:
        self.passages = passages
        self.queries: list[str] = []

    def search(self, query: str, **kwargs: object) -> SearchResult:
        self.queries.append(query)
        return SearchResult(
            query=query,
            passages=self.passages,
            candidates=len(self.passages),
            mode=SearchMode.HYBRID_RERANK,
        )


def service(passages: list[Passage], *responses: str) -> tuple[AnswerService, FakeSearch, FakeLLM]:
    search, llm = FakeSearch(passages), FakeLLM(*responses)
    return AnswerService(search, llm), search, llm  # type: ignore[arg-type]


class TestPrompt:
    def test_passages_are_numbered_and_located(self) -> None:
        """Sans numéro ni provenance, le modèle ne peut pas citer, même en le voulant."""
        messages = build_messages("Comment nettoyer ?", [passage(1), passage(2)])
        content = messages[-1].content
        assert "[1] (manuel.pdf, p. 3 — Entretien › Section 1)" in content
        assert "[2] (manuel.pdf, p. 3 — Entretien › Section 2)" in content
        assert content.rstrip().endswith("Question : Comment nettoyer ?")

    def test_system_prompt_allows_saying_no(self) -> None:
        system = build_messages("q", [passage(1)])[0]
        assert system.role == "system"
        assert "ne trouve pas" in system.content.lower()
        assert "langue de la question" in system.content

    def test_history_becomes_a_conversation(self) -> None:
        messages = build_messages("Et ensuite ?", [passage(1)], [("Première ?", "Réponse.")])
        assert [message.role for message in messages] == ["system", "user", "assistant", "user"]

    def test_condense_prompt_asks_for_a_standalone_question(self) -> None:
        messages = build_condense_messages(
            "Et pour le suivant ?", [("Quel est le premier ?", "A.")]
        )
        assert "autonome" in messages[0].content
        assert "Et pour le suivant ?" in messages[-1].content


class TestCitations:
    def test_extracts_valid_citations_in_order(self) -> None:
        passages = [passage(1), passage(2), passage(3)]
        citations, invalid = extract_citations("Fait A [2]. Fait B [1].", passages)
        assert [citation.number for citation in citations] == [1, 2]
        assert invalid == []

    def test_duplicates_count_once(self) -> None:
        citations, _ = extract_citations("A [1]. B [1]. C [1].", [passage(1)])
        assert len(citations) == 1

    def test_invented_numbers_are_flagged(self) -> None:
        """Un numéro hors liste signale une réponse qui s'écarte de ses sources."""
        citations, invalid = extract_citations("Affirmation [7].", [passage(1)])
        assert citations == []
        assert invalid == [7]

    def test_no_citation_at_all(self) -> None:
        citations, invalid = extract_citations("Une réponse sans aucune source.", [passage(1)])
        assert citations == [] and invalid == []

    def test_citation_carries_its_location(self) -> None:
        citations, _ = extract_citations("A [1].", [passage(1, source="dossier/guide.pdf")])
        assert citations[0].source == "dossier/guide.pdf"
        assert "p. 3" in citations[0].location


class TestAnswerService:
    def test_answers_with_citations(self) -> None:
        svc, _, _ = service([passage(1), passage(2)], "La réponse est là [1].")
        answer = svc.ask("Comment nettoyer ?")
        assert answer.grounded and not answer.refused
        assert [citation.number for citation in answer.citations] == [1]

    def test_llm_is_not_called_without_passages(self) -> None:
        """Rien à fournir au modèle : le faire répondre quand même invite à inventer."""
        svc, _, llm = service([], "réponse inattendue")
        answer = svc.ask("Question hors corpus")
        assert answer.refused
        assert answer.text == NO_CONTEXT_ANSWER
        assert llm.calls == []

    def test_invalid_citations_are_reported(self) -> None:
        svc, _, _ = service([passage(1)], "Affirmation [4].")
        answer = svc.ask("q")
        assert answer.invalid_citations == [4]
        assert not answer.grounded

    def test_follow_up_question_is_condensed(self) -> None:
        svc, search, _llm = service([passage(1)], "Question autonome réécrite ?", "Réponse [1].")
        answer = svc.ask("Et ensuite ?", history=[("Quel est le premier ?", "Le premier est A.")])
        assert search.queries == ["Question autonome réécrite ?"]
        assert answer.search_query == "Question autonome réécrite ?"
        assert answer.question == "Et ensuite ?"

    def test_no_condensation_without_history(self) -> None:
        svc, search, llm = service([passage(1)], "Réponse [1].")
        svc.ask("Question initiale ?")
        assert search.queries == ["Question initiale ?"]
        assert len(llm.calls) == 1  # une seule génération, pas de reformulation

    def test_condensation_failure_falls_back_to_the_question(self) -> None:
        class FailingLLM(FakeLLM):
            def complete(self, messages: Sequence[Message], **kwargs: object) -> str:
                if any("autonome" in message.content for message in messages):
                    raise RuntimeError("modèle indisponible")
                return "Réponse [1]."

        search, llm = FakeSearch([passage(1)]), FailingLLM()
        svc = AnswerService(search, llm)  # type: ignore[arg-type]
        answer = svc.ask("Et ensuite ?", history=[("Q", "A")])
        assert search.queries == ["Et ensuite ?"]
        assert answer.text.startswith("Réponse")

    def test_history_is_trimmed(self) -> None:
        svc, _, llm = service([passage(1)], "Réponse [1].")
        history = [(f"Question {i}", "R" * 1000) for i in range(6)]
        svc.ask("Suite ?", history=history)
        conversation = llm.calls[-1]
        assert len(conversation) == 2 + 2 * 3  # système + 3 échanges + question
        assert all(len(message.content) <= 1000 for message in conversation[1:-1])

    def test_stream_returns_passages_first(self) -> None:
        svc, _, _ = service([passage(1)], "Réponse en flux [1].")
        passages, stream = svc.stream("q")
        assert len(passages) == 1
        assert "".join(stream) == "Réponseenflux[1]."

    def test_stream_refuses_without_passages(self) -> None:
        svc, _, llm = service([], "jamais appelé")
        passages, stream = svc.stream("q")
        assert passages == []
        assert "".join(stream) == NO_CONTEXT_ANSWER
        assert llm.calls == []


@pytest.mark.parametrize("answer", ["Réponse [1] et [2].", "Voir [2], puis [1]."])
def test_citation_order_is_always_ascending(answer: str) -> None:
    citations, _ = extract_citations(answer, [passage(1), passage(2)])
    assert [citation.number for citation in citations] == [1, 2]
