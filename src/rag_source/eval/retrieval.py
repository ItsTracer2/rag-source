"""Banc d'évaluation de la recherche.

Mesurer avant de régler. L'implémentation d'origine n'évaluait que la réponse
finale, avec un LLM de 3 milliards de paramètres comme juge de lui-même : un
signal lent, coûteux et peu fiable, qui ne dit surtout pas *où* le système échoue.

Ici, on mesure d'abord la recherche, séparément de la génération, avec des
métriques déterministes :

- **hit@k** : la bonne source apparaît-elle dans les k premiers passages ?
- **recall@k** : quelle part des sources attendues est retrouvée ?
- **MRR** : à quel rang apparaît la première bonne source (1, 1/2, 1/3…) ?
- **refus corrects** : sur les questions hors corpus, le système s'abstient-il ?

Aucun LLM n'intervient : les chiffres sont reproductibles et comparables d'une
stratégie à l'autre.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from rag_source.retrieval.search import SearchMode, SearchService


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    question: str
    expected_sources: tuple[str, ...]
    tags: tuple[str, ...] = ()

    @property
    def out_of_corpus(self) -> bool:
        """Question sans réponse dans le corpus : le bon comportement est l'abstention."""
        return not self.expected_sources


@dataclass(slots=True)
class CaseOutcome:
    case: EvalCase
    retrieved_sources: list[str]
    hit: bool
    recall: float
    reciprocal_rank: float
    abstained: bool


@dataclass(slots=True)
class EvalReport:
    mode: SearchMode
    top_k: int
    outcomes: list[CaseOutcome] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def answerable(self) -> list[CaseOutcome]:
        return [outcome for outcome in self.outcomes if not outcome.case.out_of_corpus]

    @property
    def unanswerable(self) -> list[CaseOutcome]:
        return [outcome for outcome in self.outcomes if outcome.case.out_of_corpus]

    @property
    def hit_rate(self) -> float:
        return _mean([1.0 if outcome.hit else 0.0 for outcome in self.answerable])

    @property
    def recall(self) -> float:
        return _mean([outcome.recall for outcome in self.answerable])

    @property
    def mrr(self) -> float:
        return _mean([outcome.reciprocal_rank for outcome in self.answerable])

    @property
    def abstention_rate(self) -> float:
        """Part des questions hors corpus pour lesquelles le système s'est abstenu."""
        return _mean([1.0 if outcome.abstained else 0.0 for outcome in self.unanswerable])

    @property
    def false_abstention_rate(self) -> float:
        """Part des questions légitimes auxquelles le système a refusé de répondre."""
        return _mean([1.0 if outcome.abstained else 0.0 for outcome in self.answerable])

    def row(self) -> str:
        return (
            f"{self.mode.value:<14} hit@{self.top_k} {self.hit_rate:5.0%}   "
            f"recall {self.recall:5.0%}   MRR {self.mrr:4.2f}   "
            f"abstention utile {self.abstention_rate:5.0%}   "
            f"abstention à tort {self.false_abstention_rate:5.0%}   "
            f"{self.seconds:5.1f} s"
        )

    def failures(self) -> list[CaseOutcome]:
        return [
            outcome
            for outcome in self.outcomes
            if (not outcome.hit and not outcome.case.out_of_corpus)
            or (outcome.case.out_of_corpus and not outcome.abstained)
        ]


def load_cases(path: Path) -> list[EvalCase]:
    return list(_read(path))


def _read(path: Path) -> Iterator[EvalCase]:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            yield EvalCase(
                id=raw["id"],
                question=raw["question"],
                expected_sources=tuple(raw.get("expected_sources", ())),
                tags=tuple(raw.get("tags", ())),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"{path}:{number} — cas de test invalide : {exc}") from exc


def evaluate(
    service: SearchService,
    cases: list[EvalCase],
    *,
    mode: SearchMode = SearchMode.HYBRID_RERANK,
    top_k: int = 6,
) -> EvalReport:
    import time

    started = time.monotonic()
    report = EvalReport(mode=mode, top_k=top_k)
    for case in cases:
        result = service.search(case.question, mode=mode, top_k=top_k)
        sources = _unique(passage.source for passage in result.passages)
        expected = set(case.expected_sources)
        found = [source for source in sources if source in expected]
        rank = next((i + 1 for i, source in enumerate(sources) if source in expected), 0)
        report.outcomes.append(
            CaseOutcome(
                case=case,
                retrieved_sources=sources,
                hit=bool(found),
                recall=len(found) / len(expected) if expected else 0.0,
                reciprocal_rank=1 / rank if rank else 0.0,
                abstained=not result.found,
            )
        )
    report.seconds = time.monotonic() - started
    return report


def _unique(values: Iterator[str]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
