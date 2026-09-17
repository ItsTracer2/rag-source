"""Découpage des documents en chunks indexables.

Principes, et ce qu'ils corrigent par rapport à l'implémentation d'origine :

1. **Le découpage suit la structure du document.** Les sections produites par les
   chargeurs sont l'unité de base ; on ne coupe à l'intérieur que lorsqu'une section
   dépasse la taille maximale. L'original découpait par similarité sémantique entre
   phrases, ce qui coûtait deux passes d'embedding, ignorait les titres et
   s'arrêtait aux limites de page.
2. **La taille se mesure en tokens**, avec le tokenizer du modèle d'embedding.
3. **Chaque chunk porte son fil d'Ariane** (titre du document puis titres
   englobants). Le texte vectorisé est ainsi auto-suffisant : un passage qui dit
   « il doit être renouvelé tous les deux ans » reste compréhensible, et la source
   est citable telle quelle.
4. **Tableaux et enregistrements restent entiers.** S'il faut vraiment couper un
   tableau, l'en-tête est répété dans chaque morceau : sans lui, les cellules ne
   veulent plus rien dire.
5. **Les sections trop petites sont regroupées** avec leurs voisines de même titre :
   un chunk de dix mots n'est ni trouvable ni utile.
6. **Les identifiants sont déterministes** : ``uuid5(empreinte du fichier, rang)``.
   Deux indexations du même fichier produisent les mêmes identifiants.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from rag_source.domain import Chunk, LoadedDocument, Section, SectionKind
from rag_source.ingest.tokenizer import TokenCounter

BREADCRUMB = " › "
_NAMESPACE = uuid.UUID("0b5f7a52-7d1e-5f9a-9c6e-6f9f1a2b3c4d")
_PARAGRAPH = re.compile(r"\n{2,}")
# Fin de phrase : ponctuation forte suivie d'une espace et d'une majuscule ou d'un
# guillemet ouvrant. Volontairement simple et sans dépendance linguistique.
_SENTENCE = re.compile(r"(?<=[.!?…])\s+(?=[«\"'(\[A-ZÀ-ÖØ-Þ0-9])")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}")
# Le nombre de tokens d'un texte n'est pas exactement la somme de ses parties :
# coller le fil d'Ariane au corps peut fusionner ou scinder quelques tokens à la
# jointure. Cette marge évite de frôler la limite du modèle d'embedding.
_SAFETY_TOKENS = 8


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_tokens: int = 400
    """Taille visée : assez large pour porter un raisonnement, assez étroite pour
    que quatre à six chunks tiennent dans le contexte du LLM."""
    max_tokens: int = 700
    min_tokens: int = 60
    overlap_ratio: float = 0.15
    """Recouvrement entre deux chunks issus d'une même section, pour ne pas perdre
    une phrase à cheval. Ne s'applique pas aux tableaux ni aux enregistrements."""

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_ratio < 0.5:
            raise ValueError("overlap_ratio doit être dans [0, 0.5[.")
        if self.target_tokens > self.max_tokens:
            raise ValueError("target_tokens ne peut pas dépasser max_tokens.")

    @property
    def overlap_tokens(self) -> int:
        return int(self.target_tokens * self.overlap_ratio)


def chunk_document(
    document: LoadedDocument,
    counter: TokenCounter,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Découpe un document chargé en chunks prêts à être indexés."""
    config = config or ChunkingConfig()
    chunks: list[Chunk] = []
    for section in _merge_small_sections(document.sections, counter, config):
        for text in _split_section(section, document.title, counter, config):
            index = len(chunks)
            chunks.append(
                Chunk(
                    id=chunk_id(document.sha256, index),
                    source=document.source,
                    doc_sha256=document.sha256,
                    index=index,
                    text=text,
                    token_count=counter.count(text),
                    heading_path=section.heading_path,
                    kind=section.kind,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    metadata=section.metadata,
                )
            )
    return chunks


def chunk_id(doc_sha256: str, index: int) -> str:
    """Identifiant stable : même fichier et même rang donnent le même identifiant."""
    return str(uuid.uuid5(_NAMESPACE, f"{doc_sha256}:{index}"))


def _merge_small_sections(
    sections: tuple[Section, ...], counter: TokenCounter, config: ChunkingConfig
) -> Iterator[Section]:
    """Regroupe les sections voisines trop courtes partageant le même fil d'Ariane.

    Un titre suivi d'une seule phrase produirait sinon un chunk minuscule, noyé au
    classement et sans contexte pour le LLM.
    """
    pending: Section | None = None
    for section in sections:
        if pending is None:
            pending = section
            continue
        merged_tokens = counter.count(pending.text) + counter.count(section.text)
        can_merge = (
            pending.kind is SectionKind.PROSE
            and section.kind is SectionKind.PROSE
            and pending.heading_path == section.heading_path
            and counter.count(pending.text) < config.min_tokens
            and merged_tokens <= config.max_tokens
        )
        if can_merge:
            pending = Section(
                text=f"{pending.text}\n\n{section.text}",
                heading_path=pending.heading_path,
                kind=pending.kind,
                page_start=pending.page_start,
                page_end=section.page_end or pending.page_end,
                metadata=pending.metadata,
            )
        else:
            yield pending
            pending = section
    if pending is not None:
        yield pending


def _split_section(
    section: Section, title: str, counter: TokenCounter, config: ChunkingConfig
) -> list[str]:
    """Textes finaux d'une section, fil d'Ariane compris."""
    prefix = _breadcrumb(title, section.heading_path)
    budget = config.max_tokens - counter.count(prefix) - _SAFETY_TOKENS
    if budget <= 0:  # fil d'Ariane démesuré : on le réduit au titre du document
        prefix = f"{title}\n\n"
        budget = max(config.max_tokens - counter.count(prefix), config.min_tokens)

    body = section.text.strip()
    if counter.count(body) <= budget:
        return [prefix + body]

    if section.kind is SectionKind.TABLE:
        parts = _split_table(body, budget, counter)
    elif section.kind is SectionKind.RECORD:
        parts = _split_record(body, budget, counter)
    else:
        parts = _split_prose(body, budget, counter, config)
    return [prefix + part for part in parts]


def _breadcrumb(title: str, heading_path: tuple[str, ...]) -> str:
    trail = BREADCRUMB.join((title, *heading_path)) if title else BREADCRUMB.join(heading_path)
    return f"{trail}\n\n" if trail else ""


def _split_prose(
    body: str, budget: int, counter: TokenCounter, config: ChunkingConfig
) -> list[str]:
    """Découpe du texte courant, en respectant paragraphes puis phrases."""
    units = _prose_units(body, budget, counter)
    target = min(config.target_tokens, budget)

    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = counter.count(unit)
        if current and current_tokens + unit_tokens > target:
            parts.append("\n\n".join(current))
            current = _overlap(current, counter, config.overlap_tokens)
            current_tokens = sum(counter.count(item) for item in current)
            # Le recouvrement s'ajoute à l'unité suivante : il est réduit tant que
            # l'ensemble dépasserait la taille maximale, sans quoi un chunk hors
            # limite serait tronqué par le modèle d'embedding.
            while current and current_tokens + unit_tokens > budget:
                current_tokens -= counter.count(current.pop(0))
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        parts.append("\n\n".join(current))
    return parts


def _prose_units(body: str, budget: int, counter: TokenCounter) -> list[str]:
    """Plus petites unités insécables : paragraphes, puis phrases, puis mots."""
    units: list[str] = []
    for paragraph in _PARAGRAPH.split(body):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if counter.count(paragraph) <= budget:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if counter.count(sentence) <= budget:
                units.append(sentence)
            else:
                units.extend(_split_by_words(sentence, budget, counter))
    return units


def _split_by_words(text: str, budget: int, counter: TokenCounter) -> list[str]:
    """Dernier recours : une phrase interminable, ou une langue sans ponctuation."""
    parts: list[str] = []
    current: list[str] = []
    for word in text.split():
        candidate = " ".join([*current, word])
        if current and counter.count(candidate) > budget:
            parts.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        parts.append(" ".join(current))
    return parts


def _overlap(units: list[str], counter: TokenCounter, overlap_tokens: int) -> list[str]:
    """Dernières unités du chunk précédent, reprises en tête du suivant."""
    if overlap_tokens <= 0:
        return []
    kept: list[str] = []
    total = 0
    for unit in reversed(units):
        tokens = counter.count(unit)
        if total + tokens > overlap_tokens:
            break
        kept.insert(0, unit)
        total += tokens
    return kept


def _fit_lines(lines: list[str], budget: int, counter: TokenCounter) -> Iterator[str]:
    """Garantit qu'aucune ligne ne dépasse à elle seule le budget.

    Une cellule de tableau peut contenir un paragraphe entier : sans ce garde-fou,
    un seul enregistrement produit un chunk hors limite, tronqué à l'embedding.
    """
    for line in lines:
        if counter.count(line) <= budget:
            yield line
        else:
            yield from _split_by_words(line, budget, counter)


def _split_table(body: str, budget: int, counter: TokenCounter) -> list[str]:
    """Découpe un tableau Markdown en répétant sa ligne d'en-tête."""
    lines = body.splitlines()
    header: list[str] = []
    rows = lines
    if len(lines) >= 2 and _TABLE_SEPARATOR.search(lines[1]):
        header, rows = lines[:2], lines[2:]

    header_tokens = counter.count("\n".join(header))
    row_budget = max(budget - header_tokens, 1)
    parts: list[str] = []
    current: list[str] = []
    current_tokens = header_tokens
    for row in _fit_lines(rows, row_budget, counter):
        row_tokens = counter.count(row)
        if current and current_tokens + row_tokens > budget:
            parts.append("\n".join([*header, *current]))
            current = []
            current_tokens = header_tokens
        current.append(row)
        current_tokens += row_tokens
    if current:
        parts.append("\n".join([*header, *current]))
    return parts


def _split_record(body: str, budget: int, counter: TokenCounter) -> list[str]:
    """Découpe un enregistrement en répétant sa première ligne (l'identifiant)."""
    lines = body.splitlines()
    if not lines:
        return [body]
    key, rest = lines[0], lines[1:]
    key_tokens = counter.count(key)

    parts: list[str] = []
    current: list[str] = []
    current_tokens = key_tokens
    for line in _fit_lines(rest, max(budget - key_tokens, 1), counter):
        line_tokens = counter.count(line)
        if current and current_tokens + line_tokens > budget:
            parts.append("\n".join([key, *current]))
            current = []
            current_tokens = key_tokens
        current.append(line)
        current_tokens += line_tokens
    if current:
        parts.append("\n".join([key, *current]))
    return parts or [key]
