"""Modèles de domaine partagés par tout le pipeline.

Chaque chargeur de format (PDF, Markdown, Excel…) produit un :class:`LoadedDocument`
composé de :class:`Section`. Le reste du pipeline (découpage, indexation, recherche)
ne connaît que ces types et ignore le format d'origine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

MetadataValue = str | int | float | bool

_EMPTY: Mapping[str, MetadataValue] = MappingProxyType({})


class SectionKind(StrEnum):
    PROSE = "prose"
    """Texte courant : peut être découpé en plusieurs chunks."""
    TABLE = "table"
    """Tableau : conservé d'un seul tenant tant qu'il tient dans la taille max."""
    RECORD = "record"
    """Enregistrement structuré (ex. une ligne de recommandation Excel) : atomique."""


class DocumentFormat(StrEnum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    EXCEL = "excel"


@dataclass(frozen=True, slots=True)
class Section:
    text: str
    heading_path: tuple[str, ...] = ()
    """Titres englobants, du plus général au plus précis."""
    kind: SectionKind = SectionKind.PROSE
    page_start: int | None = None
    """Numéro de page (1-indexé) quand le format en a un."""
    page_end: int | None = None
    metadata: Mapping[str, MetadataValue] = field(default=_EMPTY)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Une section ne peut pas être vide.")
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("page_start et page_end doivent être définis ensemble.")
        if (
            self.page_start is not None
            and self.page_end is not None
            and (self.page_start < 1 or self.page_end < self.page_start)
        ):
            raise ValueError(f"Plage de pages invalide : {self.page_start}-{self.page_end}.")
        # Copie défensive : un appelant ne peut pas muter les métadonnées après coup.
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    source: str
    """Chemin relatif au dossier du corpus, séparateurs POSIX (identifiant stable)."""
    sha256: str
    """Empreinte du fichier : sert à la détection des modifications."""
    format: DocumentFormat
    title: str
    sections: tuple[Section, ...]

    def __post_init__(self) -> None:
        if len(self.sha256) != 64:
            raise ValueError("sha256 doit être une empreinte hexadécimale de 64 caractères.")


@dataclass(frozen=True, slots=True)
class Chunk:
    """Unité indexée et restituée comme passage citable."""

    id: str
    source: str
    doc_sha256: str
    index: int
    """Position du chunk dans son document (0-indexé)."""
    text: str
    """Texte envoyé à l'embedding et au LLM, préfixé du fil d'Ariane."""
    token_count: int
    heading_path: tuple[str, ...]
    kind: SectionKind
    page_start: int | None
    page_end: int | None
    metadata: Mapping[str, MetadataValue] = field(default=_EMPTY)

    def payload(self) -> dict[str, Any]:
        """Représentation sérialisable, stockée à côté des vecteurs."""
        return {
            "source": self.source,
            "doc_sha256": self.doc_sha256,
            "index": self.index,
            "text": self.text,
            "token_count": self.token_count,
            "heading_path": list(self.heading_path),
            "kind": self.kind.value,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "metadata": dict(self.metadata),
        }
