"""Chargeur texte brut (.txt, .log, .rst…).

Un fichier texte n'a pas de structure déclarée, mais en a souvent une visible :
titres soulignés (``====``), titres numérotés (``2.1 Installation``) ou en
capitales. On les reconnaît pour construire un fil d'Ariane ; à défaut, le
découpage se fait par paragraphes.
"""

from __future__ import annotations

import re
from pathlib import Path

from rag_source.domain import DocumentFormat, Section, SectionKind
from rag_source.ingest.loaders.base import Extracted, LoaderError, register_loader
from rag_source.ingest.text import is_bullet, normalize_spaces

_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
_UNDERLINE = re.compile(r"^[=\-~^#*_]{3,}$")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+\S")
_MAX_HEADING_CHARS = 90
# Profondeur maximale, pour éviter qu'un document très numéroté n'explose.
_MAX_LEVEL = 6


@register_loader
class PlainTextLoader:
    extensions = frozenset({".txt", ".text", ".log", ".rst"})
    format = DocumentFormat.TEXT

    def load(self, path: Path) -> Extracted:
        raw = _read_text(path)
        if not raw.strip():
            raise LoaderError("Fichier texte vide.")

        lines = raw.splitlines()
        sections: list[Section] = []
        heading_path: list[str] = []
        buffer: list[str] = []
        title = ""

        def flush() -> None:
            text = "\n".join(buffer).strip()
            buffer.clear()
            if text:
                sections.append(
                    Section(text=text, heading_path=tuple(heading_path), kind=SectionKind.PROSE)
                )

        index = 0
        while index < len(lines):
            line = normalize_spaces(lines[index])
            underlined = (
                bool(line)
                and index + 1 < len(lines)
                and bool(_UNDERLINE.match(lines[index + 1].strip()))
                and len(line) <= _MAX_HEADING_CHARS
            )
            followed_by_blank = index + 1 >= len(lines) or not lines[index + 1].strip()
            level = _heading_level(line, underlined, followed_by_blank)
            if level is not None:
                flush()
                del heading_path[level - 1 :]
                heading_path.append(line)
                title = title or line
                index += 2 if underlined else 1
                continue
            if not line and buffer:
                flush()
            elif line:
                buffer.append(line)
            index += 1
        flush()

        if not sections:
            raise LoaderError("Aucun contenu exploitable.")
        return Extracted(title=title or path.stem, sections=tuple(sections))


def _heading_level(line: str, underlined: bool, followed_by_blank: bool) -> int | None:
    """Niveau de titre d'une ligne, ou ``None`` si c'est du texte courant."""
    if not line or len(line) > _MAX_HEADING_CHARS:
        return None
    if underlined:
        return 1
    # « 1.1 Grounding » est un titre, « 1. Sortez le plateau » un élément de liste.
    # En texte brut, le seul indice fiable est la ligne vide qui suit un titre.
    if (match := _NUMBERED.match(line)) and followed_by_blank and not line.endswith((".", ";")):
        # Décalé d'un niveau : le niveau 1 est réservé au titre du document
        # (souligné ou en capitales), sous lequel la numérotation s'imbrique.
        return min(match.group(1).count(".") + 2, _MAX_LEVEL)
    if is_bullet(line):
        return None
    if line.isupper() and len(line.split()) <= 12:
        return 1
    return None


def _read_text(path: Path) -> str:
    for encoding in _ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise LoaderError("Encodage non reconnu.")
