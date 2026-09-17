"""Chargeur HTML (BeautifulSoup, analyseur intégré à Python).

Les pages enregistrées depuis le web contiennent surtout du décor : navigation,
scripts, bandeaux de cookies, pieds de page. Il est retiré avant tout traitement,
puis les titres ``h1``–``h6`` construisent le fil d'Ariane et les tableaux sont
convertis en Markdown, comme pour les autres formats.
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup, Tag

from rag_source.domain import DocumentFormat, Section, SectionKind
from rag_source.ingest.loaders.base import Extracted, LoaderError, register_loader
from rag_source.ingest.text import normalize_spaces

_ENCODINGS = ("utf-8", "cp1252", "latin-1")
_NOISE = ("script", "style", "nav", "footer", "header", "aside", "noscript", "form", "svg")
_BLOCKS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "table", "dd")
_MAX_LEVEL = 6


@register_loader
class HtmlLoader:
    extensions = frozenset({".html", ".htm", ".xhtml"})
    format = DocumentFormat.HTML

    def load(self, path: Path) -> Extracted:
        soup = BeautifulSoup(_read_text(path), "html.parser")
        for tag in soup(list(_NOISE)):
            tag.decompose()

        title = normalize_spaces(soup.title.get_text()) if soup.title else ""
        root = soup.body or soup

        sections: list[Section] = []
        heading_path: list[str] = []
        buffer: list[str] = []

        def flush() -> None:
            text = "\n".join(buffer).strip()
            buffer.clear()
            if text:
                sections.append(
                    Section(text=text, heading_path=tuple(heading_path), kind=SectionKind.PROSE)
                )

        for element in root.find_all(list(_BLOCKS)):
            if not isinstance(element, Tag) or _inside_block(element):
                continue  # évite de restituer deux fois un bloc imbriqué
            name = element.name
            if name.startswith("h") and name[1:].isdigit():
                flush()
                heading = normalize_spaces(element.get_text(" "))
                if not heading:
                    continue
                level = min(int(name[1:]), _MAX_LEVEL)
                del heading_path[level - 1 :]
                heading_path.append(heading)
                title = title or heading
            elif name == "table":
                flush()
                if table := _table_markdown(element):
                    sections.append(
                        Section(
                            text=table, heading_path=tuple(heading_path), kind=SectionKind.TABLE
                        )
                    )
            elif text := normalize_spaces(element.get_text(" ")):
                buffer.append(f"- {text}" if name == "li" else text)
        flush()

        if not sections:
            raise LoaderError("Aucun contenu textuel dans la page.")
        return Extracted(title=title or path.stem, sections=tuple(sections))


def _inside_block(element: Tag) -> bool:
    """L'élément est-il contenu dans un autre bloc déjà traité (li dans table…) ?"""
    return any(parent.name in _BLOCKS for parent in element.parents if isinstance(parent, Tag))


def _table_markdown(table: Tag) -> str:
    rows: list[list[str]] = []
    for row in table.find_all("tr"):
        if not isinstance(row, Tag):
            continue
        cells = [
            normalize_spaces(cell.get_text(" "))
            for cell in row.find_all(["th", "td"])
            if isinstance(cell, Tag)
        ]
        if any(cells):
            rows.append(cells)
    if len(rows) < 2:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _read_text(path: Path) -> str:
    for encoding in _ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise LoaderError("Encodage non reconnu.")
