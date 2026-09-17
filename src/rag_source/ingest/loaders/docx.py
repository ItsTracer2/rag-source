"""Chargeur DOCX (python-docx).

Word porte sa structure dans les styles : ``Heading 1``, ``Titre 2``, ``Liste à
puces``. On s'en sert plutôt que de deviner, et on parcourt le corps dans l'ordre
du document pour que les tableaux restent à leur place entre deux paragraphes.
"""

from __future__ import annotations

import re
from pathlib import Path

import docx
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from rag_source.domain import DocumentFormat, Section, SectionKind
from rag_source.ingest.loaders.base import Extracted, LoaderError, register_loader
from rag_source.ingest.text import normalize_spaces

# Les noms de style sont localisés par Word : « Heading 2 », « Titre 2 », « Überschrift 2 ».
_HEADING_STYLE = re.compile(r"^(?:heading|titre|title|überschrift|encabezado)\s*(\d+)?", re.I)
_LIST_STYLE = re.compile(r"list|liste|puces|bullet|number", re.I)
_MAX_LEVEL = 6


@register_loader
class DocxLoader:
    extensions = frozenset({".docx"})
    format = DocumentFormat.DOCX

    def load(self, path: Path) -> Extracted:
        try:
            document = docx.Document(str(path))
        except Exception as exc:
            raise LoaderError(f"Document Word illisible : {exc}") from exc

        sections: list[Section] = []
        heading_path: list[str] = []
        buffer: list[str] = []
        title = _core_title(document)

        def flush() -> None:
            text = "\n".join(buffer).strip()
            buffer.clear()
            if text:
                sections.append(
                    Section(text=text, heading_path=tuple(heading_path), kind=SectionKind.PROSE)
                )

        for block in _iter_blocks(document):
            if isinstance(block, Table):
                flush()
                if table := _table_markdown(block):
                    sections.append(
                        Section(
                            text=table, heading_path=tuple(heading_path), kind=SectionKind.TABLE
                        )
                    )
                continue
            text = normalize_spaces(block.text)
            if not text:
                continue
            style = block.style.name if block.style is not None else ""
            if (level := _heading_level(style)) is not None:
                flush()
                del heading_path[level - 1 :]
                heading_path.append(text)
                title = title or text
            elif _LIST_STYLE.search(style or ""):
                buffer.append(f"- {text}")
            else:
                buffer.append(text)
        flush()

        if not sections:
            raise LoaderError("Document Word sans contenu exploitable.")
        return Extracted(title=title or path.stem, sections=tuple(sections))


def _iter_blocks(document: DocxDocument) -> list[Paragraph | Table]:
    """Paragraphes et tableaux du corps, dans l'ordre du document."""
    body = document.element.body
    blocks: list[Paragraph | Table] = []
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            blocks.append(Paragraph(child, document))
        elif child.tag == qn("w:tbl"):
            blocks.append(Table(child, document))
    return blocks


def _heading_level(style: str) -> int | None:
    match = _HEADING_STYLE.match(style or "")
    if not match:
        return None
    return min(int(match.group(1)), _MAX_LEVEL) if match.group(1) else 1


def _table_markdown(table: Table) -> str:
    rows = [[normalize_spaces(cell.text) for cell in row.cells] for row in table.rows]
    rows = [row for row in rows if any(row)]
    if len(rows) < 2:
        return ""
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _core_title(document: DocxDocument) -> str:
    try:
        return normalize_spaces(document.core_properties.title or "")
    except Exception:
        return ""
