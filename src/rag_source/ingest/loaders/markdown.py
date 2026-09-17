"""Chargeur Markdown.

Le découpage suit les titres : chaque section porte son fil d'Ariane complet.
Les tableaux deviennent des sections à part, gardées entières.

L'analyse passe par ``markdown-it-py`` plutôt que par des expressions régulières :
un ``#`` dans un bloc de code ou dans un texte cité n'est pas un titre, et une
regex se tromperait.
"""

from __future__ import annotations

from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from rag_source.domain import DocumentFormat, Section, SectionKind
from rag_source.ingest.loaders.base import Extracted, LoaderError, register_loader

_MAX_HEADING_LEVEL = 6


@register_loader
class MarkdownLoader:
    extensions = frozenset({".md", ".markdown"})
    format = DocumentFormat.MARKDOWN

    def load(self, path: Path) -> Extracted:
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise LoaderError(f"Fichier non décodable en UTF-8 : {exc}") from exc
        lines = raw.splitlines()
        parser = MarkdownIt("commonmark").enable("table")
        tokens = parser.parse(raw)

        sections: list[Section] = []
        heading_path: list[str] = []
        buffer: list[str] = []
        title = ""

        def flush() -> None:
            text = "\n\n".join(block for block in buffer if block.strip())
            buffer.clear()
            if text.strip():
                sections.append(
                    Section(
                        text=text.strip(),
                        heading_path=tuple(heading_path),
                        kind=SectionKind.PROSE,
                    )
                )

        for index, token in enumerate(tokens):
            if token.type == "heading_open":
                flush()
                level = min(int(token.tag[1:]), _MAX_HEADING_LEVEL)
                heading = _inline_text(tokens[index + 1])
                del heading_path[level - 1 :]
                heading_path.append(heading)
                if not title and level == 1:
                    title = heading
            elif token.type == "table_open":
                flush()
                table = _source(lines, token)
                if table:
                    sections.append(
                        Section(
                            text=table,
                            heading_path=tuple(heading_path),
                            kind=SectionKind.TABLE,
                        )
                    )
            elif token.type in {"paragraph_open", "bullet_list_open", "ordered_list_open"}:
                if token.level == 0 and (block := _source(lines, token)):
                    buffer.append(block)
            elif token.type in {"fence", "code_block", "blockquote_open"} and token.level == 0:
                if block := _source(lines, token):
                    buffer.append(block)
        flush()

        if not sections:
            raise LoaderError("Fichier Markdown vide.")
        return Extracted(title=title or path.stem, sections=tuple(sections))


def _inline_text(token: Token) -> str:
    """Texte d'un titre, débarrassé de son balisage (``**Titre**`` → ``Titre``)."""
    if token.children is None:
        return token.content.strip()
    parts = [child.content for child in token.children if child.type in {"text", "code_inline"}]
    return "".join(parts).strip()


def _source(lines: list[str], token: Token) -> str:
    """Texte source d'un bloc, tel qu'écrit dans le fichier (balisage conservé)."""
    if token.map is None:
        return ""
    start, end = token.map
    return "\n".join(lines[start:end]).strip()
