"""Fabriques de fichiers d'exemple.

Les tests ne dépendent jamais du corpus réel : ils fabriquent des fichiers
minimaux mais représentatifs (titres de tailles différentes, en-tête répété,
tableau, feuille de recommandations).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import openpyxl
import pymupdf
import pytest


@pytest.fixture
def make_pdf(tmp_path: Path) -> Callable[..., Path]:
    def _make(
        pages: Sequence[Sequence[tuple[str, float, bool]]],
        name: str = "doc.pdf",
        running_head: str | None = None,
    ) -> Path:
        """Crée un PDF où chaque bloc est (texte, taille de police, gras)."""
        doc = pymupdf.open()
        for page_number, blocks in enumerate(pages, start=1):
            page = doc.new_page()
            y = 80.0
            for text, size, bold in blocks:
                font = "Times-Bold" if bold else "Times-Roman"
                page.insert_text((72, y), text, fontsize=size, fontname=font)
                y += size * 2.5
            if running_head is not None:
                page.insert_text(
                    (72, page.rect.height - 30),
                    f"{running_head} page {page_number}",
                    fontsize=8,
                    fontname="Times-Roman",
                )
        path = tmp_path / name
        doc.save(path)
        doc.close()
        return path

    return _make


@pytest.fixture
def make_xlsx(tmp_path: Path) -> Callable[..., Path]:
    def _make(sheets: dict[str, Sequence[Sequence[object]]], name: str = "classeur.xlsx") -> Path:
        workbook = openpyxl.Workbook()
        workbook.remove(workbook.active)  # type: ignore[arg-type]
        for title, rows in sheets.items():
            sheet = workbook.create_sheet(title)
            for row in rows:
                sheet.append(list(row))
        path = tmp_path / name
        workbook.save(path)
        workbook.close()
        return path

    return _make


@pytest.fixture
def make_md(tmp_path: Path) -> Callable[..., Path]:
    def _make(content: str, name: str = "doc.md") -> Path:
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return path

    return _make
