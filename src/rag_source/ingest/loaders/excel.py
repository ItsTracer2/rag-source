"""Chargeur Excel (openpyxl).

Chaque feuille est traitée selon sa forme : tableau de données ou prose en
cellules (cf. :mod:`rag_source.ingest.tabular`).
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import openpyxl
from openpyxl.worksheet.worksheet import Worksheet

from rag_source.domain import DocumentFormat, Section
from rag_source.ingest.loaders.base import Extracted, LoaderError, register_loader
from rag_source.ingest.tabular import find_header, prose_sections, record_sections
from rag_source.ingest.text import normalize_spaces


@register_loader
class ExcelLoader:
    extensions = frozenset({".xlsx", ".xlsm"})
    format = DocumentFormat.EXCEL

    def load(self, path: Path) -> Extracted:
        try:
            with warnings.catch_warnings():
                # openpyxl signale les extensions Excel non gérées (mise en forme
                # conditionnelle, validation) : sans effet sur les valeurs lues.
                warnings.simplefilter("ignore", UserWarning)
                workbook = openpyxl.load_workbook(path, data_only=True, read_only=False)
        except Exception as exc:
            raise LoaderError(f"Classeur illisible : {exc}") from exc

        sections: list[Section] = []
        notes: list[str] = []
        try:
            for sheet in workbook.worksheets:
                if sheet.sheet_state != "visible":
                    notes.append(f"feuille masquée ignorée : {sheet.title}")
                    continue
                rows = _values(sheet)
                if not rows:
                    continue
                header_index = find_header(rows)
                if header_index is None:
                    sections.extend(prose_sections(sheet.title, rows))
                else:
                    sections.extend(record_sections(sheet.title, rows, header_index))
        finally:
            workbook.close()

        if not sections:
            raise LoaderError("Aucune donnée exploitable dans le classeur.")
        return Extracted(title=path.stem, sections=tuple(sections), warnings=tuple(notes))


def _values(sheet: Worksheet) -> list[list[str]]:
    rows = [[_cell(value) for value in raw] for raw in sheet.iter_rows(values_only=True)]
    while rows and not any(rows[-1]):
        rows.pop()
    return rows


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat() if value.time() == time.min else value.isoformat(" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return normalize_spaces(str(value))
