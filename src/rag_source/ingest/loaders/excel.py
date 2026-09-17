"""Chargeur Excel (openpyxl).

Deux formes de feuilles coexistent dans le corpus et sont traitées différemment :

- **feuille tabulaire** (ligne d'en-tête puis des lignes de données) : chaque ligne
  devient une section ``RECORD`` indivisible, du type « une recommandation ». Son
  identifiant (``R24``…) est conservé en métadonnée, ce qui permet plus tard de le
  retrouver par mot-clé exact autant que par le sens ;
- **feuille de prose** (préambule, mode d'emploi, cellules fusionnées) : les valeurs
  sont concaténées en sections de texte.

L'implémentation d'origine ignorait totalement les fichiers Excel, alors qu'ils
contiennent la donnée la plus directement interrogeable du corpus.
"""

from __future__ import annotations

import re
import warnings
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import openpyxl
from openpyxl.worksheet.worksheet import Worksheet

from rag_source.domain import DocumentFormat, MetadataValue, Section, SectionKind
from rag_source.ingest.loaders.base import Extracted, LoaderError, register_loader
from rag_source.ingest.text import normalize_spaces

_HEADER_SCAN_ROWS = 12
_MIN_HEADER_CELLS = 2
_MIN_DATA_ROWS = 2
_MIN_PROSE_CHARS = 40
_ID_HEADER = re.compile(r"identifiant|^id$|référence|réf\.|^n°", re.IGNORECASE)
_ID_VALUE = re.compile(r"^[A-Z]{1,3}[-_ ]?\d+\+?$")


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
                header_index = _find_header(rows)
                if header_index is None:
                    sections.extend(_prose_sections(sheet.title, rows))
                else:
                    sections.extend(_record_sections(sheet.title, rows, header_index))
        finally:
            workbook.close()

        if not sections:
            raise LoaderError("Aucune donnée exploitable dans le classeur.")
        return Extracted(title=path.stem, sections=tuple(sections), warnings=tuple(notes))


def _values(sheet: Worksheet) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in sheet.iter_rows(values_only=True):
        rows.append([_cell(value) for value in raw])
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


def _find_header(rows: list[list[str]]) -> int | None:
    """Indice de la ligne d'en-tête, ou ``None`` si la feuille n'est pas tabulaire.

    On retient la première ligne qui a au moins deux libellés courts et distincts et
    qui est suivie d'assez de lignes de données de même largeur.
    """
    best: tuple[int, int] | None = None
    for index, row in enumerate(rows[:_HEADER_SCAN_ROWS]):
        labels = [cell for cell in row if cell]
        if len(labels) < _MIN_HEADER_CELLS or len(set(labels)) != len(labels):
            continue
        if any(len(label) > 80 or "\n" in label for label in labels):
            continue  # un paragraphe en cellule n'est pas un en-tête
        filled = [i for i, cell in enumerate(row) if cell]
        data_rows = sum(
            1
            for following in rows[index + 1 :]
            if sum(1 for i in filled if i < len(following) and following[i]) >= _MIN_HEADER_CELLS
        )
        if data_rows >= _MIN_DATA_ROWS and (best is None or data_rows > best[1]):
            best = (index, data_rows)
    return None if best is None else best[0]


def _record_sections(sheet: str, rows: list[list[str]], header_index: int) -> list[Section]:
    header = [cell.strip() for cell in rows[header_index]]
    id_column = _id_column(header, rows[header_index + 1 :])
    sections: list[Section] = []
    for offset, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
        pairs = [
            (header[i] if i < len(header) and header[i] else f"Colonne {i + 1}", cell)
            for i, cell in enumerate(row)
            if cell
        ]
        if not pairs:
            continue
        record_id = row[id_column] if id_column is not None and id_column < len(row) else ""
        metadata: dict[str, MetadataValue] = {"sheet": sheet, "row": offset}
        if record_id:
            metadata["rec_id"] = record_id
        heading_path = (sheet, record_id) if record_id else (sheet,)
        text = "\n".join(f"{label} : {value}" for label, value in pairs)
        sections.append(
            Section(
                text=text,
                heading_path=heading_path,
                kind=SectionKind.RECORD,
                metadata=metadata,
            )
        )
    return sections


def _id_column(header: list[str], data: list[list[str]]) -> int | None:
    """Colonne d'identifiant : d'abord par son libellé, sinon par la forme des valeurs."""
    for index, label in enumerate(header):
        if label and _ID_HEADER.search(label):
            return index
    for index in range(len(header)):
        values = [row[index] for row in data if index < len(row) and row[index]]
        if len(values) >= _MIN_DATA_ROWS and all(_ID_VALUE.match(value) for value in values):
            return index
    return None


def _prose_sections(sheet: str, rows: list[list[str]]) -> list[Section]:
    text = "\n\n".join(
        "\n".join(cell for cell in row if cell).strip() for row in rows if any(row)
    ).strip()
    if len(text) < _MIN_PROSE_CHARS:
        return []
    return [Section(text=text, heading_path=(sheet,), kind=SectionKind.PROSE)]
