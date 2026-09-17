"""Chargeur CSV / TSV.

Le séparateur et l'encodage sont déduits du fichier : un export métier n'est
presque jamais un CSV virgule/UTF-8 strict. Le contenu est ensuite traité comme
n'importe quelle source tabulaire.
"""

from __future__ import annotations

import csv
from pathlib import Path

from rag_source.domain import DocumentFormat
from rag_source.ingest.loaders.base import Extracted, LoaderError, register_loader
from rag_source.ingest.tabular import find_header, prose_sections, record_sections
from rag_source.ingest.text import normalize_spaces

_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
_DELIMITERS = ",;\t|"
_SNIFF_BYTES = 8192
_MAX_FIELD = 1_000_000


@register_loader
class DelimitedLoader:
    extensions = frozenset({".csv", ".tsv"})
    format = DocumentFormat.DELIMITED

    def load(self, path: Path) -> Extracted:
        raw, encoding = _read_text(path)
        if not raw.strip():
            raise LoaderError("Fichier délimité vide.")
        delimiter = _sniff(raw, path.suffix.lower())

        csv.field_size_limit(_MAX_FIELD)
        rows = [
            [normalize_spaces(cell) for cell in row]
            for row in csv.reader(raw.splitlines(), delimiter=delimiter)
        ]
        while rows and not any(rows[-1]):
            rows.pop()
        if not rows:
            raise LoaderError("Fichier délimité vide.")

        header_index = find_header(rows)
        sections = (
            prose_sections("", rows)
            if header_index is None
            else record_sections("", rows, header_index)
        )
        if not sections:
            raise LoaderError("Aucune donnée exploitable.")
        notes = [] if encoding.startswith("utf-8") else [f"encodage détecté : {encoding}"]
        return Extracted(title=path.stem, sections=tuple(sections), warnings=tuple(notes))


def _read_text(path: Path) -> tuple[str, str]:
    for encoding in _ENCODINGS:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    raise LoaderError("Encodage non reconnu.")


def _sniff(raw: str, suffix: str) -> str:
    try:
        return csv.Sniffer().sniff(raw[:_SNIFF_BYTES], delimiters=_DELIMITERS).delimiter
    except csv.Error:
        # Sniffer échoue sur les fichiers à une seule colonne : le suffixe tranche.
        return "\t" if suffix == ".tsv" else ","
