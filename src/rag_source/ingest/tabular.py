"""Traitement commun des données tabulaires (Excel, CSV, TSV).

Une feuille ou un fichier tabulaire peut prendre deux formes :

- **tableau de données** : une ligne d'en-tête puis des lignes homogènes. Chaque
  ligne devient une section indivisible (``RECORD``) : une recommandation, une
  référence de pièce détachée, un code d'erreur, une ligne d'inventaire ;
- **prose mise en cellules** : préambule, mode d'emploi, notes. Les valeurs sont
  alors concaténées en texte.

Rien ici n'est propre à un domaine : la détection repose sur la forme du tableau,
pas sur le sens des libellés. Les libellés connus (plusieurs langues) ne servent
qu'à repérer la colonne d'identifiant, et une détection par la forme des valeurs
prend le relais quand ils sont absents.
"""

from __future__ import annotations

import re

from rag_source.domain import MetadataValue, Section, SectionKind

HEADER_SCAN_ROWS = 12
MIN_HEADER_CELLS = 2
MIN_DATA_ROWS = 2
MIN_PROSE_CHARS = 40
_MAX_HEADER_LABEL = 80

# Libellés fréquents de colonne d'identifiant, en français et en anglais.
_ID_HEADER = re.compile(
    r"^(?:identifiant|référence|reference|réf\.?|ref\.?|id|code|item|clé|key|"
    r"n°|no\.?|num(?:éro|ber)?|part\s*(?:number|no)|sku)$",
    re.IGNORECASE,
)
# Identifiants courts et réguliers : R24, A-12, ERR_03, 4.2.1…
_ID_VALUE = re.compile(r"^[A-Za-z]{0,4}[-_. ]?\d+(?:[-_.]\d+)*\+?$")


def find_header(rows: list[list[str]]) -> int | None:
    """Indice de la ligne d'en-tête, ou ``None`` si la source n'est pas tabulaire.

    On retient la ligne qui a au moins deux libellés courts et distincts et qui est
    suivie du plus grand nombre de lignes de données remplissant les mêmes colonnes.
    """
    best: tuple[int, int] | None = None
    for index, row in enumerate(rows[:HEADER_SCAN_ROWS]):
        labels = [cell for cell in row if cell]
        if len(labels) < MIN_HEADER_CELLS or len(set(labels)) != len(labels):
            continue
        if any(len(label) > _MAX_HEADER_LABEL or "\n" in label for label in labels):
            continue  # un paragraphe en cellule n'est pas un en-tête
        filled = [i for i, cell in enumerate(row) if cell]
        data_rows = sum(
            1
            for following in rows[index + 1 :]
            if sum(1 for i in filled if i < len(following) and following[i]) >= MIN_HEADER_CELLS
        )
        if data_rows >= MIN_DATA_ROWS and (best is None or data_rows > best[1]):
            best = (index, data_rows)
    return None if best is None else best[0]


def id_column(header: list[str], data: list[list[str]]) -> int | None:
    """Colonne d'identifiant : d'abord par son libellé, sinon par la forme des valeurs."""
    for index, label in enumerate(header):
        if label and _ID_HEADER.match(label.strip()):
            return index
    for index in range(len(header)):
        values = [row[index] for row in data if index < len(row) and row[index]]
        if (
            len(values) >= MIN_DATA_ROWS
            and len(set(values)) == len(values)  # un identifiant est unique
            and all(_ID_VALUE.match(value) for value in values)
        ):
            return index
    return None


def record_sections(scope: str, rows: list[list[str]], header_index: int) -> list[Section]:
    """Transforme les lignes de données en sections indivisibles."""
    header = [cell.strip() for cell in rows[header_index]]
    column = id_column(header, rows[header_index + 1 :])
    sections: list[Section] = []
    for offset, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
        pairs = [
            (header[i] if i < len(header) and header[i] else f"Colonne {i + 1}", cell)
            for i, cell in enumerate(row)
            if cell
        ]
        if not pairs:
            continue
        record_id = row[column] if column is not None and column < len(row) else ""
        metadata: dict[str, MetadataValue] = {"row": offset}
        if scope:
            metadata["sheet"] = scope
        if record_id:
            metadata["rec_id"] = record_id
        heading = tuple(part for part in (scope, record_id) if part)
        sections.append(
            Section(
                text="\n".join(f"{label} : {value}" for label, value in pairs),
                heading_path=heading,
                kind=SectionKind.RECORD,
                metadata=metadata,
            )
        )
    return sections


def prose_sections(scope: str, rows: list[list[str]]) -> list[Section]:
    """Feuille non tabulaire : les cellules non vides deviennent du texte."""
    text = "\n\n".join(
        "\n".join(cell for cell in row if cell).strip() for row in rows if any(row)
    ).strip()
    if len(text) < MIN_PROSE_CHARS:
        return []
    heading = (scope,) if scope else ()
    return [Section(text=text, heading_path=heading, kind=SectionKind.PROSE)]
