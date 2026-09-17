"""Chargeur PDF (PyMuPDF).

Le but est de restituer la *structure* du document, pas seulement son texte :

- les titres sont détectés par leur style (taille et graisse relatives au corps de
  texte) et forment le fil d'Ariane de chaque section ;
- les tableaux sont extraits en Markdown et gardés entiers ;
- les en-têtes, pieds de page et sommaires imprimés sont retirés.

Le corpus de référence (documents ANSSI) n'a aucun signet PDF : la détection par
style est donc le mécanisme principal. Quand le document en a un, le sommaire
intégré est utilisé en priorité, car il est plus fiable.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from rag_source.domain import DocumentFormat, Section, SectionKind
from rag_source.ingest.loaders.base import Extracted, LoaderError, register_loader
from rag_source.ingest.text import is_bullet, join_wrapped_lines, normalize_spaces

_BOLD_FLAG = 1 << 4
# Bandes haute et basse où se trouvent en-têtes et pieds de page, en fraction de
# la hauteur de page. Un élément n'y est retiré que s'il se répète (cf. _repeated).
_TOP_BAND = 0.08
_BOTTOM_BAND = 0.88
_MIN_REPEATS = 3
_HEADING_MAX_CHARS = 200
_MIN_TITLE_CHARS = 15
_COVER_MAX_CHARS = 600
_NUMBERING = re.compile(r"^(?:[IVXLC]+|[0-9]+(?:\.[0-9]+)*|[A-Z])[.)]\s")
_DOT_LEADER = re.compile(r"\.{5,}")
_DIGITS = re.compile(r"\d+")


@dataclass(slots=True)
class _Item:
    page: int
    y: float
    x: float
    text: str
    kind: SectionKind
    size: float = 0.0
    bold: bool = False
    level: int | None = None  # renseigné si l'item est un titre


@register_loader
class PdfLoader:
    extensions = frozenset({".pdf"})
    format = DocumentFormat.PDF

    def load(self, path: Path) -> Extracted:
        try:
            doc = pymupdf.open(path)
        except Exception as exc:  # pymupdf lève des exceptions variées
            raise LoaderError(f"PDF illisible : {exc}") from exc
        with doc:
            if doc.needs_pass:
                raise LoaderError("PDF chiffré : mot de passe requis.")
            items, warnings = _extract_items(doc)
            if not items:
                raise LoaderError(
                    "Aucun texte extrait : PDF probablement scanné (OCR non supporté)."
                )
            title = _title(doc, items, path)
            if _is_cover_page(items, _body_size(items)):
                items = [item for item in items if item.page > 1]
            _mark_headings(items)
            sections = _build_sections(items)
        if not sections:
            raise LoaderError("Aucune section exploitable après nettoyage.")
        return Extracted(title=title, sections=tuple(sections), warnings=tuple(warnings))


def _extract_items(doc: pymupdf.Document) -> tuple[list[_Item], list[str]]:
    items: list[_Item] = []
    empty_pages = 0
    for page in doc:
        tables = _page_tables(page)
        page_items = [
            _Item(page=page.number + 1, y=bbox[1], x=bbox[0], text=text, kind=SectionKind.TABLE)
            for bbox, text in tables
        ]
        blocks = _text_blocks(page, [bbox for bbox, _ in tables])
        page_items.extend(blocks)
        if not blocks:
            empty_pages += 1
        if _is_toc_page(page_items):
            continue  # sommaire imprimé : redondant avec les titres eux-mêmes
        items.extend(page_items)

    kept = _drop_running_heads(items, doc)
    warnings: list[str] = []
    if empty_pages:
        warnings.append(f"{empty_pages} page(s) sans texte (illustrations ou scan).")
    return kept, warnings


def _page_tables(page: pymupdf.Page) -> list[tuple[tuple[float, ...], str]]:
    tables: list[tuple[tuple[float, ...], str]] = []
    try:
        found = page.find_tables().tables
    except Exception:  # la détection de tableaux est heuristique et peut échouer
        return tables
    for table in found:
        cells = [c for row in table.extract() for c in row if c and str(c).strip()]
        if len(cells) < 2:
            continue  # faux positif : cadre ou bordure sans contenu
        markdown = table.to_markdown().strip()
        if markdown:
            tables.append((tuple(table.bbox), markdown))
    return tables


def _text_blocks(page: pymupdf.Page, table_bboxes: list[tuple[float, ...]]) -> list[_Item]:
    items: list[_Item] = []
    raw: dict[str, Any] = page.get_text("dict")
    for block in raw["blocks"]:
        if "lines" not in block:
            continue  # bloc image
        bbox = block["bbox"]
        if any(_inside(bbox, table_bbox) for table_bbox in table_bboxes):
            continue  # déjà restitué par le tableau
        lines = ["".join(span["text"] for span in line["spans"]) for line in block["lines"]]
        text = join_wrapped_lines(lines)
        if not text:
            continue
        spans = [s for line in block["lines"] for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        items.append(
            _Item(
                page=page.number + 1,
                y=bbox[1],
                x=bbox[0],
                text=text,
                kind=SectionKind.PROSE,
                size=round(max(s["size"] for s in spans), 1),
                bold=all(bool(s["flags"] & _BOLD_FLAG) for s in spans),
            )
        )
    return items


def _inside(inner: tuple[float, ...], outer: tuple[float, ...]) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _is_toc_page(items: list[_Item]) -> bool:
    """Page de sommaire : lignes à points de conduite (« Introduction ...... 4 »)."""
    if len(items) < 3:
        return False
    leaders = sum(1 for item in items if _DOT_LEADER.search(item.text))
    return leaders >= max(3, len(items) // 2)


def _drop_running_heads(items: list[_Item], doc: pymupdf.Document) -> list[_Item]:
    """Retire les éléments de marge qui se répètent d'une page à l'autre.

    Les numéros de page changent à chaque page : ils sont neutralisés avant
    comparaison. Un élément de marge n'est retiré que s'il apparaît sur au moins
    trois pages (ou la moitié du document), ce qui évite de supprimer du contenu
    qui se trouverait simplement en bas de page.
    """
    if not items:
        return items
    heights = {page.number + 1: page.rect.height or 1.0 for page in doc}
    threshold = max(_MIN_REPEATS, doc.page_count // 2)
    margin_keys: Counter[str] = Counter()
    keys: dict[int, str] = {}
    for index, item in enumerate(items):
        height = heights.get(item.page, 1.0)
        if item.y <= _TOP_BAND * height or item.y >= _BOTTOM_BAND * height:
            key = _DIGITS.sub("#", item.text)
            keys[index] = key
            margin_keys[key] += 1
    return [
        item
        for index, item in enumerate(items)
        if margin_keys.get(keys.get(index, ""), 0) < threshold
    ]


def _mark_headings(items: list[_Item]) -> None:
    """Attribue un niveau de titre aux items dont le style se détache du corps."""
    body_size = _body_size(items)
    candidates: list[_Item] = []
    for item in items:
        if item.kind is not SectionKind.PROSE or len(item.text) > _HEADING_MAX_CHARS:
            continue
        if "\n" in item.text or is_bullet(item.text):
            continue
        bigger = item.size >= body_size + 1
        # Un titre à la taille du corps doit être en gras *et* numéroté : le gras
        # seul sert aussi à mettre en valeur un début de paragraphe.
        emphasized = (
            item.bold
            and item.size >= body_size
            and len(item.text) <= 120
            and bool(_NUMBERING.match(item.text))
        )
        if bigger or emphasized:
            candidates.append(item)

    sizes = sorted({item.size for item in candidates if item.size > body_size}, reverse=True)
    for item in candidates:
        if item.size in sizes:
            item.level = sizes.index(item.size) + 1
        else:
            item.level = len(sizes) + 1  # titre en gras à la taille du corps


def _is_cover_page(items: list[_Item], body_size: float) -> bool:
    """La première page est-elle une page de garde plutôt que du contenu ?

    Critère : peu de texte, et aucun paragraphe à la taille du corps. Une page de
    garde n'aligne que des mentions en grands caractères (éditeur, titre, version).
    Ses fragments ne sont donc pas des titres de section : sans ce test, un titre
    de couverture réparti sur trois lignes crée trois faux niveaux de titre.
    """
    first = [item for item in items if item.page == 1 and item.kind is SectionKind.PROSE]
    if not first:
        return False
    volume = sum(len(item.text) for item in first)
    return volume < _COVER_MAX_CHARS and all(item.size > body_size for item in first)


def _body_size(items: list[_Item]) -> float:
    """Taille de police du corps de texte : la plus utilisée, pondérée par le volume."""
    weights: Counter[float] = Counter()
    for item in items:
        if item.kind is SectionKind.PROSE:
            weights[item.size] += len(item.text)
    return weights.most_common(1)[0][0] if weights else 0.0


def _build_sections(items: list[_Item]) -> list[Section]:
    sections: list[Section] = []
    heading_path: list[str] = []
    buffer: list[_Item] = []

    def flush() -> None:
        if not buffer:
            return
        text = ""
        for item in buffer:
            if not text:
                text = item.text
            else:
                separator = "\n" if is_bullet(item.text) else "\n\n"
                text = f"{text}{separator}{item.text}"
        sections.append(
            Section(
                text=text,
                heading_path=tuple(heading_path),
                kind=SectionKind.PROSE,
                page_start=buffer[0].page,
                page_end=buffer[-1].page,
            )
        )
        buffer.clear()

    previous_level: int | None = None
    for item in sorted(items, key=lambda i: (i.page, round(i.y, 1), i.x)):
        if item.level is not None:
            flush()
            heading = normalize_spaces(item.text)
            if item.level == previous_level and heading_path:
                # Deux titres de même niveau qui se suivent sans contenu entre eux :
                # c'est un seul titre que la mise en page a réparti sur deux lignes.
                heading_path[-1] = f"{heading_path[-1]} {heading}"
            else:
                del heading_path[item.level - 1 :]
                heading_path.append(heading)
            previous_level = item.level
            continue
        previous_level = None
        if item.kind is SectionKind.TABLE:
            flush()
            sections.append(
                Section(
                    text=item.text,
                    heading_path=tuple(heading_path),
                    kind=SectionKind.TABLE,
                    page_start=item.page,
                    page_end=item.page,
                )
            )
        else:
            buffer.append(item)
    flush()
    return sections


def _title(doc: pymupdf.Document, items: list[_Item], path: Path) -> str:
    """Titre du document : métadonnées, sinon page de garde, sinon nom de fichier.

    Sur une page de garde, plusieurs blocs partagent la plus grande police (nom de
    l'éditeur, titre, sous-titre). Le plus long est le titre : « Premier ministre »
    perd contre « Critères d'évaluation de la conformité au règlement eIDAS ».
    """
    metadata_title = (doc.metadata or {}).get("title") or ""
    if metadata_title.strip():
        return normalize_spaces(metadata_title)
    cover = [item for item in items if item.page == 1 and item.kind is SectionKind.PROSE]
    if cover:
        largest = max(item.size for item in cover)
        candidates = [
            item for item in cover if item.size == largest and len(item.text) >= _MIN_TITLE_CHARS
        ]
        if candidates:
            return normalize_spaces(max(candidates, key=lambda item: len(item.text)).text)
    return path.stem.replace("_", " ").replace("-", " ")
