"""Chargeur PDF (PyMuPDF).

Le but est de restituer la *structure* du document, pas seulement son texte :

- les titres sont détectés par leur style (taille et graisse relatives au corps de
  texte) et forment le fil d'Ariane de chaque section ;
- les tableaux sont extraits en Markdown et gardés entiers ;
- les en-têtes, pieds de page et sommaires imprimés sont retirés.

Rien n'est propre à un type de document : les heuristiques portent sur la mise en
page (styles relatifs, répétitions, colonnes), pas sur le vocabulaire. Une notice
d'électroménager en anglais et un guide réglementaire en français passent par le
même chemin.

Les pages sans couche texte (documents scannés) sont reconnues par OCR quand
Tesseract est disponible ; sinon, elles sont signalées au lieu d'être ignorées en
silence.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import pymupdf

from rag_source.config import OcrMode, get_settings
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
_OCR_DPI = 300
# Détection de colonnes : il faut assez de lignes de part et d'autre de la
# gouttière, et peu de lignes qui la traversent.
_MIN_LINES_FOR_COLUMNS = 3
_MIN_LINES_PER_COLUMN = 2
_MAX_SPANNING_RATIO = 0.2  # part de lignes pleine largeur tolérée dans la gouttière
_MIN_GUTTER_RATIO = 0.03  # largeur minimale de la gouttière, en fraction de page
_SEARCH_FROM, _SEARCH_TO = 0.25, 0.75  # zone où chercher la gouttière
_GUTTER_PADDING = 1.0
_LINE_TOLERANCE = 2.0
_NUMBERING = re.compile(r"^(?:[IVXLC]+|[0-9]+(?:\.[0-9]+)*|[A-Z])[.)]\s")
_DOT_LEADER = re.compile(r"\.{5,}")
_DIGITS = re.compile(r"\d+")


@dataclass(frozen=True, slots=True)
class _Line:
    """Une ligne de texte et la position de chacun de ses mots."""

    words: list[tuple[float, float]]
    top: float
    bottom: float

    def covers(self, x: float, padding: float = _GUTTER_PADDING) -> bool:
        return any(x0 - padding <= x <= x1 + padding for x0, x1 in self.words)


@dataclass(slots=True)
class _Item:
    page: int
    y: float
    x: float
    x_end: float
    text: str
    kind: SectionKind
    size: float = 0.0
    bold: bool = False
    level: int | None = None  # renseigné si l'item est un titre
    column: int = 0  # index de colonne, pour l'ordre de lecture


@register_loader
class PdfLoader:
    extensions = frozenset({".pdf"})
    format = DocumentFormat.PDF

    def __init__(self, ocr_mode: OcrMode | None = None, ocr_languages: str | None = None) -> None:
        # ``None`` = suivre la configuration au moment du chargement. Les tests
        # peuvent instancier le chargeur avec des réglages explicites.
        self._ocr_mode = ocr_mode
        self._ocr_languages = ocr_languages

    def load(self, path: Path) -> Extracted:
        try:
            doc = pymupdf.open(path)
        except Exception as exc:  # pymupdf lève des exceptions variées
            raise LoaderError(f"PDF illisible : {exc}") from exc
        ocr_mode, ocr_languages = self._ocr_options()
        with doc:
            if doc.needs_pass:
                raise LoaderError("PDF chiffré : mot de passe requis.")
            items, warnings = _extract_items(doc, ocr_mode, ocr_languages)
            if not items:
                raise LoaderError(
                    "Aucun texte extrait : PDF scanné et OCR indisponible ou infructueux."
                )
            title = _title(doc, items, path)
            if _is_cover_page(items, _body_size(items)):
                items = [item for item in items if item.page > 1]
            _mark_headings(items)
            sections = _build_sections(items)
        if not sections:
            raise LoaderError("Aucune section exploitable après nettoyage.")
        return Extracted(title=title, sections=tuple(sections), warnings=tuple(warnings))

    def _ocr_options(self) -> tuple[OcrMode, str]:
        if self._ocr_mode is not None and self._ocr_languages is not None:
            return self._ocr_mode, self._ocr_languages
        settings = get_settings()
        return (
            self._ocr_mode or settings.ocr_mode,
            self._ocr_languages or settings.ocr_languages,
        )


def _extract_items(
    doc: pymupdf.Document, ocr_mode: OcrMode, ocr_languages: str
) -> tuple[list[_Item], list[str]]:
    items: list[_Item] = []
    warnings: list[str] = []
    empty_pages = 0
    ocr_pages = 0
    ocr_failure: str | None = None

    for page in doc:
        tables = _page_tables(page)
        page_items = [
            _Item(
                page=page.number + 1,
                y=bbox[1],
                x=bbox[0],
                x_end=bbox[2],
                text=text,
                kind=SectionKind.TABLE,
            )
            for bbox, text in tables
        ]
        blocks, gutter = _page_blocks(page, [bbox for bbox, _ in tables])
        needs_ocr = ocr_mode is OcrMode.FORCE or (ocr_mode is OcrMode.AUTO and not blocks)
        if needs_ocr:
            recognized, ocr_gutter, error = _ocr_blocks(page, ocr_languages)
            if error is not None:
                ocr_failure = ocr_failure or error
            elif recognized:
                blocks = recognized if ocr_mode is OcrMode.FORCE else blocks + recognized
                gutter = gutter or ocr_gutter
                ocr_pages += 1
        page_items.extend(blocks)
        if not page_items:
            empty_pages += 1
        if _is_toc_page(page_items):
            continue  # sommaire imprimé : redondant avec les titres eux-mêmes
        items.extend(_order_by_column(page_items, gutter))

    kept = _drop_running_heads(items, doc)
    if ocr_pages:
        warnings.append(
            f"{ocr_pages} page(s) reconnue(s) par OCR ({_usable_languages(ocr_languages)})."
        )
    if ocr_failure:
        warnings.append(f"OCR indisponible : {ocr_failure}")
    if empty_pages:
        warnings.append(f"{empty_pages} page(s) sans texte exploitable.")
    return kept, warnings


def _ocr_blocks(page: pymupdf.Page, languages: str) -> tuple[list[_Item], float | None, str | None]:
    """Reconnaît le texte d'une page image. Renvoie (blocs, erreur éventuelle).

    Une indisponibilité de Tesseract n'est jamais fatale : elle remonte comme
    avertissement, et le document est traité avec ce qui a pu être extrait.
    """
    usable = _usable_languages(languages)
    if not usable:
        return [], None, f"aucune des langues demandées n'est installée ({languages})"
    try:
        textpage = page.get_textpage_ocr(language=usable, dpi=_OCR_DPI, full=True)
    except Exception as exc:
        return [], None, str(exc).splitlines()[0]
    blocks, gutter = _page_blocks(page, [], textpage=textpage)
    return blocks, gutter, None


@cache
def _installed_languages() -> frozenset[str]:
    """Langues Tesseract réellement présentes sur la machine."""
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    lines = result.stdout.splitlines()[1:]  # la première ligne est un en-tête
    return frozenset(line.strip() for line in lines if line.strip())


def _usable_languages(requested: str) -> str:
    """Restreint les langues demandées à celles qui sont installées.

    Sans ce filtrage, Tesseract échoue sur toute la page dès qu'une seule langue
    manque, et écrit directement sur la sortie d'erreur — un message que le
    programme ne peut ni intercepter ni expliquer.
    """
    installed = _installed_languages()
    usable = [language for language in requested.split("+") if language in installed]
    return "+".join(usable)


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


def _page_blocks(
    page: pymupdf.Page,
    table_bboxes: list[tuple[float, ...]],
    textpage: pymupdf.TextPage | None = None,
) -> tuple[list[_Item], float | None]:
    """Blocs de texte d'une page, dans l'ordre de lecture.

    Quand la page est sur deux colonnes, PyMuPDF fusionne les lignes qui partagent
    une même ligne de base : un bloc contient alors « ligne de gauche + ligne de
    droite », et aucun tri ne peut plus les séparer. On extrait donc chaque colonne
    dans sa propre zone, et les éléments pleine largeur (titres, tableaux) restent
    des séparateurs entre deux bandes de colonnes.
    """
    lines = _word_lines(page, textpage)
    band = _column_gutter(lines, page.rect.width)
    if band is None:
        return _text_blocks(page, table_bboxes, textpage), None
    gutter = (band[0] + band[1]) / 2

    # Une ligne pleine largeur (un titre) a du texte *dans* la gouttière ; une ligne
    # à deux colonnes, elle, laisse la gouttière vide de part et d'autre.
    spanning_bands = [
        (line.top, line.bottom)
        for line in lines
        if any(x0 < band[1] and x1 > band[0] for x0, x1 in line.words)
    ]

    def on_spanning_line(item: _Item) -> bool:
        return any(
            top - _LINE_TOLERANCE <= item.y <= bottom + _LINE_TOLERANCE
            for top, bottom in spanning_bands
        )

    blocks = [item for item in _text_blocks(page, table_bboxes, textpage) if on_spanning_line(item)]
    for clip in (
        pymupdf.Rect(page.rect.x0, page.rect.y0, gutter, page.rect.height),
        pymupdf.Rect(gutter, page.rect.y0, page.rect.x1, page.rect.height),
    ):
        blocks.extend(
            item
            for item in _text_blocks(page, table_bboxes, textpage, clip=clip)
            if not on_spanning_line(item)
        )
    return blocks, gutter


def _word_lines(page: pymupdf.Page, textpage: pymupdf.TextPage | None) -> list[_Line]:
    """Lignes de la page, chacune avec la position de ses mots.

    On garde les mots un par un : c'est le seul grain qui laisse voir le blanc
    entre deux colonnes. Un bloc, et même une ligne réduite à son cadre, fusionnent
    déjà la colonne de gauche et celle de droite.
    """
    grouped: dict[int, list[tuple[float, float, float, float]]] = {}
    for word in page.get_text("words", textpage=textpage):
        x0, y0, x1, y1 = word[0], word[1], word[2], word[3]
        key = round((y0 + y1) / 2 / _LINE_TOLERANCE)
        grouped.setdefault(key, []).append((x0, x1, y0, y1))
    return [
        _Line(
            words=sorted((p[0], p[1]) for p in parts),
            top=min(p[2] for p in parts),
            bottom=max(p[3] for p in parts),
        )
        for parts in grouped.values()
    ]


def _text_blocks(
    page: pymupdf.Page,
    table_bboxes: list[tuple[float, ...]],
    textpage: pymupdf.TextPage | None = None,
    clip: pymupdf.Rect | None = None,
) -> list[_Item]:
    items: list[_Item] = []
    raw: dict[str, Any] = page.get_text("dict", textpage=textpage, clip=clip)
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
                x_end=bbox[2],
                text=text,
                kind=SectionKind.PROSE,
                size=round(max(s["size"] for s in spans), 1),
                bold=all(bool(s["flags"] & _BOLD_FLAG) for s in spans),
            )
        )
    return items


def _same_line(item: _Item, other: _Item) -> bool:
    return abs(item.y - other.y) <= _LINE_TOLERANCE


def _column_gutter(lines: list[_Line], width: float) -> tuple[float, float] | None:
    """Bande verticale séparant deux colonnes, si la page en a une.

    On mesure, pour chaque abscisse de la zone centrale, **combien de lignes** ont
    un mot à cet endroit. Une gouttière est une bande large que presque aucune ligne
    n'occupe. Raisonner en proportion de lignes, et non en simple présence de texte,
    permet de tolérer quelques titres pleine largeur : sinon un seul titre suffirait
    à masquer la séparation des colonnes.
    """
    if width <= 0 or len(lines) < _MIN_LINES_FOR_COLUMNS:
        return None

    tolerated = int(len(lines) * _MAX_SPANNING_RATIO)
    step = max(width / 400, 0.5)
    low, high = width * _SEARCH_FROM, width * _SEARCH_TO
    best: tuple[float, float] | None = None
    start: float | None = None
    x = low
    while x <= high:
        if sum(1 for line in lines if line.covers(x)) <= tolerated:
            start = x if start is None else start
        elif start is not None:
            if best is None or x - start > best[1] - best[0]:
                best = (start, x)
            start = None
        x += step
    if start is not None and (best is None or high - start > best[1] - best[0]):
        best = (start, high)

    if best is None or best[1] - best[0] < width * _MIN_GUTTER_RATIO:
        return None
    # Compter les lignes ayant des mots de part et d'autre : les lignes sont
    # fusionnées par PyMuPDF, aucune n'est « entièrement à gauche ».
    middle = (best[0] + best[1]) / 2
    left = sum(1 for line in lines if any(x1 <= middle for _, x1 in line.words))
    right = sum(1 for line in lines if any(x0 >= middle for x0, _ in line.words))
    if min(left, right) < _MIN_LINES_PER_COLUMN:
        return None
    return best


def _order_by_column(items: list[_Item], gutter: float | None) -> list[_Item]:
    """Ordonne les éléments d'une page selon l'ordre de lecture humain.

    Sans colonnes, l'ordre est simplement vertical. Avec colonnes, chaque élément
    reçoit son index de colonne et les colonnes sont lues l'une après l'autre.
    Les éléments qui traversent la gouttière (titre pleine largeur, tableau)
    restent des séparateurs : ils gardent leur place dans le flux vertical et
    referment la bande de colonnes précédente.
    """
    if gutter is None:
        return sorted(items, key=lambda i: (round(i.y, 1), i.x))

    ordered: list[_Item] = []
    band: list[_Item] = []  # blocs d'une même bande à deux colonnes
    for item in sorted(items, key=lambda i: (round(i.y, 1), i.x)):
        if item.x < gutter < item.x_end:  # bloc pleine largeur : sépare deux bandes
            ordered.extend(_flush_band(band, gutter))
            band = []
            ordered.append(item)
        else:
            band.append(item)
    ordered.extend(_flush_band(band, gutter))
    return ordered


def _flush_band(band: list[_Item], gutter: float) -> list[_Item]:
    for item in band:
        item.column = 0 if item.x_end <= gutter else 1
    return sorted(band, key=lambda i: (i.column, round(i.y, 1), i.x))


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
    for item in items:
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
