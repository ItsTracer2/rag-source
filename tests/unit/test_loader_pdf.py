from collections.abc import Callable
from pathlib import Path

import pytest

from rag_source.domain import SectionKind
from rag_source.ingest.loaders.base import LoaderError
from rag_source.ingest.loaders.pdf import PdfLoader

BODY = 11.0
H1 = 16.0
H2 = 13.0
LOREM = (
    "Le prestataire doit mettre en oeuvre une analyse de risques sur le systeme "
    "d'information utilise pour fournir le service de confiance qualifie."
)


@pytest.fixture
def loader() -> PdfLoader:
    return PdfLoader()


def test_headings_build_the_breadcrumb(loader: PdfLoader, make_pdf: Callable[..., Path]) -> None:
    path = make_pdf(
        [
            [
                ("I. Introduction", H1, True),
                (LOREM, BODY, False),
                ("I.1. Perimetre", H2, True),
                (LOREM, BODY, False),
            ],
            [("II. Exigences", H1, True), (LOREM, BODY, False)],
        ]
    )
    extracted = loader.load(path)
    paths = [s.heading_path for s in extracted.sections]
    assert ("I. Introduction",) in paths
    assert ("I. Introduction", "I.1. Perimetre") in paths
    assert ("II. Exigences",) in paths  # le niveau 1 remplace la branche precedente


def test_pages_are_tracked(loader: PdfLoader, make_pdf: Callable[..., Path]) -> None:
    path = make_pdf(
        [
            [("I. Titre", H1, True), (LOREM, BODY, False)],
            [("II. Suite", H1, True), (LOREM, BODY, False)],
        ]
    )
    extracted = loader.load(path)
    assert {s.page_start for s in extracted.sections} == {1, 2}


def test_section_spans_the_page_break(loader: PdfLoader, make_pdf: Callable[..., Path]) -> None:
    """Un paragraphe coupé par un saut de page reste une seule section.

    L'implémentation d'origine découpait page par page : une exigence à cheval sur
    deux pages était scindée en deux, et le contexte perdu.
    """
    path = make_pdf([[("I. Titre", H1, True), (LOREM, BODY, False)], [(LOREM, BODY, False)]])
    section = next(s for s in loader.load(path).sections if s.heading_path == ("I. Titre",))
    assert (section.page_start, section.page_end) == (1, 2)


def test_running_head_is_removed(loader: PdfLoader, make_pdf: Callable[..., Path]) -> None:
    pages = [[("I. Titre", H1, True), (LOREM, BODY, False)] for _ in range(4)]
    path = make_pdf(pages, running_head="Guide ANSSI - diffusion publique")
    extracted = loader.load(path)
    assert all("diffusion publique" not in s.text for s in extracted.sections)


def test_footer_kept_when_not_repeated(loader: PdfLoader, make_pdf: Callable[..., Path]) -> None:
    """Un texte en bas de page qui ne se repete pas est du contenu, pas un pied de page."""
    pages = [[("I. Titre", H1, True), (LOREM, BODY, False)] for _ in range(4)]
    path = make_pdf(pages)
    unique = make_pdf(
        [
            [
                ("I. Titre", H1, True),
                (LOREM, BODY, False),
                ("Mention unique de bas de page", BODY, False),
            ]
        ],
        name="unique.pdf",
    )
    assert loader.load(path).sections
    extracted = loader.load(unique)
    assert any("Mention unique" in s.text for s in extracted.sections)


def test_bullets_keep_their_own_line(loader: PdfLoader, make_pdf: Callable[..., Path]) -> None:
    path = make_pdf(
        [
            [
                ("I. Titre", H1, True),
                ("Les modifications comprennent :", BODY, False),
                ("- les changements de sous-traitants ;", BODY, False),
                ("- les modifications d'architecture technique.", BODY, False),
            ]
        ]
    )
    extracted = loader.load(path)
    section = next(s for s in extracted.sections if "modifications comprennent" in s.text)
    lines = section.text.splitlines()
    assert "- les changements de sous-traitants ;" in lines
    assert "- les modifications d'architecture technique." in lines


def test_toc_page_is_dropped(loader: PdfLoader, make_pdf: Callable[..., Path]) -> None:
    toc = [
        ("SOMMAIRE", BODY, True),
        ("I. Introduction ................................ 3", BODY, False),
        ("II. Exigences .................................. 5", BODY, False),
        ("III. Annexes ................................... 9", BODY, False),
    ]
    path = make_pdf([toc, [("I. Introduction", H1, True), (LOREM, BODY, False)]])
    extracted = loader.load(path)
    assert all("........" not in s.text for s in extracted.sections)


def test_title_is_the_longest_cover_text(loader: PdfLoader, make_pdf: Callable[..., Path]) -> None:
    cover = [
        ("Premier ministre", H1, True),
        ("Criteres d'evaluation de la conformite au reglement eIDAS", H1, True),
        ("Version 1.3 du 11 avril 2025", BODY, False),
    ]
    extracted = loader.load(make_pdf([cover, [("I. Titre", H2, True), (LOREM, BODY, False)]]))
    assert extracted.title == "Criteres d'evaluation de la conformite au reglement eIDAS"


def test_empty_pdf_is_an_error(loader: PdfLoader, tmp_path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page()
    path = tmp_path / "vide.pdf"
    doc.save(path)
    doc.close()
    with pytest.raises(LoaderError, match="scanné"):
        loader.load(path)


def test_tables_are_kept_whole(loader: PdfLoader, tmp_path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "I. Exigences", fontsize=H1, fontname="Times-Bold")
    # Un tableau dessine : lignes + texte dans les cellules.
    top, left, row_height, col_width = 140, 72, 30, 160
    for r in range(3):
        for c in range(2):
            rect = pymupdf.Rect(
                left + c * col_width,
                top + r * row_height,
                left + (c + 1) * col_width,
                top + (r + 1) * row_height,
            )
            page.draw_rect(rect)
            page.insert_text(
                (rect.x0 + 4, rect.y0 + 18),
                f"cellule {r}{c}",
                fontsize=BODY,
                fontname="Times-Roman",
            )
    path = tmp_path / "tableau.pdf"
    doc.save(path)
    doc.close()

    extracted = loader.load(path)
    tables = [s for s in extracted.sections if s.kind is SectionKind.TABLE]
    assert len(tables) == 1
    assert "cellule 00" in tables[0].text and "cellule 21" in tables[0].text
    # Le texte du tableau n'est pas restitue une seconde fois comme prose.
    assert all(
        "cellule 00" not in s.text for s in extracted.sections if s.kind is SectionKind.PROSE
    )


def test_cover_page_is_dropped(loader: PdfLoader, make_pdf: Callable[..., Path]) -> None:
    """Une page de garde ne doit pas polluer le fil d'Ariane avec ses fragments."""
    cover = [
        ("Premier ministre", H1, True),
        ("Services de conservation qualifies des signatures", H1, True),
        ("electroniques qualifiees", H1, True),
    ]
    path = make_pdf([cover, [("I. Introduction", H2, True), (LOREM, BODY, False)]])
    extracted = loader.load(path)
    assert all(s.heading_path in {(), ("I. Introduction",)} for s in extracted.sections), [
        s.heading_path for s in extracted.sections
    ]


def test_wrapped_heading_is_reassembled(loader: PdfLoader, make_pdf: Callable[..., Path]) -> None:
    """Un titre sur deux lignes ne doit pas creer deux niveaux concurrents."""
    path = make_pdf(
        [
            [
                ("II. EXIGENCES RELATIVES AUX PRESTATAIRES", H2, True),
                ("QUALIFIES", H2, True),
                (LOREM, BODY, False),
            ]
        ]
    )
    section = loader.load(path).sections[0]
    assert section.heading_path == ("II. EXIGENCES RELATIVES AUX PRESTATAIRES QUALIFIES",)
