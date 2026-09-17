from collections.abc import Callable
from pathlib import Path

import pytest

from rag_source.domain import SectionKind
from rag_source.ingest.loaders.base import LoaderError
from rag_source.ingest.loaders.excel import ExcelLoader

RECOMMANDATIONS = [
    ["Identifiant", "Intitulé de la recommandation", "Niveau de sécurité"],
    ["R01", "Cloisonner le SI d'administration", "1"],
    ["R02", "Journaliser les accès", "2"],
    ["R03", "Chiffrer les sauvegardes", "3"],
]


@pytest.fixture
def loader() -> ExcelLoader:
    return ExcelLoader()


def test_one_record_per_row(loader: ExcelLoader, make_xlsx: Callable[..., Path]) -> None:
    extracted = loader.load(make_xlsx({"Suivi des recommandations": RECOMMANDATIONS}))
    records = [s for s in extracted.sections if s.kind is SectionKind.RECORD]
    assert len(records) == 3
    first = records[0]
    assert "Identifiant : R01" in first.text
    assert "Intitulé de la recommandation : Cloisonner le SI d'administration" in first.text
    assert first.metadata["rec_id"] == "R01"
    assert first.metadata["sheet"] == "Suivi des recommandations"
    assert first.metadata["row"] == 2
    assert first.heading_path == ("Suivi des recommandations", "R01")


def test_empty_cells_are_skipped(loader: ExcelLoader, make_xlsx: Callable[..., Path]) -> None:
    rows = [
        ["Identifiant", "Intitulé", "Commentaire"],
        ["R01", "Faire ceci", None],
        ["R02", "Faire cela", "avec un commentaire"],
    ]
    extracted = loader.load(make_xlsx({"Feuille": rows}))
    assert "Commentaire" not in extracted.sections[0].text
    assert "Commentaire : avec un commentaire" in extracted.sections[1].text


def test_id_column_detected_by_value_shape(
    loader: ExcelLoader, make_xlsx: Callable[..., Path]
) -> None:
    """Sans libellé explicite, la colonne d'identifiants est reconnue à sa forme."""
    rows = [["Code", "Mesure"], ["R01", "Faire ceci"], ["R02", "Faire cela"]]
    extracted = loader.load(make_xlsx({"Feuille": rows}))
    assert extracted.sections[0].metadata["rec_id"] == "R01"


def test_non_tabular_sheet_becomes_prose(
    loader: ExcelLoader, make_xlsx: Callable[..., Path]
) -> None:
    preamble = [
        ["Description de la démarche : l'approche graduelle s'appuie sur trois niveaux."],
        ["Chaque entité identifie le niveau cible adapté à son exposition à la menace."],
    ]
    extracted = loader.load(make_xlsx({"Préambule": preamble}))
    assert [s.kind for s in extracted.sections] == [SectionKind.PROSE]
    assert "approche graduelle" in extracted.sections[0].text


def test_types_are_readable(loader: ExcelLoader, make_xlsx: Callable[..., Path]) -> None:
    """Un entier stocké en flottant ne doit pas s'afficher « 3.0 »."""
    rows = [["Identifiant", "Niveau"], ["R01", 3.0], ["R02", 2.0]]
    extracted = loader.load(make_xlsx({"Feuille": rows}))
    assert "Niveau : 3" in extracted.sections[0].text


def test_empty_workbook_is_an_error(loader: ExcelLoader, make_xlsx: Callable[..., Path]) -> None:
    with pytest.raises(LoaderError, match="Aucune donnée"):
        loader.load(make_xlsx({"Vide": [[None]]}))
