from collections.abc import Callable
from pathlib import Path

import pytest

from rag_source.domain import SectionKind
from rag_source.ingest.loaders.base import LoaderError
from rag_source.ingest.loaders.markdown import MarkdownLoader

DOC = """# **Titre du document**

Introduction du document.

## 1\\. Première partie

Un paragraphe.

- puce un
- puce deux

### 1.1. Sous-partie

| Colonne A | Colonne B |
| --- | --- |
| a1 | b1 |

Texte après le tableau.

## 2\\. Deuxième partie

Fin.
"""


@pytest.fixture
def loader() -> MarkdownLoader:
    return MarkdownLoader()


def test_heading_path_and_title(loader: MarkdownLoader, make_md: Callable[..., Path]) -> None:
    extracted = loader.load(make_md(DOC))
    assert extracted.title == "Titre du document"  # le balisage gras est retiré
    paths = [section.heading_path for section in extracted.sections]
    assert ("Titre du document",) in paths
    assert ("Titre du document", "1. Première partie") in paths
    assert ("Titre du document", "1. Première partie", "1.1. Sous-partie") in paths
    assert ("Titre du document", "2. Deuxième partie") in paths


def test_table_is_its_own_section(loader: MarkdownLoader, make_md: Callable[..., Path]) -> None:
    extracted = loader.load(make_md(DOC))
    tables = [s for s in extracted.sections if s.kind is SectionKind.TABLE]
    assert len(tables) == 1
    assert "Colonne A" in tables[0].text and "b1" in tables[0].text
    assert tables[0].heading_path[-1] == "1.1. Sous-partie"


def test_list_stays_with_its_paragraph(
    loader: MarkdownLoader, make_md: Callable[..., Path]
) -> None:
    extracted = loader.load(make_md(DOC))
    section = next(s for s in extracted.sections if "Un paragraphe." in s.text)
    assert "- puce un" in section.text and "- puce deux" in section.text


def test_hash_in_code_fence_is_not_a_heading(
    loader: MarkdownLoader, make_md: Callable[..., Path]
) -> None:
    content = "# Titre\n\nTexte.\n\n```bash\n# ceci est un commentaire\n```\n"
    extracted = loader.load(make_md(content))
    assert all(section.heading_path == ("Titre",) for section in extracted.sections)


def test_empty_file_is_an_error(loader: MarkdownLoader, make_md: Callable[..., Path]) -> None:
    with pytest.raises(LoaderError, match="vide"):
        loader.load(make_md("   \n\n"))
