from collections.abc import Callable
from pathlib import Path

import pytest

from rag_source.domain import DocumentFormat
from rag_source.ingest.corpus import file_sha256, iter_corpus, load_corpus, load_document
from rag_source.ingest.loaders import LoaderError, loader_for, supported_extensions

MD = "# Titre\n\nUn paragraphe de contenu suffisamment long pour être conservé.\n"


class TestRegistry:
    def test_all_formats_registered(self) -> None:
        assert {
            ".pdf",
            ".md",
            ".markdown",
            ".xlsx",
            ".xlsm",
            ".csv",
            ".tsv",
            ".txt",
            ".html",
            ".htm",
            ".docx",
        } <= supported_extensions()

    def test_extension_is_case_insensitive(self, tmp_path: Path) -> None:
        assert loader_for(tmp_path / "DOC.PDF") is not None

    def test_unknown_extension(self, tmp_path: Path) -> None:
        assert loader_for(tmp_path / "archive.zip") is None


class TestDiscovery:
    def test_walks_subdirectories_and_skips_unknown(
        self, tmp_path: Path, make_md: Callable[..., Path]
    ) -> None:
        """Le pipeline d'origine ne lisait que la racine : les PDF rangés dans un
        sous-dossier n'étaient jamais indexés, sans le moindre avertissement."""
        (tmp_path / "sous" / "dossier").mkdir(parents=True)
        (tmp_path / "a.md").write_text(MD, encoding="utf-8")
        (tmp_path / "sous" / "dossier" / "b.md").write_text(MD, encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")

        found = [p.relative_to(tmp_path).as_posix() for p in iter_corpus(tmp_path)]
        assert found == ["a.md", "sous/dossier/b.md"]

    def test_skips_hidden_and_office_lock_files(self, tmp_path: Path) -> None:
        (tmp_path / ".cache").mkdir()
        (tmp_path / ".cache" / "a.md").write_text(MD, encoding="utf-8")
        (tmp_path / "~$ouvert.xlsx").write_bytes(b"lock")
        assert list(iter_corpus(tmp_path)) == []


class TestLoadDocument:
    def test_normalized_document(self, tmp_path: Path) -> None:
        (tmp_path / "sous").mkdir()
        path = tmp_path / "sous" / "guide.md"
        path.write_text(MD, encoding="utf-8")

        document, warnings = load_document(path, tmp_path)
        assert document.source == "sous/guide.md"  # chemin relatif, stable
        assert document.format is DocumentFormat.MARKDOWN
        assert document.sha256 == file_sha256(path)
        assert document.title == "Titre"
        assert warnings == ()

    def test_sha_changes_with_content(self, tmp_path: Path) -> None:
        path = tmp_path / "a.md"
        path.write_text(MD, encoding="utf-8")
        before = file_sha256(path)
        path.write_text(MD + "\nAjout de contenu.\n", encoding="utf-8")
        assert file_sha256(path) != before

    def test_source_is_normalized_to_nfc(self, tmp_path: Path) -> None:
        """macOS décompose les accents des noms de fichiers : à normaliser.

        Sans cela, « nécessaires.md » écrit à la main ne correspond pas au même nom
        lu depuis le disque, et les filtres par source échouent en silence.
        """
        import unicodedata

        decomposed = unicodedata.normalize("NFD", "référence.md")
        path = tmp_path / decomposed
        path.write_text(MD, encoding="utf-8")

        document, _ = load_document(path, tmp_path)
        assert document.source == unicodedata.normalize("NFC", "référence.md")

    def test_unsupported_format(self, tmp_path: Path) -> None:
        path = tmp_path / "a.zip"
        path.write_bytes(b"PK")
        with pytest.raises(LoaderError, match="Format non géré"):
            load_document(path, tmp_path)


class TestLoadCorpus:
    def test_failure_does_not_stop_the_run(self, tmp_path: Path) -> None:
        (tmp_path / "bon.md").write_text(MD, encoding="utf-8")
        (tmp_path / "vide.md").write_text("", encoding="utf-8")
        (tmp_path / "casse.pdf").write_bytes(b"ceci n'est pas un PDF")

        report = load_corpus(tmp_path)
        assert [d.source for d in report.documents] == ["bon.md"]
        assert {f.source for f in report.failures} == {"vide.md", "casse.pdf"}
        assert all(f.error for f in report.failures)

    def test_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_corpus(tmp_path / "absent")
