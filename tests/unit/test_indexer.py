"""Tests de l'indexation incrémentale, avec un embedder et un index simulés.

Ces doublures rendent visibles les deux propriétés qui comptent vraiment : ce qui
est *réellement* vectorisé (l'opération coûteuse) et ce qui reste en base après
coup (la correction des réponses).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rag_source.domain import Chunk
from rag_source.ingest.indexer import IndexReport, index_corpus
from rag_source.ingest.tokenizer import EstimatedTokenCounter
from rag_source.store.sparse import SparseVector

MD = "# Titre\n\nUn paragraphe de contenu suffisamment long pour produire un chunk.\n"
OTHER = "# Autre titre\n\nUn second document, avec un contenu différent du premier.\n"


class FakeEmbedder:
    """Compte les textes réellement vectorisés."""

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension
        self.embedded: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[float(len(text))] * self._dimension for text in texts]


class FakeStore:
    """Index en mémoire, avec le même contrat que QdrantStore."""

    def __init__(self, dimension: int | None = None) -> None:
        self.points: dict[str, Chunk] = {}
        self._dimension = dimension
        self.deleted_sources: list[str] = []

    def create(self, dimension: int) -> None:
        if self._dimension is None:
            self._dimension = dimension

    def dimension(self) -> int | None:
        return self._dimension

    def indexed_versions(self) -> dict[str, str]:
        return {chunk.source: chunk.doc_sha256 for chunk in self.points.values()}

    def delete_source(self, source: str) -> None:
        self.deleted_sources.append(source)
        for key in [k for k, c in self.points.items() if c.source == source]:
            del self.points[key]

    def upsert(
        self,
        chunks: Sequence[Chunk],
        dense: Sequence[list[float]],
        sparse: Sequence[SparseVector],
    ) -> None:
        assert len(chunks) == len(dense) == len(sparse)
        for chunk in chunks:
            self.points[chunk.id] = chunk

    def count(self) -> int:
        return len(self.points)


@pytest.fixture
def counter() -> EstimatedTokenCounter:
    return EstimatedTokenCounter()


def run(
    root: Path,
    store: FakeStore,
    embedder: FakeEmbedder,
    counter: EstimatedTokenCounter,
    **kwargs: Any,
) -> IndexReport:
    return index_corpus(root, store, embedder, counter, **kwargs)


class TestIncremental:
    def test_first_run_indexes_everything(
        self, tmp_path: Path, counter: EstimatedTokenCounter
    ) -> None:
        (tmp_path / "a.md").write_text(MD, encoding="utf-8")
        (tmp_path / "b.md").write_text(OTHER, encoding="utf-8")
        store, embedder = FakeStore(), FakeEmbedder()

        report = run(tmp_path, store, embedder, counter)
        assert sorted(report.added) == ["a.md", "b.md"]
        assert report.chunks_written == store.count() == 2

    def test_unchanged_files_are_not_embedded_again(
        self, tmp_path: Path, counter: EstimatedTokenCounter
    ) -> None:
        """Le second passage ne doit rien recalculer : c'est là que se joue le coût."""
        (tmp_path / "a.md").write_text(MD, encoding="utf-8")
        store, embedder = FakeStore(), FakeEmbedder()
        run(tmp_path, store, embedder, counter)
        embedder.embedded.clear()

        report = run(tmp_path, store, embedder, counter)
        assert report.unchanged == ["a.md"]
        assert embedder.embedded == []
        assert report.chunks_written == 0

    def test_modified_file_replaces_its_chunks(
        self, tmp_path: Path, counter: EstimatedTokenCounter
    ) -> None:
        """Défaut majeur de l'original : l'ancien contenu restait indexé."""
        path = tmp_path / "a.md"
        path.write_text(MD, encoding="utf-8")
        store, embedder = FakeStore(), FakeEmbedder()
        run(tmp_path, store, embedder, counter)
        before = set(store.points)

        path.write_text("# Titre\n\nLe contenu a entièrement changé de sujet ici.\n", "utf-8")
        report = run(tmp_path, store, embedder, counter)

        assert report.updated == ["a.md"]
        assert set(store.points).isdisjoint(before)  # aucun ancien chunk conservé
        assert all("entièrement changé" in chunk.text for chunk in store.points.values())

    def test_deleted_file_is_removed_from_the_index(
        self, tmp_path: Path, counter: EstimatedTokenCounter
    ) -> None:
        (tmp_path / "a.md").write_text(MD, encoding="utf-8")
        (tmp_path / "b.md").write_text(OTHER, encoding="utf-8")
        store, embedder = FakeStore(), FakeEmbedder()
        run(tmp_path, store, embedder, counter)

        (tmp_path / "b.md").unlink()
        report = run(tmp_path, store, embedder, counter)

        assert report.removed == ["b.md"]
        assert {chunk.source for chunk in store.points.values()} == {"a.md"}

    def test_force_reindexes_everything(
        self, tmp_path: Path, counter: EstimatedTokenCounter
    ) -> None:
        (tmp_path / "a.md").write_text(MD, encoding="utf-8")
        store, embedder = FakeStore(), FakeEmbedder()
        run(tmp_path, store, embedder, counter)
        embedder.embedded.clear()

        report = run(tmp_path, store, embedder, counter, force=True)
        assert report.added == ["a.md"]
        assert embedder.embedded  # tout est revectorisé


class TestRobustness:
    def test_unreadable_file_is_reported_not_fatal(
        self, tmp_path: Path, counter: EstimatedTokenCounter
    ) -> None:
        (tmp_path / "bon.md").write_text(MD, encoding="utf-8")
        (tmp_path / "casse.pdf").write_bytes(b"ceci n'est pas un PDF")
        store, embedder = FakeStore(), FakeEmbedder()

        report = run(tmp_path, store, embedder, counter)
        assert report.added == ["bon.md"]
        assert [source for source, _ in report.failed] == ["casse.pdf"]
        assert store.count() == 1

    def test_dimension_mismatch_is_refused(
        self, tmp_path: Path, counter: EstimatedTokenCounter
    ) -> None:
        """Une collection créée avec un autre modèle rendrait la recherche absurde."""
        (tmp_path / "a.md").write_text(MD, encoding="utf-8")
        store = FakeStore(dimension=1024)
        with pytest.raises(ValueError, match="dimension 1024"):
            run(tmp_path, store, FakeEmbedder(dimension=4), counter)

    def test_missing_directory(self, tmp_path: Path, counter: EstimatedTokenCounter) -> None:
        with pytest.raises(FileNotFoundError):
            run(tmp_path / "absent", FakeStore(), FakeEmbedder(), counter)


def test_progress_hook_reports_each_action(tmp_path: Path, counter: EstimatedTokenCounter) -> None:
    (tmp_path / "a.md").write_text(MD, encoding="utf-8")
    store, embedder = FakeStore(), FakeEmbedder()
    run(tmp_path, store, embedder, counter)

    seen: list[tuple[str, str]] = []
    run(
        tmp_path,
        store,
        embedder,
        counter,
        progress=lambda action, source: seen.append((action, source)),
    )
    assert seen == [("inchangé", "a.md")]
