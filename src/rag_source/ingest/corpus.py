"""Parcours du corpus : découverte des fichiers, empreintes, gestion des erreurs.

Un fichier illisible n'interrompt jamais le traitement : il est consigné dans le
rapport. L'implémentation d'origine se contentait d'un ``print`` au milieu du flux
d'exécution, ce qui rendait les échecs faciles à manquer.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from rag_source.domain import LoadedDocument
from rag_source.ingest.loaders import LoaderError, loader_for, supported_extensions

_CHUNK = 1 << 20
# Fichiers temporaires d'Office et fichiers cachés : jamais du contenu utile.
_IGNORED_PREFIXES = (".", "~$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def iter_corpus(root: Path) -> Iterator[Path]:
    """Fichiers du corpus dont le format est géré, en ordre stable et récursif."""
    extensions = supported_extensions()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if any(part.startswith(_IGNORED_PREFIXES) for part in path.relative_to(root).parts):
            continue
        yield path


def relative_source(path: Path, root: Path) -> str:
    """Identifiant stable d'un document : son chemin relatif, normalisé en NFC.

    macOS enregistre les noms de fichiers en forme décomposée (« é » = « e » + accent
    combinant), Linux et les fichiers de configuration écrits à la main utilisent la
    forme composée. Les deux s'affichent de façon identique mais ne sont pas égales :
    sans normalisation, un filtre par source ou un jeu d'évaluation cesse
    silencieusement de correspondre selon la machine.
    """
    return unicodedata.normalize("NFC", path.relative_to(root).as_posix())


def load_document(path: Path, root: Path) -> tuple[LoadedDocument, tuple[str, ...]]:
    """Charge un fichier et renvoie le document normalisé et ses avertissements."""
    loader = loader_for(path)
    if loader is None:
        raise LoaderError(f"Format non géré : {path.suffix}")
    extracted = loader.load(path)
    document = LoadedDocument(
        source=relative_source(path, root),
        sha256=file_sha256(path),
        format=loader.format,
        title=extracted.title,
        sections=extracted.sections,
    )
    return document, extracted.warnings


@dataclass(frozen=True, slots=True)
class DocumentReport:
    source: str
    document: LoadedDocument | None
    warnings: tuple[str, ...] = ()
    error: str | None = None


@dataclass(slots=True)
class CorpusReport:
    documents: list[LoadedDocument] = field(default_factory=list)
    entries: list[DocumentReport] = field(default_factory=list)

    @property
    def failures(self) -> list[DocumentReport]:
        return [entry for entry in self.entries if entry.error is not None]


def load_corpus(root: Path) -> CorpusReport:
    if not root.is_dir():
        raise FileNotFoundError(f"Dossier de corpus introuvable : {root}")
    report = CorpusReport()
    for path in iter_corpus(root):
        source = relative_source(path, root)
        try:
            document, warnings = load_document(path, root)
        except (LoaderError, OSError) as exc:
            report.entries.append(DocumentReport(source=source, document=None, error=str(exc)))
            continue
        report.documents.append(document)
        report.entries.append(DocumentReport(source=source, document=document, warnings=warnings))
    return report
