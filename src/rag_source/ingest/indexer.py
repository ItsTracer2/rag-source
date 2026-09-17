"""Indexation incrémentale du corpus.

Le défaut le plus coûteux de l'implémentation d'origine se situait ici. Ses
identifiants de chunk étaient positionnels (``document.pdf:6:2``) : modifier un
document ne changeait pas les identifiants, donc les anciens chunks restaient en
base. Le système continuait de répondre avec du contenu périmé, sans le signaler.
Un fichier supprimé restait indexé indéfiniment.

Ici, chaque document est comparé à l'empreinte déjà indexée :

- **inchangé** : rien à faire, et surtout aucun embedding recalculé ;
- **nouveau** : découpé, vectorisé, ajouté ;
- **modifié** : tous ses chunks sont supprimés avant réindexation ;
- **disparu** : ses chunks sont supprimés.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rag_source.clients.embedder import Embedder
from rag_source.domain import Chunk, LoadedDocument
from rag_source.ingest.chunker import ChunkingConfig, chunk_document
from rag_source.ingest.corpus import file_sha256, iter_corpus, load_document
from rag_source.ingest.loaders import LoaderError
from rag_source.ingest.tokenizer import TokenCounter
from rag_source.store.base import ChunkStore
from rag_source.store.sparse import average_token_length, encode_document

logger = logging.getLogger(__name__)

ProgressHook = Callable[[str, str], None]
"""Appelée avec (action, source) : « ajouté », « modifié », « inchangé », « supprimé »."""


@dataclass(slots=True)
class IndexReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    chunks_written: int = 0
    seconds: float = 0.0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)

    def summary(self) -> str:
        return (
            f"{len(self.added)} ajouté(s), {len(self.updated)} modifié(s), "
            f"{len(self.unchanged)} inchangé(s), {len(self.removed)} supprimé(s), "
            f"{len(self.failed)} en échec — {self.chunks_written} chunk(s) écrit(s) "
            f"en {self.seconds:.1f} s"
        )


def index_corpus(
    root: Path,
    store: ChunkStore,
    embedder: Embedder,
    counter: TokenCounter,
    *,
    config: ChunkingConfig | None = None,
    force: bool = False,
    progress: ProgressHook | None = None,
) -> IndexReport:
    """Met l'index en accord avec le contenu de ``root``.

    ``force`` réindexe tout, même ce qui n'a pas changé : utile après un changement
    de découpage ou de modèle d'embedding, où les vecteurs existants ne sont plus
    comparables aux nouveaux.
    """
    started = time.monotonic()
    report = IndexReport()

    if not root.is_dir():
        raise FileNotFoundError(f"Dossier de corpus introuvable : {root}")

    store.create(embedder.dimension)
    _check_dimension(store, embedder)
    indexed = {} if force else store.indexed_versions()

    documents_to_write: list[LoadedDocument] = []
    present: set[str] = set()
    for path in iter_corpus(root):
        source = path.relative_to(root).as_posix()
        present.add(source)
        # L'empreinte se calcule sur les octets du fichier : inutile d'ouvrir un PDF,
        # d'en extraire la structure et de le découper pour découvrir ensuite qu'il
        # n'a pas bougé. Sur le corpus de référence, cela ramène un passage sans
        # changement de 26 secondes à moins d'une seconde.
        if indexed.get(source) == file_sha256(path):
            report.unchanged.append(source)
            _notify(progress, "inchangé", source)
            continue
        try:
            document, _warnings = load_document(path, root)
        except (LoaderError, OSError) as exc:
            report.failed.append((source, str(exc)))
            _notify(progress, "échec", source)
            continue
        if source in indexed:
            report.updated.append(source)
            _notify(progress, "modifié", source)
        else:
            report.added.append(source)
            _notify(progress, "ajouté", source)
        documents_to_write.append(document)

    for source in sorted(set(indexed) - present):
        store.delete_source(source)
        report.removed.append(source)
        _notify(progress, "supprimé", source)

    if documents_to_write:
        report.chunks_written = _write(documents_to_write, store, embedder, counter, config)

    report.seconds = time.monotonic() - started
    logger.info("Indexation terminée : %s", report.summary())
    return report


def _write(
    documents: list[LoadedDocument],
    store: ChunkStore,
    embedder: Embedder,
    counter: TokenCounter,
    config: ChunkingConfig | None,
) -> int:
    chunks: list[Chunk] = []
    for document in documents:
        # Un document déjà connu voit ses anciens chunks retirés avant réécriture :
        # leurs identifiants dépendent de l'empreinte du fichier, donc une simple
        # réécriture laisserait cohabiter deux versions du même document.
        store.delete_source(document.source)
        chunks.extend(chunk_document(document, counter, config))

    if not chunks:
        return 0

    texts = [chunk.text for chunk in chunks]
    average_length = average_token_length(texts)
    dense = embedder.embed(texts)
    sparse = [encode_document(text, average_length) for text in texts]
    store.upsert(chunks, dense, sparse)
    return len(chunks)


def _check_dimension(store: ChunkStore, embedder: Embedder) -> None:
    """Refuse d'écrire dans une collection créée avec un autre modèle d'embedding.

    Sans ce contrôle, l'index mélangerait des vecteurs incomparables et la recherche
    deviendrait silencieusement absurde — exactement le risque que courait
    l'implémentation d'origine, dont le modèle d'embedding n'était inscrit nulle part.
    """
    existing = store.dimension()
    if existing is not None and existing != embedder.dimension:
        raise ValueError(
            f"La collection attend des vecteurs de dimension {existing}, "
            f"le modèle en produit {embedder.dimension}. Réindexer avec --force "
            "après avoir supprimé la collection."
        )


def _notify(progress: ProgressHook | None, action: str, source: str) -> None:
    if progress is not None:
        progress(action, source)
