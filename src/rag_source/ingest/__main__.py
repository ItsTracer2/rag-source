"""Indexation en ligne de commande, en attendant la commande ``rag-source``.

uv run python -m rag_source.ingest data            # incrémental
uv run python -m rag_source.ingest data --force    # tout réindexer
uv run python -m rag_source.ingest data --reset    # vider puis réindexer
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rag_source.clients.embedder import LlamaCppEmbedder
from rag_source.config import get_settings
from rag_source.ingest.indexer import index_corpus
from rag_source.ingest.tokenizer import get_token_counter
from rag_source.store.qdrant import QdrantStore

_ICONS = {"ajouté": "＋", "modifié": "↻", "supprimé": "－", "inchangé": "="}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    settings = get_settings()
    root = Path(args[0]) if args and not args[0].startswith("-") else settings.data_dir
    force = "--force" in args
    reset = "--reset" in args
    verbose = "--verbose" in args or "-v" in args

    api_key = (
        settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key is not None else None
    )
    counter = get_token_counter(settings.tokenizer_path)

    with (
        LlamaCppEmbedder(settings.embed_url) as embedder,
        QdrantStore(settings.qdrant_url, settings.qdrant_collection, api_key=api_key) as store,
    ):
        if reset:
            print("Suppression de la collection existante.")
            store.drop()

        def progress(action: str, source: str) -> None:
            if verbose or action != "inchangé":
                print(f"  {_ICONS.get(action, '·')} {action:9} {source}")

        report = index_corpus(
            root, store, embedder, counter, force=force or reset, progress=progress
        )
        total = store.count()

    for source, error in report.failed:
        print(f"  ✗ échec      {source} : {error}")
    print(f"\n{report.summary()}.")
    print(f"Index : {total} chunk(s) dans « {settings.qdrant_collection} ».")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
