"""Rapport d'extraction : ``uv run python -m rag_source.ingest.report [dossier]``.

Outil de vérification pour l'humain : il montre ce que les chargeurs tirent
réellement du corpus (sections, pages, titres, avertissements) avant tout
découpage ou indexation. Il sera plus tard exposé par la commande
``rag-source inspect``.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from rag_source.domain import SectionKind
from rag_source.ingest.corpus import load_corpus


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0]) if args else Path("data")
    show_sample = "--sample" in args

    report = load_corpus(root)
    totals: Counter[str] = Counter()

    for entry in report.entries:
        if entry.error is not None:
            print(f"❌ {entry.source}\n   {entry.error}")
            totals["échecs"] += 1
            continue
        document = entry.document
        assert document is not None
        kinds = Counter(section.kind for section in document.sections)
        chars = sum(len(section.text) for section in document.sections)
        headings = len({section.heading_path for section in document.sections})
        print(
            f"✅ {entry.source}\n"
            f"   titre    : {document.title[:80]}\n"
            f"   sections : {len(document.sections)} "
            f"(prose {kinds[SectionKind.PROSE]}, tableaux {kinds[SectionKind.TABLE]}, "
            f"records {kinds[SectionKind.RECORD]}) · {headings} niveaux de titre distincts\n"
            f"   volume   : {chars} caractères"
        )
        for warning in entry.warnings:
            print(f"   ⚠️  {warning}")
        if show_sample and document.sections:
            section = max(document.sections, key=lambda s: len(s.text))
            path = " › ".join(section.heading_path) or "(racine)"
            print(f"   extrait  : [{path}] {section.text[:300]!r}")
        totals["documents"] += 1
        totals["sections"] += len(document.sections)
        totals["caractères"] += chars

    print(
        f"\n{totals['documents']} document(s), {totals['sections']} section(s), "
        f"{totals['caractères']} caractères, {totals['échecs']} échec(s)."
    )
    return 1 if totals["échecs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
