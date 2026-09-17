"""Compare les stratégies de recherche sur un jeu de questions.

    uv run python -m rag_source.eval
    uv run python -m rag_source.eval --mode hybrid+rerank --failures
    uv run python -m rag_source.eval --mode hybrid+rerank --candidates 12

La pile doit être démarrée et le corpus indexé. Les options permettent de mesurer
l'effet d'un réglage au lieu de le choisir au jugé.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_source.clients.embedder import LlamaCppEmbedder
from rag_source.clients.reranker import LlamaCppReranker
from rag_source.config import get_settings
from rag_source.eval.retrieval import evaluate, load_cases
from rag_source.ingest.tokenizer import get_token_counter
from rag_source.retrieval.search import SearchConfig, SearchMode, SearchService
from rag_source.store.qdrant import QdrantStore

DEFAULT_DATASET = Path("eval/datasets/corpus-reference.jsonl")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Évalue la qualité de la recherche.")
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--mode",
        action="append",
        choices=[mode.value for mode in SearchMode],
        help="Mode à évaluer (répétable). Par défaut : tous.",
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=SearchConfig().candidates)
    parser.add_argument("--min-score", type=float, default=SearchConfig().min_score)
    parser.add_argument("--failures", action="store_true", help="Détaille les cas ratés.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    modes = [SearchMode(value) for value in args.mode] if args.mode else list(SearchMode)

    cases = load_cases(args.dataset)
    settings = get_settings()
    api_key = (
        settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key is not None else None
    )
    config = SearchConfig(candidates=args.candidates, min_score=args.min_score)

    print(
        f"{len(cases)} question(s) — {args.dataset} "
        f"(candidats {config.candidates}, seuil {config.min_score})\n",
        flush=True,
    )
    with (
        QdrantStore(settings.qdrant_url, settings.qdrant_collection, api_key=api_key) as store,
        LlamaCppEmbedder(settings.embed_url) as embedder,
        LlamaCppReranker(settings.rerank_url) as reranker,
    ):
        counter = get_token_counter(settings.tokenizer_path)
        service = SearchService(store, embedder, reranker, counter, config)

        for mode in modes:
            report = evaluate(service, cases, mode=mode, top_k=args.top_k)
            # flush : la sortie est souvent redirigée vers un fichier, et un résultat
            # qui n'apparaît qu'à la fin du calcul n'aide personne à patienter.
            print(report.row(), flush=True)
            if args.failures:
                for outcome in report.failures():
                    attendu = ", ".join(outcome.case.expected_sources) or "(aucune source)"
                    trouve = ", ".join(outcome.retrieved_sources[:3]) or "(abstention)"
                    print(f"\n  ✗ {outcome.case.id} — {outcome.case.question}", flush=True)
                    print(f"      attendu : {attendu}")
                    print(f"      trouvé  : {trouve}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
