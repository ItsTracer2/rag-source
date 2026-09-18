"""Ligne de commande de RAG-Source.

    rag-source health | docs | ask | search        ← clients de l'API
    rag-source ingest | inspect | eval             ← outils d'exploitation

Deux familles, et la distinction est volontaire.

Les commandes d'**interrogation** parlent à l'API par HTTP, exactement comme
l'interface web : une seule implémentation de la recherche et de la génération, un
seul endroit où corriger un défaut. Elles fonctionnent donc aussi à travers un
tunnel, sans accès aux fichiers.

Les commandes d'**exploitation** travaillent en local, parce qu'elles lisent le
corpus sur disque : indexer suppose de voir les fichiers, pas seulement l'API.

Aucune dépendance supplémentaire : ``argparse`` et ``httpx``, déjà présents.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx

from rag_source.config import Settings, get_settings

DEFAULT_API_URL = "http://127.0.0.1:8000"
TIMEOUT = 600.0

# Couleurs ANSI, désactivées dès que la sortie n'est pas un terminal : un pipe ou
# un fichier de journal ne doit pas se remplir de codes d'échappement.
_COLOR = sys.stdout.isatty()
DIM = "\033[2m" if _COLOR else ""
BOLD = "\033[1m" if _COLOR else ""
GREEN = "\033[32m" if _COLOR else ""
YELLOW = "\033[33m" if _COLOR else ""
RED = "\033[31m" if _COLOR else ""
OFF = "\033[0m" if _COLOR else ""


class CommandError(RuntimeError):
    """Erreur destinée à l'utilisateur : affichée sans trace d'appels."""


# ── Client HTTP ──────────────────────────────────────────────────────────────


def api_client(settings: Settings, base_url: str) -> httpx.Client:
    token = settings.api_token.get_secret_value() if settings.api_token else None
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=TIMEOUT)


def call(client: httpx.Client, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    try:
        response = client.request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        raise CommandError(
            f"API injoignable ({client.base_url}) : {exc}\nDémarrer la pile : docker compose up -d"
        ) from exc
    if response.status_code == 401:
        raise CommandError("Jeton refusé. Vérifier RAG_SOURCE_API_TOKEN dans .env")
    if response.status_code >= 400:
        raise CommandError(f"L'API a répondu {response.status_code} : {response.text[:300]}")
    payload: dict[str, Any] = response.json()
    return payload


def sse_events(client: httpx.Client, path: str, body: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    """Parcourt un flux d'événements SSE renvoyé par l'API."""
    try:
        with client.stream("POST", path, json=body) as response:
            if response.status_code >= 400:
                response.read()
                raise CommandError(
                    f"L'API a répondu {response.status_code} : {response.text[:300]}"
                )
            name = ""
            for line in response.iter_lines():
                if line.startswith("event:"):
                    name = line[len("event:") :].strip()
                elif line.startswith("data:") and name:
                    yield name, json.loads(line[len("data:") :].strip())
    except httpx.HTTPError as exc:
        raise CommandError(f"Flux interrompu : {exc}") from exc


# ── Commandes d'interrogation ────────────────────────────────────────────────


def cmd_health(args: argparse.Namespace, settings: Settings) -> int:
    with api_client(settings, args.api_url) as client:
        body = call(client, "GET", "/health")
    if args.json:
        print(json.dumps(body, ensure_ascii=False, indent=2))
        return 0 if body["status"] == "ok" else 1

    ok = body["status"] == "ok"
    print(f"{BOLD}État{OFF}          {GREEN if ok else YELLOW}{body['status']}{OFF}")
    external = body["sovereignty"] == "external"
    mark = (
        f"{YELLOW}externe — les extraits du corpus quittent l'hôte{OFF}"
        if external
        else f"{GREEN}local — aucune donnée ne sort{OFF}"
    )
    print(f"{BOLD}Souveraineté{OFF}  {mark}")
    print(f"{BOLD}Collection{OFF}    {body['collection']} · {body['indexed_chunks']} extrait(s)")
    for service in body["services"]:
        state = f"{GREEN}ok{OFF}" if service["reachable"] else f"{RED}injoignable{OFF}"
        detail = f" {DIM}{service['detail']}{OFF}" if service.get("detail") else ""
        print(f"  {service['name']:<8} {state}{detail}")
    return 0 if ok else 1


def cmd_docs(args: argparse.Namespace, settings: Settings) -> int:
    with api_client(settings, args.api_url) as client:
        body = call(client, "GET", "/v1/documents")
    if args.json:
        print(json.dumps(body, ensure_ascii=False, indent=2))
        return 0
    if not body["documents"]:
        print("Aucun document indexé. Lancer : rag-source ingest data")
        return 1
    width = max(len(document["source"]) for document in body["documents"])
    for document in body["documents"]:
        print(f"{document['source']:<{width}}  {document['chunks']:>5} extrait(s)")
    print(f"\n{len(body['documents'])} document(s), {body['total_chunks']} extrait(s).")
    return 0


def cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    with api_client(settings, args.api_url) as client:
        body = call(
            client,
            "POST",
            "/v1/search",
            json={
                "query": args.query,
                "mode": args.mode,
                "source": args.source,
                "top_k": args.top_k,
            },
        )
    if args.json:
        print(json.dumps(body, ensure_ascii=False, indent=2))
        return 0
    if not body["passages"]:
        print(
            f"{YELLOW}Aucun passage pertinent{OFF} ({body['candidates']} candidat(s) examiné(s))."
        )
        return 1
    for number, passage in enumerate(body["passages"], start=1):
        print(f"{BOLD}[{number}]{OFF} {passage['location']}  {DIM}score {passage['score']}{OFF}")
        print(f"     {_indent(passage['text'], args.width)}\n")
    return 0


def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    body = {
        "question": args.question,
        "mode": args.mode,
        "source": args.source,
        "top_k": args.top_k,
    }
    with api_client(settings, args.api_url) as client:
        if args.json or args.no_stream:
            answer = call(client, "POST", "/v1/ask", json=body)
            if args.json:
                print(json.dumps(answer, ensure_ascii=False, indent=2))
            else:
                print(answer["answer"])
                _print_citations(answer["citations"], answer["invalid_citations"])
            return 0 if not answer["refused"] else 1
        return _ask_streaming(client, body)


def _ask_streaming(client: httpx.Client, body: dict[str, Any]) -> int:
    """Affiche la réponse au fil de l'eau, sources d'abord.

    L'ordre compte : sur CPU, la génération demande plusieurs secondes, et voir
    d'où viendra la réponse avant qu'elle ne s'écrive rend l'attente lisible.
    """
    passages: list[dict[str, Any]] = []
    refused = False
    for name, data in sse_events(client, "/v1/ask/stream", body):
        if name == "passages":
            passages = data["passages"]
            refused = not passages
            if passages:
                print(f"{DIM}Sources retenues :{OFF}")
                for number, passage in enumerate(passages, start=1):
                    print(f"{DIM}  [{number}] {passage['location']}{OFF}")
                print()
            else:
                print(f"{YELLOW}Aucun passage pertinent dans le corpus.{OFF}")
        elif name == "token":
            print(data["text"], end="", flush=True)
        elif name == "error":
            print(f"\n{RED}Erreur : {data['detail']}{OFF}", file=sys.stderr)
            return 1
        elif name == "done":
            print()
            _print_citations(data["citations"], data["invalid_citations"])
    return 1 if refused else 0


def _print_citations(citations: list[dict[str, Any]], invalid: list[int]) -> None:
    if citations:
        print()
        for citation in citations:
            print(f"{DIM}  [{citation['number']}] {citation['location']}{OFF}")
    if invalid:
        print(f"{YELLOW}  Citations inexistantes : {', '.join(map(str, invalid))}{OFF}")


# ── Commandes d'exploitation ─────────────────────────────────────────────────


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    from rag_source.clients.embedder import LlamaCppEmbedder
    from rag_source.ingest.indexer import index_corpus
    from rag_source.ingest.tokenizer import get_token_counter
    from rag_source.store.qdrant import QdrantStore

    root = args.path or settings.data_dir
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    icons = {"ajouté": "＋", "modifié": "↻", "supprimé": "－", "inchangé": "=", "échec": "✗"}

    with (
        LlamaCppEmbedder(settings.embed_url) as embedder,
        QdrantStore(settings.qdrant_url, settings.qdrant_collection, api_key=api_key) as store,
    ):
        if args.reset:
            print("Suppression de la collection existante.")
            store.drop()

        def progress(action: str, source: str) -> None:
            if args.verbose or action != "inchangé":
                print(f"  {icons.get(action, '·')} {action:9} {source}")

        report = index_corpus(
            root,
            store,
            embedder,
            get_token_counter(settings.tokenizer_path),
            force=args.force or args.reset,
            progress=progress,
        )
        total = store.count()

    for source, error in report.failed:
        print(f"  {RED}✗ {source}{OFF} : {error}")
    print(f"\n{report.summary()}.")
    print(f"Index : {total} extrait(s) dans « {settings.qdrant_collection} ».")
    return 1 if report.failed else 0


def cmd_inspect(args: argparse.Namespace, settings: Settings) -> int:
    """Montre ce que l'ingestion tire du corpus, sans rien indexer.

    C'est l'outil à dégainer quand une réponse est mauvaise : avant de suspecter la
    recherche ou le modèle, on vérifie que le texte a bien été extrait.
    """
    import statistics
    from collections import Counter

    from rag_source.domain import SectionKind
    from rag_source.ingest.chunker import chunk_document
    from rag_source.ingest.corpus import load_corpus
    from rag_source.ingest.tokenizer import get_token_counter

    root = args.path or settings.data_dir
    counter = get_token_counter(settings.tokenizer_path)
    sizes: list[int] = []
    kinds: Counter[str] = Counter()
    failures = 0
    documents = 0

    report = load_corpus(root)
    for entry in report.entries:
        if entry.error is not None:
            print(f"{RED}✗ {entry.source}{OFF}\n   {entry.error}")
            failures += 1
            continue
        document = entry.document
        assert document is not None
        chunks = chunk_document(document, counter)
        sizes.extend(chunk.token_count for chunk in chunks)
        kinds.update(chunk.kind.value for chunk in chunks)
        documents += 1
        section_kinds = Counter(section.kind for section in document.sections)
        print(f"{GREEN}✓ {entry.source}{OFF}")
        print(f"   titre    : {document.title[:80]}")
        print(
            f"   sections : {len(document.sections)} "
            f"(prose {section_kinds[SectionKind.PROSE]}, "
            f"tableaux {section_kinds[SectionKind.TABLE]}, "
            f"records {section_kinds[SectionKind.RECORD]})"
        )
        print(f"   chunks   : {len(chunks)}")
        for warning in entry.warnings:
            print(f"   {YELLOW}⚠ {warning}{OFF}")
        if args.sample and document.sections:
            section = max(document.sections, key=lambda s: len(s.text))
            trail = " › ".join(section.heading_path) or "(racine)"
            print(f"   extrait  : [{trail}] {section.text[:300]!r}")

    if sizes:
        print(
            f"\n{documents} document(s), {len(sizes)} chunk(s), {failures} échec(s).\n"
            f"Tokens par chunk : médiane {statistics.median(sizes):.0f}, "
            f"moyenne {statistics.mean(sizes):.0f}, max {max(sizes)}. "
            f"Répartition : {dict(kinds)}."
        )
    return 1 if failures else 0


def cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    from rag_source.clients.embedder import LlamaCppEmbedder
    from rag_source.clients.reranker import LlamaCppReranker
    from rag_source.eval.retrieval import evaluate, load_cases
    from rag_source.ingest.tokenizer import get_token_counter
    from rag_source.retrieval.search import SearchConfig, SearchMode, SearchService
    from rag_source.store.qdrant import QdrantStore

    cases = load_cases(args.dataset)
    modes = [SearchMode(value) for value in args.mode] if args.mode else list(SearchMode)
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
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
        service = SearchService(
            store, embedder, reranker, get_token_counter(settings.tokenizer_path), config
        )
        for mode in modes:
            report = evaluate(service, cases, mode=mode, top_k=args.top_k)
            print(report.row(), flush=True)
            if args.failures:
                for outcome in report.failures():
                    expected = ", ".join(outcome.case.expected_sources) or "(aucune source)"
                    found = ", ".join(outcome.retrieved_sources[:3]) or "(abstention)"
                    print(f"\n  {RED}✗ {outcome.case.id}{OFF} — {outcome.case.question}")
                    print(f"      attendu : {expected}")
                    print(f"      trouvé  : {found}")
    return 0


# ── Analyse des arguments ────────────────────────────────────────────────────


def _indent(text: str, width: int) -> str:
    """Extrait d'une ligne, sans le fil d'Ariane déjà affiché au-dessus.

    Chaque chunk commence par son fil d'Ariane (utile au modèle d'embedding et au
    LLM) ; le répéter sous la référence n'apporterait rien au lecteur.
    """
    body = text.split("\n\n", 1)[-1]
    excerpt = " ".join(body.split())
    return excerpt[:width] + ("…" if len(excerpt) > width else "")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-source",
        description="Réponses sourcées et citées sur un corpus documentaire local.",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"URL de l'API pour les commandes d'interrogation (défaut : {DEFAULT_API_URL}).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_query_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--source", help="Restreindre à un document.")
        sub.add_argument(
            "--mode",
            default="hybrid+rerank",
            choices=["dense", "sparse", "hybrid", "hybrid+rerank"],
            help="Stratégie de recherche.",
        )
        sub.add_argument("--top-k", type=int, help="Nombre de passages retenus.")
        sub.add_argument("--json", action="store_true", help="Sortie JSON brute.")

    health = subparsers.add_parser("health", help="État des services et souveraineté.")
    health.add_argument("--json", action="store_true")
    health.set_defaults(handler=cmd_health)

    docs = subparsers.add_parser("docs", help="Documents indexés.")
    docs.add_argument("--json", action="store_true")
    docs.set_defaults(handler=cmd_docs)

    ask = subparsers.add_parser("ask", help="Poser une question (réponse citée).")
    ask.add_argument("question")
    ask.add_argument("--no-stream", action="store_true", help="Attendre la réponse complète.")
    add_query_options(ask)
    ask.set_defaults(handler=cmd_ask)

    search = subparsers.add_parser("search", help="Chercher sans générer de réponse.")
    search.add_argument("query")
    search.add_argument("--width", type=int, default=160, help="Largeur des extraits affichés.")
    add_query_options(search)
    search.set_defaults(handler=cmd_search)

    ingest = subparsers.add_parser("ingest", help="Indexer le corpus (incrémental).")
    ingest.add_argument("path", nargs="?", type=Path, help="Dossier du corpus.")
    ingest.add_argument("--force", action="store_true", help="Réindexer même l'inchangé.")
    ingest.add_argument("--reset", action="store_true", help="Vider la collection d'abord.")
    ingest.add_argument("-v", "--verbose", action="store_true", help="Lister aussi l'inchangé.")
    ingest.set_defaults(handler=cmd_ingest)

    inspect = subparsers.add_parser("inspect", help="Voir ce que l'ingestion extrait.")
    inspect.add_argument("path", nargs="?", type=Path)
    inspect.add_argument("--sample", action="store_true", help="Montrer un extrait par document.")
    inspect.set_defaults(handler=cmd_inspect)

    evaluate_parser = subparsers.add_parser("eval", help="Mesurer la qualité de la recherche.")
    evaluate_parser.add_argument(
        "dataset", nargs="?", type=Path, default=Path("eval/datasets/corpus-reference.jsonl")
    )
    evaluate_parser.add_argument(
        "--mode", action="append", choices=["dense", "sparse", "hybrid", "hybrid+rerank"]
    )
    evaluate_parser.add_argument("--top-k", type=int, default=6)
    evaluate_parser.add_argument("--candidates", type=int, default=12)
    evaluate_parser.add_argument("--min-score", type=float, default=-5.0)
    evaluate_parser.add_argument("--failures", action="store_true")
    evaluate_parser.set_defaults(handler=cmd_eval)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = get_settings()
        handler: Any = args.handler
        return int(handler(args, settings))
    except CommandError as exc:
        print(f"{RED}{exc}{OFF}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrompu.", file=sys.stderr)
        return 130
    except Exception as exc:  # message lisible plutôt qu'une trace d'appels
        print(f"{RED}Erreur : {exc}{OFF}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
