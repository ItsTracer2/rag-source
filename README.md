# RAG-Source

Posez des questions à vos documents, obtenez des réponses **sourcées, citées et
vérifiées** — ou un refus franc quand l'information n'y est pas.

RAG-Source est un système de *Retrieval-Augmented Generation* auto-hébergeable de
bout en bout : vos documents sont indexés localement, interrogés localement, et la
réponse est produite par un modèle qui tourne sur votre machine. Aucune donnée ne
sort par défaut.

```
$ rag-source ask "Combien de temps dure la qualification d'un prestataire ?"

Sources retenues :
  [1] exigences/eidas_psc-qualifies_v1.3.pdf, p. 5 — II.1. Modalités de qualification
  [2] exigences/eidas_horodatage-qualifie_v1.1.pdf, p. 5 — II.1. Modalités…

La qualification d'un prestataire dure au maximum deux ans [1].
```

**Générique par construction** : notice d'électroménager, manuel technique, export
métier ou documentation réglementaire, en français comme en anglais.

| | |
|---|---|
| **Formats** | PDF (OCR compris), Markdown, Excel, CSV/TSV, texte, HTML, DOCX |
| **Interfaces** | API HTTP, interface web, ligne de commande |
| **Modèles** | Qwen2.5 (3B / 7B / 14B), bge-m3, bge-reranker-v2-m3 — tous locaux |
| **Recherche** | hybride (sémantique + mots-clés), reclassée, avec seuil de pertinence |
| **Mesuré** | hit@6 100 %, MRR 0,98, 100 % d'abstention sur les questions hors corpus |

---

## Sommaire

- [Ce qui distingue ce projet](#ce-qui-distingue-ce-projet)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Configuration](#configuration)
- [Évaluation](#évaluation)
- [Déploiement sur un serveur](#déploiement-sur-un-serveur)
- [Structure du dépôt](#structure-du-dépôt)
- [Développement](#développement)
- [Dépannage](#dépannage)
- [Limites connues](#limites-connues)

---

## Ce qui distingue ce projet

**Les réponses sont citées, et les citations sont vérifiées.** Chaque passage est
numéroté et situé (`[1] manuel.pdf, p. 4 — Entretien › Plateau`) ; après
génération, les `[n]` sont extraits et confrontés aux passages réellement fournis.
Un numéro inventé est signalé, pas présenté comme une source.

**Le système sait se taire.** Un seuil de pertinence, calé par la mesure, écarte
les passages hors sujet. Si rien ne convient, le modèle **n'est pas appelé du
tout** : « je ne trouve pas cette information ». Sur le jeu de référence, les
questions sans réponse dans le corpus sont refusées dans 100 % des cas.

**La recherche est hybride.** Le vecteur dense sait qu'« entretenir le plateau »
ressemble à « nettoyer le plateau tournant » ; BM25 retrouve « R24 », « E07 » ou
« MW-104 », que la recherche sémantique situe mal. Les deux classements sont
fusionnés, puis reclassés par un modèle multilingue.

**L'indexation est réellement incrémentale.** Chaque chunk porte l'empreinte de son
fichier : un document modifié voit ses anciens chunks supprimés avant réécriture, un
document supprimé disparaît de l'index. Un passage sans changement prend moins d'une
seconde sur le corpus de référence.

**Les réglages viennent de mesures, pas d'intuitions.** Le nombre de candidats
reclassés, le seuil de pertinence et jusqu'à la formulation du prompt ont été
choisis en comparant des chiffres — voir [`docs/adr/`](docs/adr/), où chaque
décision est datée, justifiée et chiffrée.

---

## Architecture

```
                    ┌──────────────┐        ┌──────────────────┐
   Navigateur ────► │  ui (Caddy)  │ ─────► │                  │
                    └──────────────┘        │       api        │
   rag-source ─────────────────────────────►│    (FastAPI)     │
                                            │                  │
                                            └───┬────┬─────┬───┘
                        ┌───────────────────────┘    │     └──────────────┐
                        ▼                            ▼                    ▼
                 ┌─────────────┐            ┌────────────────┐    ┌──────────────┐
                 │   qdrant    │            │ embed · rerank │    │     llm      │
                 │ dense +BM25 │            │    bge-m3      │    │   Qwen2.5    │
                 └─────────────┘            └────────────────┘    └──────────────┘
                                               llama.cpp            llama.cpp
```

**Indexation** — un chargeur par format extrait la *structure* (titres, tableaux,
enregistrements), le découpage suit cette structure et mesure les tailles en tokens,
puis chaque chunk est vectorisé deux fois : densément (bge-m3) et en mots-clés
(BM25).

**Interrogation** — la question suit le même double chemin ; Qdrant fusionne les
deux classements (RRF) ; le reranker relit les candidats un par un et les réordonne ;
les passages trop faibles sont écartés ; ce qui reste tient dans un budget de tokens
et part au LLM avec la consigne de citer.

Les trois serveurs de modèles partagent la même image `llama.cpp` avec des arguments
différents : un seul runtime à connaître, et **aucun PyTorch** dans le projet.

Les décisions d'architecture sont documentées une par une :

| ADR | Sujet |
|---|---|
| [0001](docs/adr/0001-socle-du-projet.md) | Socle : uv, typage strict, garde de souveraineté |
| [0002](docs/adr/0002-chargeurs-par-format.md) | Un chargeur par format, extraction structurée |
| [0003](docs/adr/0003-outil-generique.md) | Outil générique, souveraineté propre au déploiement |
| [0004](docs/adr/0004-chunking-structurel.md) | Découpage guidé par la structure |
| [0005](docs/adr/0005-pile-de-modeles-et-services.md) | Un seul runtime, modèles épinglés |
| [0006](docs/adr/0006-indexation-incrementale.md) | Indexation incrémentale, deux vecteurs |
| [0007](docs/adr/0007-recherche-hybride-et-evaluation.md) | Recherche hybride et banc d'évaluation |
| [0008](docs/adr/0008-generation-citee-et-api.md) | Génération citée, API HTTP |
| [0009](docs/adr/0009-interface-web.md) | Interface web sans framework |
| [0010](docs/adr/0010-deploiement-scaleway.md) | Déploiement Scaleway |
| [0011](docs/adr/0011-ligne-de-commande.md) | Ligne de commande |

---

## Prérequis

| | Pourquoi |
|---|---|
| **Docker** avec Compose v2 | base vectorielle, serveurs de modèles, API, interface |
| **[uv](https://docs.astral.sh/uv/)** | dépendances Python et commande `rag-source` |
| **≈ 6 Go de RAM** pour Docker | trois modèles et l'index cohabitent |
| **≈ 5 Go de disque** | modèles GGUF (profil `small`) |

Python n'a pas besoin d'être installé : `uv` s'en charge.

Optionnel : **Tesseract** (`brew install tesseract tesseract-lang`) pour l'OCR des
PDF scannés en local — l'image Docker de l'API l'embarque déjà.

---

## Installation

```bash
git clone <votre-dépôt> rag-source && cd rag-source

uv sync                      # dépendances Python (versions verrouillées)
./scripts/init-env.sh        # .env local, secrets générés
./scripts/fetch-models.sh    # modèles GGUF, vérifiés par empreinte (~3,4 Go)
docker compose up -d         # qdrant, embed, rerank, llm, api, ui
```

Placez vos documents dans `data/` (les sous-dossiers sont lus), puis indexez :

```bash
uv run rag-source ingest data
open http://127.0.0.1:8080
```

La première indexation dure quelques minutes ; les suivantes ne traitent que ce qui
a changé.

### Choisir un modèle plus puissant

```bash
./scripts/fetch-models.sh medium     # Qwen2.5-7B (~4,7 Go)
# puis dans .env : RAG_SOURCE_LLM_FILE=Qwen2.5-7B-Instruct-Q4_K_M.gguf
docker compose up -d llm
```

Profils disponibles dans [`models.lock`](models.lock) : `small` (3B), `medium` (7B),
`large` (14B). La mémoire disponible est la vraie limite.

### macOS : LLM natif, nettement plus rapide

Docker n'accède pas au GPU sur macOS. Exécuté nativement, le même modèle profite de
Metal — **39 tokens/s au lieu de ~10** — et libère 1,2 Go :

```bash
brew install llama.cpp
llama-server --model models/qwen2.5-3b-instruct-q4_k_m.gguf \
             --host 127.0.0.1 --port 8081 --ctx-size 8192

docker compose stop llm      # le conteneur devient inutile
```

L'API le joint automatiquement (`host.docker.internal:8081`).

---

## Utilisation

### Interface web

<http://127.0.0.1:8080> — les sources s'affichent pendant que la réponse s'écrit,
les `[n]` sont cliquables, et le mode de souveraineté est visible en permanence.

### Ligne de commande

```bash
rag-source health                      # services, souveraineté, taille de l'index
rag-source docs                        # documents indexés
rag-source ask "Votre question ?"      # réponse citée, en flux
rag-source search "mots clés"          # passages seuls, sans génération
rag-source ingest data                 # indexation incrémentale
rag-source inspect data --sample       # ce que l'ingestion extrait d'un corpus
rag-source eval                        # qualité de la recherche
```

Options utiles : `--source <fichier>` pour restreindre à un document, `--mode` pour
changer de stratégie de recherche, `--json` pour une sortie machine.

Codes de retour : `0` succès, `1` résultat vide ou service dégradé, `2` erreur —
`rag-source health` fait donc une sonde de supervision utilisable telle quelle.

`health`, `docs`, `ask` et `search` passent par l'API et fonctionnent à travers un
tunnel SSH ; `ingest`, `inspect` et `eval` lisent le corpus sur disque et
s'exécutent donc là où il se trouve.

### API HTTP

Documentation interactive : <http://127.0.0.1:8000/docs>.

```bash
TOKEN=$(grep RAG_SOURCE_API_TOKEN .env | cut -d= -f2)

curl -s -X POST localhost:8000/v1/ask \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"question": "Votre question ?"}'
```

| Route | Rôle |
|---|---|
| `GET /health` | état des services, souveraineté, nombre d'extraits |
| `GET /v1/documents` | documents indexés |
| `POST /v1/search` | recherche seule — sépare ce qui est *trouvé* de ce qui est *dit* |
| `POST /v1/ask` | réponse complète avec ses citations vérifiées |
| `POST /v1/ask/stream` | même chose en flux SSE : `passages`, puis `token`, puis `done` |

Toutes les routes exigent un jeton Bearer. L'interface web n'en manipule pas :
Caddy l'ajoute côté serveur.

---

## Configuration

Tout passe par des variables d'environnement préfixées `RAG_SOURCE_`, lues dans
`.env`. Voir [`.env.example`](.env.example) pour la liste commentée.

| Variable | Défaut | Rôle |
|---|---|---|
| `RAG_SOURCE_API_TOKEN` | *(généré)* | jeton Bearer de l'API |
| `RAG_SOURCE_QDRANT_API_KEY` | *(généré)* | accès à la base vectorielle |
| `RAG_SOURCE_DATA_DIR` | `data` | dossier du corpus |
| `RAG_SOURCE_LLM_FILE` | `qwen2.5-3b-…gguf` | modèle servi par le conteneur `llm` |
| `RAG_SOURCE_LLM_CONTEXT_SIZE` | `4096` | fenêtre de contexte |
| `RAG_SOURCE_OCR_MODE` | `auto` | `auto`, `off` ou `force` |
| `RAG_SOURCE_OCR_LANGUAGES` | `eng+fra` | langues Tesseract |
| `RAG_SOURCE_REQUIRE_LOCAL_LLM` | `false` | refuse de démarrer si le LLM est externe |

### Utiliser un modèle distant

RAG-Source est un outil générique : rien n'oblige à rester local pour interroger une
documentation publique.

```ini
RAG_SOURCE_LLM_PROVIDER=external
RAG_SOURCE_LLM_BASE_URL=https://api.exemple.com/v1
RAG_SOURCE_LLM_MODEL=nom-du-modèle
RAG_SOURCE_LLM_API_KEY=…
```

Tout service exposant une API compatible OpenAI convient. **Les extraits de vos
documents sont alors envoyés à ce service** : `/health` bascule sur
`sovereignty: external`, l'interface affiche un bandeau, et le profil de déploiement
serveur refuse purement et simplement de démarrer dans ce mode.

---

## Évaluation

Un banc de mesure déterministe, **sans LLM juge**, compare les stratégies de
recherche sur un jeu de questions écrites à la main :

```bash
rag-source eval                                  # tous les modes
rag-source eval --mode hybrid+rerank --failures  # détail des cas ratés
rag-source eval --mode hybrid+rerank --candidates 30
```

Corpus de référence (30 questions, 1 114 extraits, Mac M2) :

| Mode | hit@6 | recall | MRR | abstention utile | durée |
|---|---|---|---|---|---|
| dense seul | 100 % | 95 % | 0,89 | 0 % | 3 s |
| BM25 seul | 96 % | 92 % | 0,87 | 0 % | 0,5 s |
| hybride | 96 % | 92 % | 0,86 | 0 % | 2 s |
| **hybride + reranking** | **100 %** | **96 %** | **0,98** | **100 %** | 242 s |

Lecture : le reranking n'améliore pas tant le taux de succès que le *classement*
(MRR 0,86 → 0,98) et, surtout, il est le seul mode capable de refuser une question
hors corpus. Détails et méthode dans [`eval/README.md`](eval/README.md).

Pour votre propre corpus, écrivez votre jeu de questions : un fichier JSONL, une
vingtaine de lignes suffisent pour comparer des configurations.

---

## Déploiement sur un serveur

Un profil Scaleway (France) est fourni : instance provisionnée par OpenTofu, données
sur un volume persistant qui survit à la destruction de la VM, **aucun port
applicatif ouvert** — l'accès passe par un tunnel SSH — et LLM local imposé.

```bash
cd deploy/scaleway
cp env.example .env && source .env            # identifiants Scaleway
cp terraform.tfvars.example terraform.tfvars  # clé SSH, adresse autorisée
./scripts/up.sh
```

Voir [`deploy/scaleway/README.md`](deploy/scaleway/README.md) : sauvegarde par
instantané Qdrant, restauration, arrêt sélectif (`down.sh` conserve les données,
`down.sh --all` les supprime après confirmation).

---

## Structure du dépôt

```
rag-source/
├── src/rag_source/
│   ├── config.py          Configuration typée, garde de souveraineté
│   ├── domain.py          Section, LoadedDocument, Chunk (immuables)
│   ├── cli.py             Commande rag-source
│   ├── ingest/
│   │   ├── loaders/       Un module par format (registre extensible)
│   │   ├── chunker.py     Découpage structurel, mesuré en tokens
│   │   ├── corpus.py      Parcours, empreintes, rapport d'erreurs
│   │   ├── indexer.py     Indexation incrémentale
│   │   └── tabular.py     Logique commune Excel / CSV
│   ├── clients/           embedder, reranker, llm (HTTP, sans SDK lourd)
│   ├── store/             Qdrant (REST), vecteurs creux BM25, protocole
│   ├── retrieval/         Recherche hybride, reranking, seuil, budget
│   ├── generation/        Prompts, citations vérifiées, service de réponse
│   ├── eval/              Métriques de recherche
│   └── api/               FastAPI : routes, schémas, dépendances
├── ui/                    Interface web (HTML/CSS/JS) + Caddyfile
├── tests/
│   ├── unit/              231 tests, sans réseau ni modèle
│   └── integration/       25 tests sur la pile réelle
├── eval/datasets/         Jeux de questions (JSONL)
├── deploy/scaleway/       OpenTofu, cloud-init, scripts d'exploitation
├── docker/api.Dockerfile  Image de l'API
├── docs/adr/              Décisions d'architecture, datées et chiffrées
├── scripts/               init-env.sh, fetch-models.sh
├── compose.yaml           Pile locale
├── models.lock            Modèles épinglés par empreinte sha256
└── data/                  Vos documents (jamais versionnés)
```

---

## Développement

```bash
uv run pre-commit install     # lint, format et typage avant chaque commit

uv run pytest                 # 231 tests unitaires, rapides, sans réseau
uv run pytest -m integration  # 25 tests sur la pile démarrée
uv run ruff check && uv run mypy src tests
```

Les tests unitaires ne dépendent d'aucun modèle ni service : les fichiers d'exemple
sont fabriqués par les tests eux-mêmes (PDF sur deux colonnes, page scannée, export
CSV, document Word…), et les services externes sont simulés. Les tests
d'intégration, eux, vérifient le contrat avec `llama.cpp`, Qdrant et l'API — ce
qu'aucun test unitaire ne peut voir.

L'intégration continue (GitHub Actions) exécute lint, format, typage strict et tests
unitaires sur chaque poussée.

---

## Dépannage

**Un conteneur redémarre pendant l'indexation** — mémoire insuffisante : les trois
modèles travaillent ensemble. Augmentez la mémoire allouée à Docker (6 Go
recommandés) ou exécutez le LLM nativement (voir plus haut).

**`rag-source ask` répond « je ne trouve pas » à tort** — vérifiez d'abord la
recherche, pas le modèle : `rag-source search "vos mots clés"`. Si les passages sont
absents, le problème est à l'ingestion (`rag-source inspect data --sample`) ou à
l'indexation ; s'ils sont présents, le seuil de pertinence est trop haut — le banc
d'évaluation permet de le régler (`rag-source eval --min-score …`).

**« Aucun texte extrait : PDF scanné »** — installez Tesseract et les paquets de
langue voulus, ou forcez l'OCR avec `RAG_SOURCE_OCR_MODE=force`.

**Qdrant répond 401** — le `.env` est absent ou la clé a changé :
`./scripts/init-env.sh`, puis `docker compose up -d`.

**La génération est lente** — c'est du CPU. Ordre de grandeur sur un Mac M2 avec le
modèle 3B natif : 6 s pour une réponse complète, contre une trentaine de secondes
pour le même modèle en conteneur. Un GPU change l'échelle.

---

## Limites connues

- **La vitesse dépend entièrement du matériel.** Sans GPU, le reranking (≈ 8 s par
  question) et la génération dominent le temps de réponse.
- **La vérification des citations contrôle qu'un numéro existe**, pas que la phrase
  est fidèle au passage cité. Mesurer la fidélité demanderait un modèle juge.
- **Le jeu d'évaluation compte 30 questions** : un écart de 4 points correspond à une
  seule question. Ces chiffres comparent des configurations entre elles, ils
  n'annoncent pas une performance absolue.
- **`indexed_versions()` parcourt la collection** pour retrouver les documents
  indexés : immédiat sur des milliers d'extraits, à revoir sur des centaines de
  milliers.
- **Le déploiement Scaleway n'a pas encore été exécuté en conditions réelles** : la
  configuration est validée (`tofu validate`, rendu du cloud-init, compose de
  production), mais la durée d'initialisation et le débit restent à mesurer.
- **Pas de multi-utilisateurs** : ni comptes, ni conversations sauvegardées, ni
  cloisonnement des documents. L'API est protégée par un jeton unique.
