# RAG-Source

Posez des questions à vos documents et obtenez des réponses **sourcées et citées**.

Générique par construction : notice d'électroménager, manuel technique, export
métier ou documentation réglementaire, en français comme en anglais. Formats gérés :
PDF (OCR compris), Markdown, Excel, CSV/TSV, texte brut, HTML et DOCX.

Auto-hébergeable de bout en bout : le mode 100 % local est le défaut, et le profil
de déploiement Scaleway l'impose.

> 🚧 Projet en construction, par étapes incrémentales. La documentation complète
> (installation, configuration, utilisation) arrivera avec la première version
> utilisable. Les décisions d'architecture sont tracées dans [`docs/adr/`](docs/adr/).

## Démarrage

```bash
uv sync                      # dépendances Python (versions verrouillées)
./scripts/init-env.sh        # .env local + secrets générés
./scripts/fetch-models.sh    # modèles GGUF vérifiés par empreinte (~3,4 Go)
docker compose up -d         # llm, embed, rerank, qdrant, api, interface
```

Indexer le corpus placé dans `data/`, puis ouvrir l'interface :

```bash
uv run python -m rag_source.ingest data   # indexation (incrémentale)
open http://127.0.0.1:8080
```

Vérifier la pile :

```bash
docker compose ps
uv run pytest -m integration
```

### Interroger l'API directement

Le jeton est dans `.env` (l'interface web, elle, n'en a pas besoin : Caddy le
fournit pour elle).

```bash
TOKEN=$(grep RAG_SOURCE_API_TOKEN .env | cut -d= -f2)

curl -s localhost:8000/health -H "Authorization: Bearer $TOKEN"

curl -s -X POST localhost:8000/v1/ask \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"question": "Votre question ?"}'
```

Documentation interactive de l'API : <http://localhost:8000/docs>.

Autres outils :

```bash
uv run python -m rag_source.ingest.report data --chunks   # ce que l'ingestion extrait
uv run python -m rag_source.eval                          # qualité de la recherche
```

### Profils de modèle

`./scripts/fetch-models.sh medium` (7B) ou `large` (14B), puis renseigner
`RAG_SOURCE_LLM_FILE` dans `.env`. Voir `models.lock`.

### Mémoire Docker

Les quatre services cohabitent dans la VM de Docker. Avec moins de 6 Gio alloués,
le noyau peut tuer un conteneur quand les trois modèles travaillent en même temps
(symptôme : un service qui redémarre seul en pleine indexation). Augmenter la
mémoire dans Docker Desktop, ou exécuter le LLM nativement — ci-dessous.

### macOS : LLM natif (Metal)

Docker n'accède pas au GPU sur macOS. Pour des réponses nettement plus rapides :

```bash
brew install llama.cpp
llama-server --model models/qwen2.5-3b-instruct-q4_k_m.gguf --port 8081 --ctx-size 8192
docker compose -f compose.yaml -f compose.native-llm.yaml up -d
```

## Déploiement sur un serveur

Un profil Scaleway (France) est fourni dans [`deploy/scaleway/`](deploy/scaleway/) :
instance provisionnée par OpenTofu, données sur un volume persistant, aucun port
applicatif ouvert — l'accès passe par un tunnel SSH — et LLM local imposé.

```bash
cd deploy/scaleway
cp env.example .env && source .env            # identifiants Scaleway
cp terraform.tfvars.example terraform.tfvars  # clé SSH, adresse autorisée
./scripts/up.sh
```

Voir [`deploy/scaleway/README.md`](deploy/scaleway/README.md).

## Développement

```bash
uv run pre-commit install  # hooks lint / format / typage
uv run pytest              # tests unitaires (rapides, sans réseau)
uv run pytest -m integration  # tests sur la pile Docker démarrée
```
