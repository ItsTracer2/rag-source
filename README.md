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
docker compose up -d         # llm, embed, rerank, qdrant
```

Vérifier la pile :

```bash
docker compose ps
uv run pytest -m integration
```

Inspecter ce que l'ingestion tire du corpus placé dans `data/` :

```bash
uv run python -m rag_source.ingest.report data --chunks
```

### Profils de modèle

`./scripts/fetch-models.sh medium` (7B) ou `large` (14B), puis renseigner
`RAG_SOURCE_LLM_FILE` dans `.env`. Voir `models.lock`.

### macOS : LLM natif (Metal)

Docker n'accède pas au GPU sur macOS. Pour des réponses nettement plus rapides :

```bash
brew install llama.cpp
llama-server --model models/qwen2.5-3b-instruct-q4_k_m.gguf --port 8081 --ctx-size 8192
docker compose -f compose.yaml -f compose.native-llm.yaml up -d
```

## Développement

```bash
uv run pre-commit install  # hooks lint / format / typage
uv run pytest              # tests unitaires (rapides, sans réseau)
uv run pytest -m integration  # tests sur la pile Docker démarrée
```
