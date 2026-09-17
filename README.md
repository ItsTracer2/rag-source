# RAG-Source

RAG souverain et auto-hébergé : réponses sourcées et citées sur un corpus
documentaire local (PDF, Markdown, Excel). Aucune donnée ne quitte l'hôte par défaut.

> 🚧 Projet en construction, par étapes incrémentales. La documentation complète
> (installation, configuration, utilisation) arrivera avec la première version
> utilisable. Les décisions d'architecture sont tracées dans [`docs/adr/`](docs/adr/).

## Développement

```bash
uv sync                    # dépendances (versions verrouillées)
uv run pre-commit install  # hooks lint / format / typage
uv run pytest              # tests unitaires
```
