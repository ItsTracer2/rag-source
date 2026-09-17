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

## Développement

```bash
uv sync                    # dépendances (versions verrouillées)
uv run pre-commit install  # hooks lint / format / typage
uv run pytest              # tests unitaires
```
