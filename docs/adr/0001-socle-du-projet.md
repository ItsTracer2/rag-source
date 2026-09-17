# ADR 0001 — Socle du projet

- Statut : accepté
- Date : 2026-09-17

## Contexte

RAG-Source répond à des questions sur un corpus documentaire (PDF, Markdown, Excel)
dont une partie peut être marquée Diffusion Restreinte. Deux exigences structurent
tout le reste :

1. **Souveraineté par défaut** : aucun extrait du corpus ne quitte l'hôte, sauf
   décision explicite et documentée.
2. **Explicabilité** : chaque composant doit pouvoir être compris et justifié ;
   on évite les dépendances qui masquent le fonctionnement.

## Décisions

| Sujet | Choix | Raison principale |
|---|---|---|
| Langage | Python 3.12 | Écosystème documentaire et ML ; typage moderne |
| Gestion du projet | `uv` + `pyproject.toml` + `uv.lock` | Installation reproductible et rapide, un seul outil |
| Qualité | `ruff`, `mypy --strict`, `pytest`, pre-commit, CI | Mêmes versions partout (hooks et CI passent par `uv run`) |
| Configuration | `pydantic-settings`, préfixe `RAG_SOURCE_` | Typée, validée au démarrage, une seule source de vérité |
| Orchestration RAG | Code maison, sans LangChain | Peu d'appels à des modèles ; on évite un large graphe de dépendances et des API instables |
| Modèles de domaine | `dataclasses` figées | Légères, immuables ; Pydantic reste réservé aux frontières (config, API) |

## Garde de souveraineté

`Settings` refuse de démarrer si le LLM configuré est externe, **ou** si un
fournisseur « local » pointe vers un hôte public, sauf si
`RAG_SOURCE_ALLOW_EXTERNAL_LLM=i-understand-data-leaves-the-host`.

La valeur d'acquittement est volontairement verbeuse : il ne s'agit pas d'un réglage
de qualité mais d'un changement de modèle de menace.

Limite connue : la détection d'hôte privé ne fait pas de résolution DNS. Un nom
public qui résoudrait vers une IP privée est traité comme externe. C'est un faux
positif sans danger, contournable par l'acquittement ou par l'usage d'une IP.
