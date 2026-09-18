# Décisions d'architecture

Une décision structurante par fichier : le contexte qui l'a rendue nécessaire, ce
qui a été choisi, et ce que cela coûte. Les fichiers ne sont pas réécrits après
coup — quand une décision en remplace une autre, une nouvelle ADR l'explique.

La plupart de ces décisions se comparent à celles du projet dont RAG-Source
s'inspire : cette comparaison est conservée, parce qu'une décision sans son
alternative n'apprend rien.

| # | Décision | Ce qu'elle remplace |
|---|---|---|
| [0001](0001-socle-du-projet.md) | Socle : `uv`, typage strict, configuration validée | dépendances non figées, `os.getenv` dispersés |
| [0002](0002-chargeurs-par-format.md) | Un chargeur par format, extraction structurée | PDF seuls, non récursif, structure perdue |
| [0003](0003-outil-generique.md) | Outil générique ; souveraineté propre au déploiement | heuristiques calées sur un seul corpus |
| [0004](0004-chunking-structurel.md) | Découpage guidé par la structure, mesuré en tokens | découpage sémantique, deux passes d'embedding |
| [0005](0005-pile-de-modeles-et-services.md) | Un seul runtime (llama.cpp), modèles épinglés | PyTorch CUDA sur machine sans GPU, dépôt privé |
| [0006](0006-indexation-incrementale.md) | Index incrémental, vecteurs dense + BM25 | identifiants positionnels, contenu périmé conservé |
| [0007](0007-recherche-hybride-et-evaluation.md) | Recherche hybride, reranking multilingue, banc de mesure | reranker anglophone, LLM juge de lui-même |
| [0008](0008-generation-citee-et-api.md) | Passages numérotés, citations vérifiées, API | prompt sans provenance, boucle `input()` |
| [0009](0009-interface-web.md) | Interface web sans framework, jeton porté par le proxy | aucune interface |
| [0010](0010-deploiement-scaleway.md) | Infrastructure autonome, données persistantes | module privé, teardown destructeur |
| [0011](0011-ligne-de-commande.md) | Une commande, cliente de l'API | cinq scripts autonomes |

## Écrire une ADR

Reprendre la trame commune : **Contexte** (le problème observé, pas la solution),
**Décisions** (numérotées, avec leur raison), **Mesures** quand il y en a,
**Conséquences** (y compris les coûts et les limites). Dater, et ne rien enjoliver
après coup.
