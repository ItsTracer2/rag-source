# ADR 0011 : Une ligne de commande, cliente de l'API

- Statut : accepté
- Date : 2026-09-18

## Contexte

Le projet d'origine exposait cinq scripts autonomes (`chat.py`, `query_data.py`,
`populate_database.py`, `inspect_db.py`, `generate_tests.py`), chacun avec ses
propres arguments et sa propre façon de se connecter aux modèles. Il fallait se
placer dans le bon dossier pour que les chemins relatifs fonctionnent, et
`chat.py` s'arrêtait à la première exception.

Pendant les étapes précédentes, ce projet avait accumulé ses propres points
d'entrée temporaires (`python -m rag_source.ingest`, `.ingest.report`, `.eval`) :
commodes pour avancer, mais ce n'est pas une interface.

## Décisions

1. **Une seule commande, `rag-source`**, avec des sous-commandes. Les trois points
   d'entrée temporaires sont supprimés, pas dépréciés : le projet n'a pas encore
   d'utilisateurs à ménager, et deux chemins pour la même chose se contredisent tôt
   ou tard.
2. **Deux familles, assumées** :
   - `health`, `docs`, `ask`, `search` parlent à l'**API** par HTTP, comme
     l'interface web. Une seule implémentation de la recherche et de la génération,
     un seul endroit où corriger un défaut, et ces commandes fonctionnent à
     travers un tunnel SSH, sans accès aux fichiers ;
   - `ingest`, `inspect`, `eval` travaillent **en local**, parce qu'elles lisent le
     corpus sur disque.
3. **Codes de retour utilisables** : `0` succès, `1` résultat vide ou service
   dégradé, `2` erreur. `rag-source health` devient ainsi une sonde de supervision,
   et `rag-source ask` peut servir dans un script.
4. **`--json` partout** où une sortie machine a un sens.
5. **Aucune dépendance supplémentaire** : `argparse` et `httpx`, déjà présents.
   Click ou Typer auraient apporté du confort d'écriture pour un coût de
   dépendance et une couche d'indirection de plus.
6. **Les couleurs s'éteignent hors terminal** : un pipe ou un fichier de journal ne
   doit pas se remplir de codes d'échappement.

## Conséquences

Les erreurs deviennent des messages, pas des traces d'appels : une API injoignable
affiche « démarrer la pile : docker compose up -d », un jeton refusé renvoie à
`.env`.

`rag-source ask` affiche les sources avant la réponse, puis les citations
vérifiées. Sur le corpus de référence, une question répondable prend environ 6 s
de bout en bout avec le LLM natif.

Limite assumée : `ingest` et `eval` ont besoin des services (embedding, reranking,
base) joignables depuis la machine qui les lance. Sur le serveur, on les exécute
donc dans le conteneur de l'API (`docker compose exec api rag-source ingest /data`),
là où ces adresses sont les bonnes.
