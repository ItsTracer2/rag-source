# ADR 0005 : Un seul runtime de modèles, et des versions épinglées

- Statut : accepté
- Date : 2026-09-17

## Contexte

L'implémentation d'origine faisait cohabiter deux mondes : `llama.cpp` pour le chat,
et `sentence-transformers` (donc PyTorch) pour l'embedding et le reranking. Sur une
machine sans GPU, `pip` installait quand même la version CUDA de PyTorch, plusieurs
gigaoctets inutiles, 5 à 15 minutes d'installation, et un pic de mémoire qui faisait
tuer le processus par le noyau sur une VM de 8 Go.

Les modèles eux-mêmes n'étaient épinglés nulle part : `bge-m3` était téléchargé
« à la dernière version », depuis un dépôt privé pour le LLM.

## Décisions

1. **llama.cpp pour les trois rôles.** Trois conteneurs, la même image, des
   arguments différents : `--model` pour la génération, `--embeddings` pour
   bge-m3, `--reranking` pour bge-reranker-v2-m3. Plus aucun PyTorch dans le projet.
2. **Modèles épinglés par empreinte** dans `models.lock` (rôle, profil, dépôt,
   fichier, sha256, taille). `scripts/fetch-models.sh` refuse tout fichier dont
   l'empreinte diffère, et ne retélécharge rien inutilement. Aucun jeton requis :
   tous les dépôts sont publics, contrairement au GitLab privé d'origine.
3. **Trois profils de LLM** : `small` (3B, 2,1 Go), `medium` (7B, 4,7 Go),
   `large` (14B, 9,0 Go), pour monter en qualité sans sortir les données.
4. **Embedding et reranking en Q8_0** : ces modèles sont petits, et une
   quantification agressive dégraderait directement la pertinence de la recherche.
   Le LLM est en Q4_K_M, compromis habituel entre taille et qualité.
5. **Qdrant avec clé d'API obligatoire**, générée par `scripts/init-env.sh`. Une
   clé vide n'est pas « pas d'authentification » : Qdrant refuse alors toute
   écriture avec un 401 impossible à diagnostiquer. Compose échoue désormais avec
   un message clair si le `.env` n'a pas été initialisé.
6. **Aucun port exposé au-delà de 127.0.0.1.**
7. **Override `compose.native-llm.yaml`** : sur macOS, Docker n'accède pas au GPU.
   Lancer `llama-server` nativement profite de Metal, plusieurs fois plus vite.

## Conséquences

Mesures sur la pile locale (Mac M2, Docker, conteneurs CPU) :

| Vérification | Résultat |
|---|---|
| Dimension des vecteurs bge-m3 | 1024 |
| Similarité « nettoyer le plateau ? » FR vs EN | 0,75 : l'interrogation multilingue fonctionne |
| Reranking : passage pertinent vs bruit | +3,7 contre −8,2 et −11,0 |
| Génération 3B, 34 tokens | ~3,3 s |
| Démarrage complet de la pile | < 60 s, modèles déjà téléchargés |

Ces contrôles sont désormais automatisés : `uv run pytest -m integration`.

Coût assumé : trois conteneurs de modèles au lieu d'un processus Python. C'est le
prix de l'isolement des rôles : on peut remplacer le reranker sans toucher au reste,
et déplacer un service sur une autre machine sans changer une ligne de code.
