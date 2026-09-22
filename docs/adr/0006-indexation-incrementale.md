# ADR 0006 : Indexation incrémentale, et deux vecteurs par chunk

- Statut : accepté
- Date : 2026-09-17

## Contexte

L'implémentation d'origine identifiait ses chunks par leur position
(`document.pdf:6:2`), puis n'ajoutait que les identifiants absents de la base.
Conséquences, toutes silencieuses :

- un document modifié gardait ses anciens chunks : le système répondait avec du
  contenu périmé, en le citant comme actuel ;
- un document supprimé restait indexé indéfiniment ;
- le modèle d'embedding n'était inscrit nulle part : changer de modèle mélangeait
  des vecteurs incomparables sans le moindre avertissement.

La recherche, elle, était purement sémantique.

## Décisions

1. **L'index est sa propre mémoire.** Chaque chunk porte l'empreinte sha256 de son
   fichier d'origine. Comparer cette empreinte à celle du disque suffit à classer
   chaque document : inchangé, nouveau, modifié, disparu. Aucun fichier d'état
   annexe qui pourrait diverger de la réalité.
2. **L'empreinte est calculée avant d'ouvrir le document.** Analyser un PDF pour
   découvrir ensuite qu'il n'a pas bougé est du travail perdu : sur le corpus de
   référence, un passage sans changement passe de 26 secondes à 0,2 seconde.
3. **Un document modifié voit ses chunks supprimés avant réécriture**, par filtre
   sur `source`. Les identifiants dérivant de l'empreinte du fichier, une simple
   réécriture laisserait cohabiter deux versions.
4. **Deux vecteurs par chunk** : `dense` (bge-m3, sémantique) et `sparse` (BM25,
   mots-clés). Le dense sait qu'« entretenir le plateau » ressemble à « nettoyer le
   plateau tournant » ; le creux retrouve « E01 », « R24 » ou « MW-104 », que la
   recherche sémantique situe mal. Le client envoie des fréquences saturées, Qdrant
   applique l'IDF (`modifier: idf`) car lui seul connaît le corpus entier.
5. **La dimension est vérifiée** avant toute écriture : une collection créée avec un
   autre modèle fait échouer l'indexation avec un message explicite.
6. **Client Qdrant écrit à la main, sur l'API REST.** Le client officiel tire gRPC,
   numpy et portalocker pour une quinzaine d'appels HTTP. L'indexation dépend du
   protocole `ChunkStore`, pas de Qdrant : elle se teste sans conteneur, et un autre
   moteur se brancherait sans toucher au pipeline.

## Conséquences

Mesures sur le corpus de référence (17 documents, Mac M2, conteneurs CPU) :

| Opération | Durée |
|---|---|
| Indexation complète (1 114 chunks) | 369 s |
| Passage sans changement | 0,2 s |
| Un document modifié sur 17 | 32 s |
| Suppression d'un document | 0,1 s |

Cycle vérifié de bout en bout : 1 114 chunks, puis 1 115 après ajout d'une section,
1 041 après suppression du fichier, et de nouveau 1 114 après restauration.

Limite connue : `indexed_versions()` parcourt la collection pour reconstruire la
liste des documents indexés. C'est immédiat sur des milliers de chunks, mais il
faudra passer à l'API de facettes de Qdrant sur des centaines de milliers.

Contrainte matérielle rencontrée : Docker ne disposait que de 3,8 Gio, et les
serveurs d'embedding et de reranking réservaient 8192 tokens de contexte multipliés
par quatre créneaux parallèles, et le conteneur se faisait tuer en pleine indexation.
Les contextes sont ramenés à 2048 avec un seul créneau, largement suffisant pour des
chunks d'au plus 700 tokens.
