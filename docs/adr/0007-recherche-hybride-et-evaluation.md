# ADR 0007 : Recherche hybride, reranking, et mesure avant réglage

- Statut : accepté
- Date : 2026-09-17

## Contexte

L'implémentation d'origine faisait une recherche vectorielle seule (k=20), puis un
reclassement par `ms-marco-MiniLM-L-6-v2` (un modèle entraîné uniquement sur de
l'anglais, appliqué à un corpus français) et gardait les 4 premiers passages, sans
seuil ni budget de contexte. Sa seule évaluation demandait à un LLM de 3 milliards
de paramètres de juger ses propres réponses : lent, coûteux, peu fiable, et surtout
incapable de dire *où* le système échoue.

## Décisions

1. **Recherche hybride**, dense (bge-m3) et mots-clés (BM25), fusionnées par Qdrant
   avec RRF. La fusion combine les *rangs* et non les scores : un cosinus et un
   score BM25 ne sont pas comparables, leurs rangs si.
2. **Reranking multilingue** par `bge-reranker-v2-m3`, de la même famille que
   l'embedding.
3. **Seuil de pertinence** sur le score du reranker : si même le meilleur passage
   est jugé hors sujet, la recherche ne renvoie rien.
4. **Budget de contexte** en tokens, pour que la sélection tienne dans la fenêtre
   du LLM.
5. **Banc d'évaluation déterministe, sans LLM** : hit@k, recall@k, MRR, et deux
   mesures d'abstention. Le jeu de référence compte 30 questions écrites à la main à
   partir du corpus (identifiants exacts, questions reformulées, tableaux,
   acronymes, plusieurs formats), dont 3 sans réponse dans le corpus.

## Mesures

Corpus de référence, 30 questions, 1 114 chunks, Mac M2 (conteneurs CPU) :

| Mode | hit@6 | recall | MRR | abstention utile | abstention à tort | durée |
|---|---|---|---|---|---|---|
| dense seul | 100 % | 95 % | 0,89 | 0 % | 0 % | 3,3 s |
| BM25 seul | 96 % | 92 % | 0,87 | 0 % | 0 % | 0,5 s |
| hybride | 96 % | 92 % | 0,86 | 0 % | 0 % | 1,8 s |
| hybride + reranking | 96 % | 94 % | 0,94 | 100 % | 4 % | 837 s |
| hybride + reranking **réglé** | **100 %** | **96 %** | **0,98** | **100 %** | **0 %** | 242 s |

La dernière ligne est la configuration retenue : 12 candidats reclassés, seuil de
pertinence à -5. Les deux réglages ont été choisis par la mesure, détail ci-dessous.

Lecture honnête de ces chiffres :

- **Le reranking n'améliore pas le taux de succès**, il améliore le *classement*
  (MRR 0,86 → 0,94) : la bonne source arrive en tête, pas en quatrième position.
  C'est ce qui compte quand on ne garde que quelques passages pour le LLM.
- **Son apport décisif est l'abstention** : c'est le seul mode qui refuse les
  questions hors corpus (100 % contre 0 %). Sans lui, « quelle est la recette du
  gratin dauphinois ? » renvoie trois passages sur la segmentation réseau et la
  conservation de signatures électroniques, et un LLM nourri de ces passages
  produira une réponse, fausse. Le reranking et le seuil ne servent pas à mieux
  trouver, ils servent à **savoir quand il n'y a rien à trouver**.
- **BM25 seul tient la comparaison** (96 % contre 100 %) pour un coût sept fois
  moindre que le dense : sur un corpus normatif, riche en identifiants et en
  vocabulaire répété, les mots-clés suffisent souvent. Le garder coûte peu et sauve
  les recherches d'identifiants exacts, que le dense situe mal.
- Les écarts de 4 points correspondent à **une seule question** sur 30 : ils ne
  départagent rien à eux seuls.

### Deux réglages, deux mesures

**Combien de candidats reclasser ?**

| Candidats | hit@6 | recall | MRR | abstention utile | durée |
|---|---|---|---|---|---|
| 30 | 96 % | 94 % | 0,94 | 100 % | 837 s |
| 12 | 96 % | 93 % | 0,94 | 100 % | **358 s** |

Même qualité, temps divisé par 2,3 : le défaut est fixé à **12**. L'intuition pousse
à reclasser large « au cas où » ; la mesure dit que c'est payer sans rien gagner.

**Où placer le seuil de pertinence ?**

Le seuil initial, 0, reposait sur une idée fausse : « un score positif signale un
passage pertinent ». Les scores mesurés sur bge-reranker-v2-m3 racontent autre chose.

| Question | meilleur passage | passages hors sujet |
|---|---|---|
| « comment entretenir le plateau ? » | **-1,98** | -10,8 et -11,0 |
| « how long is the warranty? » | **-0,16** | -10,7 et -11,0 |
| « que faire si la porte reste ouverte ? » | **-2,34** | -11,0 et -11,0 |
| « quelle est la recette du gratin dauphinois ? » | -11,02 | -11,0 et -11,0 |

La séparation est franche, environ neuf points, mais elle ne passe pas par zéro.
Un seuil à 0 rejetait donc de bonnes réponses. Placé à **-5**, au milieu de l'écart
mesuré, il conserve 100 % d'abstention sur les questions hors corpus et supprime
l'abstention à tort :

| Seuil | hit@6 | recall | MRR | abstention utile | abstention à tort |
|---|---|---|---|---|---|
| 0 | 96 % | 93 % | 0,94 | 100 % | 4 % |
| **-5** | **100 %** | **96 %** | **0,98** | **100 %** | **0 %** |

## Un bug révélé par la mesure

Les premiers chiffres donnaient 85 à 89 % de réussite. L'examen des échecs a montré
que trois questions étaient comptées fausses alors que la bonne source figurait bien
dans les résultats : macOS enregistre les noms de fichiers en forme Unicode
**décomposée** (« é » = « e » + accent combinant), là où un fichier écrit à la main
utilise la forme composée. Les deux chaînes s'affichent à l'identique et ne sont pas
égales.

Les chemins de documents sont désormais normalisés en NFC à l'ingestion, en un seul
point (`relative_source`), et un test le vérifie. Sans le banc d'évaluation, ce
défaut serait resté invisible : les réponses auraient semblé correctes, mais tout
filtre par source aurait échoué en silence selon la machine.

## Conséquences

Le reranking reste le mode par défaut, pour l'abstention. Son coût sur CPU (environ
12 s par question avec 12 candidats) reste la principale limite du projet sur une
machine sans GPU. Un `llama-server` natif avec Metal, ou un GPU côté serveur, le
ramène à moins d'une seconde.

Le banc permet de trancher un réglage par la mesure plutôt qu'au jugé : `--mode`,
`--candidates`, `--min-score` et `--failures` comparent deux configurations en une
commande.

Limite assumée : 30 questions écrites par une seule personne ne constituent pas un
jeu statistiquement solide. Ces chiffres servent à comparer des configurations entre
elles sur le même jeu, pas à annoncer une performance absolue.
