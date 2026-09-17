# ADR 0004 — Découpage guidé par la structure

- Statut : accepté
- Date : 2026-09-17
- Remplace : le découpage sémantique (`SemanticChunker`) de l'implémentation d'origine

## Contexte

L'implémentation d'origine proposait deux découpages : `SemanticChunker` (défaut) et
un découpage récursif à 1 000 caractères. Le premier vectorise chaque phrase pour
mesurer les ruptures de sens, puis les chunks sont vectorisés à nouveau à
l'indexation. Il ignorait les titres, n'imposait aucune taille maximale, s'arrêtait
aux limites de page et dépendait de `langchain-experimental`.

## Décisions

1. **La section est l'unité de base.** Les chargeurs produisent déjà une structure
   (titres, tableaux, enregistrements) : la respecter coûte une seule passe
   d'embedding, à l'indexation, au lieu de deux.
2. **La taille se mesure en tokens** avec le tokenizer du modèle d'embedding
   (`tokenizers`), et une estimation par caractères prend le relais tant que le
   fichier n'est pas téléchargé. Cible 400 tokens, maximum 700, recouvrement 15 %
   entre deux morceaux d'une même section.
3. **Chaque chunk porte son fil d'Ariane** : titre du document puis titres
   englobants. Le texte vectorisé devient auto-suffisant, et la citation affichable
   telle quelle.
4. **Tableaux et enregistrements sont atomiques**, et lorsqu'un découpage est
   inévitable, l'en-tête du tableau ou la ligne d'identifiant est répétée dans
   chaque morceau.
5. **Les sections trop courtes sont regroupées** avec leurs voisines de même titre.
6. **Identifiants déterministes** : `uuid5(sha256 du fichier, rang)`. Deux
   indexations identiques donnent les mêmes identifiants ; un fichier modifié donne
   des identifiants entièrement différents, ce qui permet à l'étape 5 de remplacer
   ses chunks au lieu de laisser cohabiter deux versions. Les identifiants
   positionnels d'origine (`source:page:index`) provoquaient exactement ce défaut :
   le contenu changeait, l'identifiant non, et l'ancien chunk restait indexé.

## Conséquences

Corpus de référence : 941 sections donnent 1 160 chunks — médiane 136 tokens,
moyenne 248, maximum 694 (prose 429, tableaux 153, enregistrements 578). Aucun
dépassement de la limite, vérifiable par `python -m rag_source.ingest.report data --chunks`.

La médiane est basse parce que les enregistrements (une recommandation, une
référence de pièce) sont courts par nature : c'est le comportement recherché, un
enregistrement complet vaut mieux qu'un enregistrement dilué.

Deux garde-fous sont nés de l'épreuve du corpus réel : une ligne de tableau plus
longue que la limite est recoupée aux mots, et le recouvrement est réduit quand il
ferait déborder le chunk suivant. Sans eux, quelques chunks dépassaient la limite et
auraient été tronqués silencieusement par le modèle d'embedding.

Le découpage sémantique n'est pas conservé en option : il coûtait cher, dépendait
d'un paquet expérimental, et le gain supposé est déjà obtenu par la structure du
document, qui est une information exacte plutôt qu'une estimation.
