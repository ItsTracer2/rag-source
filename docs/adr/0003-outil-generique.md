# ADR 0003 : Outil générique, souveraineté propre au déploiement

- Statut : accepté
- Date : 2026-09-17
- Complète : [ADR 0001](0001-socle-du-projet.md), [ADR 0002](0002-chargeurs-par-format.md)

## Contexte

Le corpus de développement (documents réglementaires français) a servi de banc
d'essai, mais RAG-Source doit traiter n'importe quelle documentation : une notice
de four à micro-ondes en anglais, un manuel technique sur deux colonnes, un export
CSV de pièces détachées, une page web enregistrée.

Deux réglages trahissaient l'origine du projet : des heuristiques accordées à un
seul type de document, et une contrainte de souveraineté imposée partout.

## Décisions

1. **Formats** : PDF, Markdown, Excel, CSV/TSV, texte brut, HTML et DOCX. Le
   registre de chargeurs rend l'ajout d'un format local à un seul fichier.
2. **Ordre de lecture sur plusieurs colonnes** : détection d'une gouttière par
   densité de mots par ligne, puis extraction colonne par colonne. Sans cela, une
   notice sur deux colonnes produit un texte entrelacé, donc inexploitable. Le
   défaut était invisible sur le corpus de développement, entièrement sur une
   colonne.
3. **OCR intégré** (Tesseract via PyMuPDF), en mode `auto` : seules les pages sans
   couche texte sont reconnues. Son absence n'est jamais fatale, elle est signalée.
4. **Identifiants tabulaires multilingues** : `Identifiant`, `Référence`, `ID`,
   `Code`, `Part number`, `SKU`… avec repli sur la forme des valeurs.
5. **Souveraineté propre au déploiement** : `RAG_SOURCE_REQUIRE_LOCAL_LLM` vaut
   `false` par défaut et `true` dans le profil Scaleway. Le mode effectif
   (`local` ou `external`) reste toujours calculé et affiché.

## Conséquences

Interroger une documentation publique avec un LLM externe ne demande plus aucun
contournement ; héberger un corpus sensible sur Scaleway garde un refus de
démarrage explicite. Le niveau d'exigence suit l'usage, au lieu d'être figé.

Les heuristiques de mise en page restent des heuristiques. Elles sont couvertes par
des tests délibérément hors du corpus d'origine : notice anglaise, deux colonnes,
page scannée, export CSV, page web, document Word.

Tesseract devient une dépendance système (installée dans l'image Docker à l'étape
suivante ; `brew install tesseract` en local). Les paquets de langue conditionnent
la qualité de l'OCR : `eng` est fourni par défaut, `RAG_SOURCE_OCR_LANGUAGES` permet
d'en combiner d'autres.
