# ADR 0002 — Un chargeur par format, et extraction structurée

- Statut : accepté
- Date : 2026-09-17

## Contexte

L'implémentation d'origine lisait uniquement les `*.pdf` présents à la racine de
`data/`, page par page, via `PyPDFLoader`, puis « nettoyait » le texte avec deux
expressions régulières sur les retours à la ligne.

Sur le corpus réel, cela donnait **zéro document indexé** : les PDF sont dans un
sous-dossier, et les fichiers Markdown et Excel n'étaient pas gérés du tout.

## Décisions

1. **Registre de chargeurs** (`@register_loader`) : un module par format, une
   interface commune, et un modèle de sortie unique (`Section`). Ajouter DOCX ou
   HTML plus tard ne touchera aucun autre fichier du pipeline.
2. **Parcours récursif**, ordre stable, fichiers cachés et verrous Office ignorés.
3. **Un échec n'arrête pas le traitement** : il est consigné dans un rapport
   (`CorpusReport`), consultable via `python -m rag_source.ingest.report`.
4. **PDF (PyMuPDF)** : titres détectés par le style (taille et graisse relatives au
   corps de texte), tableaux extraits en Markdown, en-têtes et pieds de page
   retirés par répétition, sommaires imprimés et pages de garde écartés, titres
   coupés en deux lignes recollés. Les sections traversent les sauts de page.
5. **Markdown (markdown-it-py)** : découpage par titres ; un `#` dans un bloc de
   code n'est pas un titre, ce qu'une expression régulière manquerait.
6. **Excel (openpyxl)** : une ligne de recommandation devient une section
   indivisible, avec son identifiant (`R24`) en métadonnée. Les feuilles de prose
   (préambule, mode d'emploi) sont conservées comme texte.

## Conséquences

Sur le corpus de référence : 17 documents, 934 sections, environ 618 000
caractères, aucun échec — contre zéro document auparavant.

La détection de titres par le style reste une heuristique. Elle est couverte par
des tests sur des PDF fabriqués (titres, en-tête répété, sommaire, page de garde,
tableau, titre sur deux lignes), ce qui permet de la faire évoluer sans régression.

PyMuPDF est sous licence AGPL. Choix assumé pour un usage personnel ou interne :
c'est la bibliothèque qui restitue le mieux la structure. `pypdf` (BSD) reste le
repli si le projet devait être distribué autrement.
