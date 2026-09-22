# Évaluation de la recherche

Mesurer avant de régler. Ce banc compare des stratégies de recherche sur des
métriques déterministes, sans faire intervenir de LLM : les chiffres sont
reproductibles, rapides à obtenir, et disent *où* le système échoue.

```bash
uv run rag-source eval                                  # tous les modes
uv run rag-source eval --mode hybrid+rerank --failures  # détail des ratés
uv run rag-source eval --mode hybrid+rerank --candidates 12
```

Prérequis : la pile démarrée (`docker compose up -d`) et le corpus indexé
(`uv run rag-source ingest data`).

## Métriques

| Mesure | Question à laquelle elle répond |
|---|---|
| **hit@k** | La bonne source apparaît-elle dans les k premiers passages ? |
| **recall@k** | Quelle part des sources attendues est retrouvée ? |
| **MRR** | À quel rang arrive la première bonne source (1 ; 1/2 ; 1/3…) ? |
| **abstention utile** | Sur une question sans réponse dans le corpus, le système se tait-il ? |
| **abstention à tort** | S'est-il tu alors que la réponse existait ? |

Les deux dernières comptent autant que les premières : un RAG qui répond toujours
quelque chose est un RAG qui invente.

## Jeu de données

`datasets/corpus-reference.jsonl`, une question par ligne :

```json
{"id": "r24-passerelle", "question": "Que dit la recommandation R24 … ?",
 "expected_sources": ["anssi-oasis-passerelle_internet_securisee-v3.0.xlsx"],
 "tags": ["identifiant", "excel"]}
```

`expected_sources` vide signale une question **sans réponse dans le corpus** : le
comportement attendu est alors l'abstention.

Les questions sont écrites à la main à partir du contenu réel, et couvrent
volontairement des situations différentes : identifiants exacts (`R24`,
`DEV-FILT-APPL`), questions reformulées sans mot commun avec le texte, acronymes
cachés dans des tableaux, réponses réparties sur plusieurs documents, et formats
variés (PDF, Markdown, Excel).

## Écrire son propre jeu

Un jeu d'évaluation ne se transporte pas d'un corpus à l'autre : ces questions ne
valent que pour les documents de `data/`. Pour un autre corpus, créer un fichier
JSONL au même format. Une vingtaine de questions suffit pour comparer des
configurations entre elles.

**Limite à garder en tête** : sur 30 questions, un écart de 4 points correspond à
une seule question. Ces chiffres servent à comparer des configurations sur le même
jeu, pas à annoncer une performance absolue.
