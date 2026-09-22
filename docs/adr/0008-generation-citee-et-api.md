# ADR 0008 : Génération citée et API HTTP

- Statut : accepté
- Date : 2026-09-18

## Contexte

Le prompt d'origine collait les passages bout à bout, séparés par des tirets, sans
indiquer leur provenance :

```
{context}
---
Question: {question}
```

Le modèle ne *pouvait pas* citer ses sources, même en le voulant : elles ne lui
étaient pas données. Aucune consigne ne l'autorisait à dire qu'il ne savait pas, et
rien ne vérifiait après coup ce qu'il avançait. L'interface se limitait à une boucle
`input()` en ligne de commande, dont la première exception mettait fin à la session.

## Décisions

1. **Passages numérotés et situés** dans le prompt : `[1] (manuel.pdf, p. 4 —
   Entretien › Plateau)`. C'est ce qui rend la citation possible, puis vérifiable.
2. **Le modèle a le droit de ne pas savoir**, et la consigne le dit explicitement.
3. **Les citations sont vérifiées après génération** : les `[n]` sont extraits, ceux
   qui ne désignent aucun passage fourni sont signalés (`invalid_citations`). Une
   consigne de prompt n'est pas une garantie.
4. **Sans passage pertinent, le modèle n'est pas appelé.** La recherche a déjà
   conclu qu'il n'y avait rien ; lui demander de répondre quand même, c'est
   l'inviter à inventer. Cela épargne aussi une génération complète.
5. **API HTTP d'abord** (FastAPI) : `/v1/ask`, `/v1/ask/stream` (SSE), `/v1/search`
   et `/health`. L'interface web de l'étape suivante sera un simple client.
6. **Jeton Bearer obligatoire**, comparé à temps constant. Sans jeton configuré,
   l'API refuse de servir : une API ouverte serait pire qu'une API inutilisable.
7. **`/health` expose le mode de souveraineté**, pour qu'une interface puisse
   afficher en permanence si les extraits du corpus quittent l'hôte.

## Le prompt a été choisi par la mesure

Une consigne raisonnable (« cite tes sources entre crochets ») ne suffit pas avec un
modèle de 3 milliards de paramètres. Sur huit questions du jeu de référence :

| Formulation | réponses citées |
|---|---|
| « Cite tes sources avec leur numéro entre crochets » | 5 / 8 |
| « CHAQUE phrase doit se terminer par son numéro » + exemple | **8 / 8** |

Même coût de génération, taux de citation multiplié. Le prompt retenu est le second.
Ce genre d'écart ne se devine pas : il se mesure, comme les réglages de recherche.

## Mesures

Pile locale, LLM natif (Metal), corpus de 1 114 chunks :

| Opération | Durée |
|---|---|
| `/v1/ask`, question répondable | 12 s |
| Flux : passages affichés | 1,6 s |
| Flux : premier token | 2,9 s |
| Flux : réponse complète | 5,3 s |
| Question hors corpus (refus, sans appel au modèle) | 16 s |

Le flux change la nature de l'attente : les sources s'affichent en 1,6 s, la réponse
s'écrit ensuite. Sans lui, l'interface resterait muette une dizaine de secondes.

## Conséquences

Le LLM en conteneur se faisait tuer par manque de mémoire dès que les trois modèles
travaillaient ensemble (3,8 Gio pour toute la VM Docker). Exécuté nativement avec
Metal (le chemin déjà prévu par `compose.native-llm.yaml`), il passe de ~10 à
**39 tokens/s** et libère 1,2 Gio. C'est désormais le mode recommandé sur macOS, et
le README le dit.

Limite assumée : la vérification des citations contrôle qu'un numéro cité existe,
pas que la phrase citée est fidèle au passage. Vérifier la fidélité demande un
modèle juge, donc du temps et de la prudence : ce sera une option d'évaluation, pas
un contrôle à chaud.
