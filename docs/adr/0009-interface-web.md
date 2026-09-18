# ADR 0009 — Interface web sans framework, jeton porté par le proxy

- Statut : accepté
- Date : 2026-09-18

## Contexte

L'implémentation d'origine n'avait qu'une boucle `input()` en ligne de commande,
dont la première exception mettait fin à la session. Son README annonçait « pas
encore d'interface graphique web ; cible prochaine itération : Streamlit ou Gradio ».

## Décisions

1. **HTML, CSS et JavaScript, sans framework ni étape de compilation.** Trois
   fichiers lisibles tels quels, aucune dépendance à installer ou à mettre à jour,
   rien à reconstruire pour déployer. Streamlit ou Gradio auraient ajouté un second
   runtime Python et brouillé la frontière avec l'API ; ici l'interface est un
   client comme un autre, au même titre que la future ligne de commande.
2. **Le jeton est porté par Caddy**, pas par le navigateur. La page ne stocke aucun
   secret : il n'y a rien à voler dans son code source ni dans le `localStorage`.
   En contrepartie, quiconque atteint le port atteint l'API — d'où la publication
   sur `127.0.0.1` uniquement et, à distance, un tunnel SSH.
3. **Tout passe par le flux SSE.** Les sources s'affichent dès que la recherche a
   conclu (1,6 s), la réponse s'écrit ensuite. Sur CPU, l'alternative serait un
   écran figé pendant une dizaine de secondes.
4. **Le mode de souveraineté est affiché en permanence**, pas enfoui dans un menu :
   « 100 % local » ou « ⚠ LLM externe », lu depuis `/health`.
5. **Les `[n]` de la réponse deviennent des renvois cliquables** vers le passage
   correspondant, et un numéro invalide s'affiche en rouge. La vérification des
   citations n'a d'intérêt que si elle se voit.
6. **`/v1/documents`** alimente un filtre par document, et sert aussi à vérifier
   d'un coup d'œil ce qui est réellement indexé.

## Conséquences

La pile complète tient en une commande : `docker compose up -d`, puis
`http://127.0.0.1:8080`. L'image de l'API pèse 570 Mo, sans PyTorch — les modèles
vivent dans les conteneurs llama.cpp.

Mesuré à travers le proxy, avec le LLM natif : passages à 1,6 s, premier token à
5,1 s, réponse complète à 5,6 s.

L'interface reste volontairement simple : une conversation, un filtre, un choix de
mode de recherche. Elle n'a ni comptes, ni conversations sauvegardées, ni envoi de
documents — autant de fonctions qui appelleraient une base de données et un modèle
d'autorisation, c'est-à-dire un autre projet.

Le fournisseur `anthropic` de la configuration devient `external` : l'étape qui
devait lui donner un client dédié est abandonnée, et presque tous les services
exposent de toute façon une API compatible OpenAI. Mieux vaut nommer ce qui existe
que promettre ce qui n'existera pas.
