# ADR 0010 — Déploiement Scaleway, autonome et sans exposition

- Statut : accepté
- Date : 2026-09-18

## Contexte

L'infrastructure d'origine visait Hetzner et dépendait d'un module OpenTofu privé
(`git::ssh://git@gitlab.com/llm_tests/iac-modules`), ainsi que d'assets publiés
dans les paquets privés d'un projet GitLab. Sans accès à ce dépôt, `tofu init`
échoue : l'infrastructure n'était pas reproductible pour qui n'était pas déjà dans
l'organisation.

Son `teardown.sh` détruisait la VM — et avec elle la base vectorielle, les
documents et les modèles. La sauvegarde était une suggestion de commande dans le
README, dont le chemin était d'ailleurs faux.

## Décisions

1. **Aucune dépendance externe.** Le cloud-init est versionné dans le dépôt, les
   modèles viennent de dépôts publics vérifiés par empreinte, et la configuration
   tient dans quatre fichiers lisibles. `tofu validate` passe sans accès privilégié.
2. **Scaleway, zone `fr-par-*`**, imposée par une validation de variable : ce profil
   existe pour une juridiction précise.
3. **Les données vivent sur un volume séparé**, monté sur `/data`, protégé par
   `prevent_destroy`. `down.sh` détruit l'instance et conserve le volume ;
   `down.sh --all` supprime tout, après avoir fait taper « supprimer ». Éteindre
   une machine et perdre ses données sont deux décisions différentes.
4. **Rien n'est publié sur Internet.** Le groupe de sécurité n'ouvre que SSH, depuis
   une adresse déclarée. L'interface s'atteint par un tunnel SSH — donc pas de
   certificat TLS à gérer, pas de mot de passe applicatif exposé, et rien à scanner.
5. **Aucun secret dans l'état OpenTofu.** Les identifiants Scaleway restent dans
   l'environnement (`SCW_ACCESS_KEY`, `SCW_SECRET_KEY`, `SCW_DEFAULT_PROJECT_ID`) ;
   le jeton de l'API et la clé Qdrant sont générés par cloud-init sur la VM, au
   premier démarrage, et jamais régénérés ensuite.
6. **La souveraineté est imposée ici, pas ailleurs** : `RAG_SOURCE_REQUIRE_LOCAL_LLM=true`
   est posé par cloud-init et par l'override de production. Le poste de
   développement reste libre (cf. ADR 0003).
7. **Sauvegarde par instantané Qdrant**, pas par copie de fichiers : copier une base
   pendant qu'elle écrit produit une sauvegarde incohérente.

## Vérifications

Sans déployer — aucun identifiant Scaleway n'est nécessaire pour cela :

| Contrôle | Résultat |
|---|---|
| `tofu validate` | configuration valide |
| `tofu fmt` | conforme |
| Rendu du cloud-init (`templatefile`) | YAML valide : 3 fichiers, 11 commandes |
| Souveraineté dans le `.env` généré | `RAG_SOURCE_REQUIRE_LOCAL_LLM=true` |
| Jetons générés sur la VM | 2 (`openssl rand`), aucun dans l'état |
| `docker compose -f compose.yaml -f compose.prod.yaml config` | valide |
| `bash -n` sur les cinq scripts | syntaxe correcte |
| `git check-ignore` sur `.env`, `tfstate`, `tfvars` | tous ignorés |

## Conséquences

Le déploiement réel reste à faire : il demande des identifiants Scaleway et
engage des frais. Tout ce qui pouvait être vérifié sans dépenser l'a été ; ce qui
ne peut l'être qu'en conditions réelles — durée de cloud-init, débit de
téléchargement des modèles, vitesse du 7B sur ce gabarit — est explicitement en
attente.

Limite assumée : la mémoire reste le facteur dimensionnant, comme en local. Un
`PRO2-S` (16 Go) tient le profil `medium` (7B) avec les trois modèles et l'index ;
un gabarit plus petit ne tiendra pas, et le fichier de variables le dit.
