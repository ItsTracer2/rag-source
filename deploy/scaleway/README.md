# Déploiement sur Scaleway

Profil serveur de RAG-Source : une instance Scaleway en France, où **rien n'est
publié sur Internet** et où le LLM est obligatoirement local.

Ce profil est séparé du profil local (`compose.yaml` à la racine) : même
application, contraintes différentes.

| | Poste de développement | Déploiement Scaleway |
|---|---|---|
| LLM | libre — externe autorisé | **local imposé** (`RAG_SOURCE_REQUIRE_LOCAL_LLM=true`) |
| Accès | `127.0.0.1` | tunnel SSH, aucun port applicatif ouvert |
| Données | dossiers du dépôt | volume persistant, survit à la destruction de la VM |
| Modèle | profil `small` (3B) | profil `medium` (7B) par défaut |

## Prérequis

- OpenTofu ≥ 1.6, `rsync`, `ssh` ;
- un compte Scaleway et une clé API (console → IAM → Clés API) ;
- une clé SSH.

## Mise en route

```bash
cd deploy/scaleway

cp env.example .env && $EDITOR .env && source .env        # identifiants Scaleway
cp terraform.tfvars.example terraform.tfvars              # clé SSH, IP autorisée
$EDITOR terraform.tfvars

../../deploy/scaleway/scripts/up.sh                       # provisionne et déploie
```

`up.sh` enchaîne : `tofu apply`, attente de la fin de cloud-init, transfert du
code, téléchargement des modèles (vérifiés par empreinte), démarrage de la pile.
Comptez une quinzaine de minutes la première fois, l'essentiel en téléchargement.

Ensuite :

```bash
./scripts/push-data.sh          # envoie data/ vers le volume persistant
ssh <vm> 'cd /opt/rag-source && docker compose exec -T api python -m rag_source.ingest /data'
./scripts/tunnel.sh             # puis http://127.0.0.1:8080
```

## Ce que crée le déploiement

- **une instance** (`PRO2-S` par défaut : 4 vCPU, 16 Go) ;
- **un volume de données** monté sur `/data` — modèles, corpus, index Qdrant ;
- **un groupe de sécurité** qui n'ouvre que le port 22, depuis l'adresse déclarée
  dans `allowed_ssh_cidr`.

L'interface, l'API, la base vectorielle et les serveurs de modèles n'écoutent que
sur la boucle locale de la VM. Le tunnel SSH est le seul chemin d'accès : pas de
certificat TLS à gérer, pas de mot de passe applicatif exposé, et rien à scanner
pour un tiers.

## Secrets

Aucun secret n'entre dans l'état OpenTofu :

- les identifiants Scaleway restent dans l'environnement (`source .env`) ;
- le jeton de l'API et la clé Qdrant sont **générés sur la VM** par cloud-init, au
  premier démarrage, et ne sont jamais régénérés ensuite.

`terraform.tfvars` ne contient que des valeurs non sensibles : gabarit, clé SSH
publique, adresse autorisée.

## Sauvegarde et restauration

```bash
./scripts/backup.sh                     # instantané Qdrant + corpus, rapatriés
```

La sauvegarde passe par un **instantané Qdrant**, pas par une copie des fichiers :
copier une base pendant qu'elle écrit produit un résultat incohérent.

Restaurer un instantané sur la VM :

```bash
scp backup/<date>/<instantané>.snapshot <vm>:/tmp/
ssh <vm>
cd /opt/rag-source && source .env
curl -X POST "http://127.0.0.1:6333/collections/rag_source/snapshots/upload?priority=snapshot" \
  -H "api-key: $RAG_SOURCE_QDRANT_API_KEY" \
  -F "snapshot=@/tmp/<instantané>.snapshot"
```

## Arrêt

```bash
./scripts/down.sh          # détruit l'instance, conserve les données
./scripts/down.sh --all    # supprime aussi le volume (confirmation demandée)
```

Éteindre une machine et perdre ses données sont deux décisions distinctes, et les
deux commandes le sont aussi. Après un `down.sh`, un nouvel `up.sh` retrouve le
volume avec son index et son corpus : seul le temps de reconstruction de la VM est
à repayer.

## Coût indicatif

Instance `PRO2-S` et volume de 50 Go : quelques dizaines d'euros par mois si la
machine tourne en continu. `down.sh` arrête la facturation de l'instance et ne
laisse que celle du volume, de l'ordre de quelques euros.

Vérifier les tarifs en vigueur sur le site de Scaleway : ils changent, ce fichier
non.
