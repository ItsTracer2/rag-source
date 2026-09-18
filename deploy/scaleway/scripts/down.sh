#!/usr/bin/env bash
# Détruit l'instance. Le volume de données est conservé par défaut.
#
#   ./deploy/scaleway/scripts/down.sh          # arrête la facturation de la VM
#   ./deploy/scaleway/scripts/down.sh --all    # supprime aussi corpus et index
#
# La séparation est volontaire : éteindre une machine et perdre ses données sont
# deux décisions différentes. Le projet d'origine les confondait — son teardown
# effaçait la base vectorielle, les documents et les modèles sans prévenir.

set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALL="${1:-}"

if [ "$ALL" = "--all" ]; then
    cat <<'WARNING'
⚠  Suppression TOTALE demandée : instance, volume de données, index et corpus.
   Cette opération est irréversible. Sauvegarder d'abord :
       ./deploy/scaleway/scripts/backup.sh

WARNING
    read -r -p "Taper « supprimer » pour confirmer : " answer
    [ "$answer" = "supprimer" ] || { echo "Annulé."; exit 1; }

    # prevent_destroy protège le volume : on lève la protection explicitement,
    # le temps de cette suppression volontaire.
    tofu -chdir="$DEPLOY_DIR" state rm scaleway_block_volume.data >/dev/null
    tofu -chdir="$DEPLOY_DIR" destroy -auto-approve
    echo "⚠  Le volume a été retiré de l'état : le supprimer dans la console Scaleway."
    exit 0
fi

echo "==> destruction de l'instance (le volume de données est conservé)"
tofu -chdir="$DEPLOY_DIR" destroy -auto-approve \
    -target=scaleway_instance_server.rag \
    -target=scaleway_instance_ip.rag \
    -target=scaleway_instance_security_group.rag

cat <<'EOF2'

✅ Instance détruite. Le volume de données reste facturé (quelques euros par mois)
   et sera réutilisé au prochain ./deploy/scaleway/scripts/up.sh.

   Pour tout supprimer : ./deploy/scaleway/scripts/down.sh --all
EOF2
