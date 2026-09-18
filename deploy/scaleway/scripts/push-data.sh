#!/usr/bin/env bash
# Envoie le corpus local vers le volume persistant de la VM.
#
#   ./deploy/scaleway/scripts/push-data.sh [dossier]
#
# Les documents vont sur /data, qui survit à la destruction de l'instance. Le
# transfert est incrémental : seuls les fichiers modifiés repartent.

set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$DEPLOY_DIR/../.." && pwd)"
SOURCE="${1:-$REPO_ROOT/data}"
TARGET="$(tofu -chdir="$DEPLOY_DIR" output -raw ssh_target)"

[ -d "$SOURCE" ] || { echo "ERREUR : $SOURCE introuvable." >&2; exit 1; }

echo "==> envoi de $SOURCE vers $TARGET:/data/data/"
rsync -az --delete --info=stats1 \
    -e "ssh -o StrictHostKeyChecking=accept-new" \
    --exclude '.DS_Store' --exclude '.gitkeep' \
    "$SOURCE/" "$TARGET:/data/data/"

echo ""
echo "Indexer maintenant :"
echo "  ssh $TARGET 'cd /opt/rag-source && docker compose exec -T api python -m rag_source.ingest /data'"
