#!/usr/bin/env bash
# Rapatrie une sauvegarde de l'index et du corpus.
#
#   ./deploy/scaleway/scripts/backup.sh [dossier-de-destination]
#
# L'index est exporté par un instantané Qdrant plutôt qu'en copiant ses fichiers :
# copier une base pendant qu'elle écrit produit une sauvegarde incohérente.
#
# Le projet d'origine n'avait aucune procédure : son teardown détruisait la base
# vectorielle, les documents et les modèles en même temps que la VM.

set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$DEPLOY_DIR/../.." && pwd)"
DESTINATION="${1:-$REPO_ROOT/backup/$(date +%Y-%m-%d_%H%M)}"
TARGET="$(tofu -chdir="$DEPLOY_DIR" output -raw ssh_target)"
SSH_OPTS="-o StrictHostKeyChecking=accept-new"

mkdir -p "$DESTINATION"

echo "==> instantané Qdrant sur la VM"
SNAPSHOT=$(ssh $SSH_OPTS "$TARGET" bash <<'REMOTE'
set -euo pipefail
cd /opt/rag-source
source .env
collection="${RAG_SOURCE_QDRANT_COLLECTION:-rag_source}"
name=$(curl -fsS -X POST "http://127.0.0.1:6333/collections/${collection}/snapshots" \
    -H "api-key: ${RAG_SOURCE_QDRANT_API_KEY}" | jq -r '.result.name')
curl -fsS "http://127.0.0.1:6333/collections/${collection}/snapshots/${name}" \
    -H "api-key: ${RAG_SOURCE_QDRANT_API_KEY}" -o "/tmp/${name}"
echo "$name"
REMOTE
)

echo "==> rapatriement de $SNAPSHOT"
rsync -az --info=stats1 -e "ssh $SSH_OPTS" "$TARGET:/tmp/$SNAPSHOT" "$DESTINATION/"
ssh $SSH_OPTS "$TARGET" "rm -f /tmp/$SNAPSHOT"

echo "==> rapatriement du corpus"
rsync -az --info=stats1 -e "ssh $SSH_OPTS" "$TARGET:/data/data/" "$DESTINATION/data/"

echo ""
echo "✅ Sauvegarde dans $DESTINATION"
echo "   Restauration : voir deploy/scaleway/README.md"
