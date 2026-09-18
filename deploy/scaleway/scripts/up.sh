#!/usr/bin/env bash
# Provisionne la VM, y déploie RAG-Source, et laisse la pile prête à indexer.
#
#   source deploy/scaleway/.env     # identifiants Scaleway
#   ./deploy/scaleway/scripts/up.sh
#
# Réexécutable sans dommage : OpenTofu ne recrée que ce qui manque, le rsync ne
# transfère que ce qui a changé, et les secrets déjà générés sur la VM sont
# conservés.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$HERE")"
REPO_ROOT="$(cd "$DEPLOY_DIR/../.." && pwd)"
SSH_OPTS="-o StrictHostKeyChecking=accept-new"

for tool in tofu rsync ssh; do
    command -v "$tool" >/dev/null || { echo "ERREUR : $tool est requis." >&2; exit 1; }
done
: "${SCW_ACCESS_KEY:?identifiants Scaleway absents — voir deploy/scaleway/env.example}"
: "${SCW_SECRET_KEY:?identifiants Scaleway absents — voir deploy/scaleway/env.example}"
: "${SCW_DEFAULT_PROJECT_ID:?identifiants Scaleway absents — voir deploy/scaleway/env.example}"

echo "==> [1/5] provisionnement de l'infrastructure"
tofu -chdir="$DEPLOY_DIR" init -upgrade -input=false >/dev/null
tofu -chdir="$DEPLOY_DIR" apply -auto-approve -input=false

TARGET="$(tofu -chdir="$DEPLOY_DIR" output -raw ssh_target)"
echo "    VM : $TARGET"

echo "==> [2/5] attente de la fin de l'initialisation (cloud-init)"
for _ in $(seq 1 120); do
    if ssh $SSH_OPTS -o ConnectTimeout=5 "$TARGET" \
        "test -f /var/lib/cloud/rag-source-ready" 2>/dev/null; then
        echo "    VM prête"
        break
    fi
    sleep 10
done

echo "==> [3/5] transfert du code"
ssh $SSH_OPTS "$TARGET" "mkdir -p /opt/rag-source"
# Le corpus et les modèles ne transitent pas ici : ils vivent sur /data et se
# poussent séparément (push-data.sh), ce qui évite de tout renvoyer à chaque fois.
rsync -az --delete \
    -e "ssh $SSH_OPTS" \
    --exclude '.git' --exclude '.venv' --exclude 'data' --exclude 'models' \
    --exclude '.env' --exclude '__pycache__' --exclude '.terraform' \
    "$REPO_ROOT/" "$TARGET:/opt/rag-source/"

echo "==> [4/5] téléchargement des modèles (première fois : plusieurs minutes)"
ssh $SSH_OPTS "$TARGET" bash <<'REMOTE'
set -euo pipefail
cd /opt/rag-source
source .env
RAG_SOURCE_MODELS_DIR=/data/models ./scripts/fetch-models.sh "${RAG_SOURCE_LLM_PROFILE:-medium}"
REMOTE

echo "==> [5/5] démarrage de la pile"
ssh $SSH_OPTS "$TARGET" bash <<'REMOTE'
set -euo pipefail
cd /opt/rag-source
docker compose -f compose.yaml -f deploy/scaleway/compose.prod.yaml up -d --build
docker compose ps --format 'table {{.Service}}\t{{.Status}}'
REMOTE

cat <<EOF

────────────────────────────────────────────────────────────────────
✅ RAG-Source déployé.

  Pousser des documents   ./deploy/scaleway/scripts/push-data.sh
  Indexer                 ssh $TARGET 'cd /opt/rag-source && docker compose exec api rag-source ingest /data'
  Ouvrir l'interface      ./deploy/scaleway/scripts/tunnel.sh
  Sauvegarder l'index     ./deploy/scaleway/scripts/backup.sh
  Éteindre                ./deploy/scaleway/scripts/down.sh
────────────────────────────────────────────────────────────────────
EOF
