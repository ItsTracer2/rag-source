#!/usr/bin/env bash
# Ouvre l'interface RAG-Source dans un tunnel SSH.
#
#   ./deploy/scaleway/scripts/tunnel.sh      puis http://127.0.0.1:8080
#
# Rien n'est publié sur Internet : ni l'interface, ni l'API, ni la base
# vectorielle. Le tunnel est le seul chemin d'accès, et il n'ouvre l'application
# qu'à cette machine. C'est aussi ce qui permet de se passer de TLS et de mots de
# passe applicatifs supplémentaires.

set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$(tofu -chdir="$DEPLOY_DIR" output -raw ssh_target)"
PORT="${RAG_SOURCE_TUNNEL_PORT:-8080}"

echo "Interface : http://127.0.0.1:${PORT}  (Ctrl+C pour fermer)"
exec ssh -N -o StrictHostKeyChecking=accept-new \
    -L "${PORT}:127.0.0.1:8080" "$TARGET"
