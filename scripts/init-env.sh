#!/usr/bin/env bash
# Crée le fichier .env local à partir de .env.example, avec des secrets générés.
#
#   ./scripts/init-env.sh
#
# Deux secrets sont nécessaires même en local :
#   - RAG_SOURCE_API_TOKEN     jeton Bearer de l'API RAG-Source ;
#   - RAG_SOURCE_QDRANT_API_KEY clé d'accès à la base vectorielle.
#
# Ils sont générés aléatoirement plutôt que laissés vides : une valeur vide n'est
# pas « pas d'authentification », c'est une authentification impossible à satisfaire
# (Qdrant refuse alors toute écriture avec un 401 difficile à diagnostiquer).
#
# Le fichier .env n'est jamais versionné. Relancer ce script ne l'écrase pas.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
EXAMPLE="$REPO_ROOT/.env.example"

if [ -f "$ENV_FILE" ]; then
    echo "ℹ️  $ENV_FILE existe déjà — rien à faire."
    echo "   (le supprimer pour repartir de .env.example)"
    exit 0
fi

command -v openssl >/dev/null || { echo "openssl est requis." >&2; exit 1; }

secret() { openssl rand -hex 32; }

cp "$EXAMPLE" "$ENV_FILE"
# sed -i diffère entre GNU et BSD : on passe par un fichier temporaire.
tmp="$(mktemp)"
sed \
    -e "s|^RAG_SOURCE_API_TOKEN=.*|RAG_SOURCE_API_TOKEN=$(secret)|" \
    -e "s|^RAG_SOURCE_QDRANT_API_KEY=.*|RAG_SOURCE_QDRANT_API_KEY=$(secret)|" \
    "$ENV_FILE" > "$tmp"
mv "$tmp" "$ENV_FILE"
chmod 600 "$ENV_FILE"

echo "✅ $ENV_FILE créé, secrets générés."
