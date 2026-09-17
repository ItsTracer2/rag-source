#!/usr/bin/env bash
# Télécharge les modèles listés dans models.lock et vérifie leur empreinte.
#
#   ./scripts/fetch-models.sh            # profil LLM "small" (défaut)
#   ./scripts/fetch-models.sh medium     # LLM 7B
#   ./scripts/fetch-models.sh --all      # tous les profils
#
# Idempotent : un fichier déjà présent et valide est conservé. Un fichier dont
# l'empreinte ne correspond pas est refusé et supprimé — mieux vaut une erreur
# franche qu'un modèle silencieusement différent.
#
# Aucun jeton n'est nécessaire : tous les dépôts sont publics.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${RAG_SOURCE_MODELS_DIR:-$REPO_ROOT/models}"
LOCKFILE="$REPO_ROOT/models.lock"
PROFILE="${1:-small}"

[ -f "$LOCKFILE" ] || { echo "models.lock introuvable." >&2; exit 1; }
command -v curl >/dev/null || { echo "curl est requis." >&2; exit 1; }

if command -v sha256sum >/dev/null; then
    checksum() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null; then
    checksum() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
    echo "sha256sum ou shasum est requis." >&2
    exit 1
fi

human() { awk -v b="$1" 'BEGIN { printf "%.1f Go", b/1000000000 }'; }

mkdir -p "$MODELS_DIR"
downloaded=0 kept=0

while IFS='|' read -r role profile repo filename sha size; do
    case "$role" in ''|\#*) continue ;; esac
    if [ "$PROFILE" != "--all" ] && [ "$profile" != "all" ] && [ "$profile" != "$PROFILE" ]; then
        continue
    fi

    target="$MODELS_DIR/$filename"
    if [ -f "$target" ]; then
        if [ "$(checksum "$target")" = "$sha" ]; then
            echo "  ✔ $filename déjà présent et vérifié"
            kept=$((kept + 1))
            continue
        fi
        echo "  ⚠ $filename : empreinte incorrecte, re-téléchargement"
        rm -f "$target"
    fi

    echo "  ↓ $filename ($(human "$size")) depuis $repo"
    url="https://huggingface.co/$repo/resolve/main/$filename?download=true"
    curl --fail --location --progress-bar --output "$target.part" "$url"

    actual="$(checksum "$target.part")"
    if [ "$actual" != "$sha" ]; then
        rm -f "$target.part"
        echo "ERREUR : empreinte inattendue pour $filename" >&2
        echo "  attendue : $sha" >&2
        echo "  obtenue  : $actual" >&2
        exit 1
    fi
    mv "$target.part" "$target"
    downloaded=$((downloaded + 1))
done < "$LOCKFILE"

echo ""
echo "✅ Modèles prêts dans $MODELS_DIR ($downloaded téléchargé(s), $kept conservé(s))."
