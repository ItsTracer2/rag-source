# Image de l'API RAG-Source.
#
# Construction en deux temps : les dépendances d'abord (couche mise en cache tant
# que le verrou ne change pas), le code ensuite. Modifier une ligne de Python ne
# réinstalle donc rien.
#
# L'image reste légère parce que le projet ne dépend d'aucun PyTorch : les modèles
# vivent dans les conteneurs llama.cpp, pas ici.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dépendances seules : cette couche survit aux modifications du code. Le README est
# copié parce que pyproject le déclare comme description longue du paquet.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


FROM python:3.12-slim-bookworm AS runtime

# tesseract sert à l'OCR des PDF scannés, utilisé par l'ingestion lancée depuis
# le conteneur. Les paquets de langue conditionnent la qualité de la
# reconnaissance : en ajouter d'autres avec tesseract-ocr-<langue>.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng curl \
    && rm -rf /var/lib/apt/lists/*

# Un compte non privilégié : un service exposé ne tourne pas en root.
RUN useradd --create-home --uid 10001 rag
WORKDIR /app

COPY --from=builder --chown=rag:rag /app/.venv /app/.venv
COPY --chown=rag:rag src ./src
COPY --chown=rag:rag models.lock ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER rag
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS -o /dev/null http://127.0.0.1:8000/docs || exit 1

CMD ["uvicorn", "rag_source.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
