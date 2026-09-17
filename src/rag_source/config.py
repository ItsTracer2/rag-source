"""Configuration de l'application, lue depuis l'environnement (préfixe ``RAG_SOURCE_``).

Toute la configuration passe par ce module : une seule source de vérité, typée et
validée au démarrage. Une configuration invalide fait échouer le démarrage plutôt
que de produire un comportement silencieusement incorrect.

Garde de souveraineté
---------------------
Par défaut, aucune donnée du corpus ne quitte l'hôte. Deux situations enverraient
des extraits du corpus à un tiers :

1. un fournisseur LLM externe (``llm_provider != "local"``) ;
2. un fournisseur « local » dont l'URL pointe en fait vers un hôte public.

Dans les deux cas, le démarrage est refusé sauf si ``RAG_SOURCE_ALLOW_EXTERNAL_LLM``
vaut exactement :data:`EXTERNAL_LLM_ACK`. La valeur est volontairement verbeuse :
ce n'est pas un réglage de qualité, c'est une rupture de la garantie de souveraineté.
"""

from __future__ import annotations

import ipaddress
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EXTERNAL_LLM_ACK = "i-understand-data-leaves-the-host"

# Suffixes DNS qui ne sont résolus que sur un réseau local ou privé.
_PRIVATE_DNS_SUFFIXES = (".local", ".internal", ".localhost", ".lan", ".home.arpa")


class LLMProvider(StrEnum):
    LOCAL = "local"
    """Serveur compatible OpenAI auto-hébergé (llama-server par défaut)."""
    ANTHROPIC = "anthropic"
    """API Claude : les extraits du corpus sont envoyés à Anthropic."""


class Sovereignty(StrEnum):
    LOCAL = "local"
    EXTERNAL = "external"


def is_private_endpoint(url: str) -> bool:
    """Indique si ``url`` désigne un hôte qui ne quitte pas le réseau local.

    Sont considérés comme privés : ``localhost``, les IP de bouclage et privées,
    les noms courts sans point (noms de service Docker, ex. ``llm``) et les
    suffixes DNS réservés aux réseaux locaux.

    Limite assumée : aucune résolution DNS n'est faite. Un nom public qui
    résoudrait vers une IP privée est traité comme externe (faux positif sûr).
    """
    host = urlparse(url).hostname
    if not host:
        return False
    host = host.lower().rstrip(".")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host == "localhost" or "." not in host or host.endswith(_PRIVATE_DNS_SUFFIXES)
    return ip.is_private or ip.is_loopback or ip.is_link_local


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_SOURCE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Corpus ──────────────────────────────────────────────────────────────
    data_dir: Path = Path("data")

    # ── API ─────────────────────────────────────────────────────────────────
    api_token: SecretStr | None = None
    """Jeton Bearer exigé par l'API. Obligatoire dès que l'API est démarrée."""

    # ── Modèles auto-hébergés (llama-server) ────────────────────────────────
    embed_url: str = "http://embed:8080"
    rerank_url: str = "http://rerank:8080"

    # ── LLM de génération ───────────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.LOCAL
    llm_base_url: str = "http://llm:8080/v1"
    llm_model: str = "local"
    llm_api_key: SecretStr | None = None
    llm_context_size: int = Field(default=8192, ge=2048)
    llm_max_tokens: int = Field(default=1024, ge=64)
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    allow_external_llm: str | None = None

    # ── Base vectorielle ────────────────────────────────────────────────────
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "rag_source"

    @model_validator(mode="after")
    def _check_sovereignty(self) -> Settings:
        if self.sovereignty is Sovereignty.EXTERNAL and self.allow_external_llm != EXTERNAL_LLM_ACK:
            if self.llm_provider is LLMProvider.LOCAL:
                reason = (
                    f"llm_base_url ({self.llm_base_url}) ne pointe pas vers un hôte local ou privé"
                )
            else:
                reason = f"le fournisseur LLM '{self.llm_provider}' est une API externe"
            raise ValueError(
                f"Garantie de souveraineté rompue : {reason}. Des extraits du corpus "
                "quitteraient l'hôte. Pour l'accepter en connaissance de cause, définir "
                f"RAG_SOURCE_ALLOW_EXTERNAL_LLM={EXTERNAL_LLM_ACK}"
            )
        if self.llm_provider is LLMProvider.ANTHROPIC and self.llm_api_key is None:
            raise ValueError("RAG_SOURCE_LLM_API_KEY est requis pour le fournisseur 'anthropic'.")
        return self

    @property
    def sovereignty(self) -> Sovereignty:
        if self.llm_provider is LLMProvider.LOCAL and is_private_endpoint(self.llm_base_url):
            return Sovereignty.LOCAL
        return Sovereignty.EXTERNAL


@lru_cache
def get_settings() -> Settings:
    """Settings de l'application, construites une seule fois par processus."""
    return Settings()
