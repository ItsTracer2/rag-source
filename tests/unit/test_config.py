from typing import Any

import pytest
from pydantic import ValidationError

from rag_source.config import (
    EXTERNAL_LLM_ACK,
    LLMProvider,
    Settings,
    Sovereignty,
    is_private_endpoint,
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Les tests ne doivent pas dépendre de l'environnement du poste."""
    import os

    for key in list(os.environ):
        if key.startswith("RAG_SOURCE_"):
            monkeypatch.delenv(key)


def make(**kwargs: Any) -> Settings:
    return Settings(_env_file=None, **kwargs)


class TestPrivateEndpoint:
    @pytest.mark.parametrize(
        "url",
        [
            "http://llm:8080/v1",  # nom de service Docker
            "http://localhost:8080",
            "http://127.0.0.1:8080/v1",
            "http://[::1]:8080",
            "http://10.0.0.5:8080",
            "http://192.168.1.20:8080",
            "http://172.17.0.1:8080",
            "http://host.docker.internal:8080",
            "http://gpu-box.lan:8080",
            "http://mac-studio.local:8080",
        ],
    )
    def test_private(self, url: str) -> None:
        assert is_private_endpoint(url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.openai.com/v1",
            "https://api.anthropic.com",
            "http://8.8.8.8:8080",
            "http://llm.example.com:8080",
            "not a url",
            "",
        ],
    )
    def test_public_or_invalid(self, url: str) -> None:
        assert not is_private_endpoint(url)


class TestSovereignty:
    def test_defaults_are_local(self) -> None:
        settings = make()
        assert settings.llm_provider is LLMProvider.LOCAL
        assert settings.sovereignty is Sovereignty.LOCAL

    def test_external_provider_allowed_by_default(self) -> None:
        """Outil générique : un LLM externe est permis, mais toujours signalé."""
        settings = make(llm_provider="external", llm_api_key="sk-test")
        assert settings.sovereignty is Sovereignty.EXTERNAL

    def test_external_provider_refused_when_local_required(self) -> None:
        with pytest.raises(ValidationError, match="souverain"):
            make(require_local_llm=True, llm_provider="external", llm_api_key="sk-test")

    def test_wrong_acknowledgement_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="souverain"):
            make(
                require_local_llm=True,
                llm_provider="external",
                llm_api_key="sk-test",
                allow_external_llm="true",
            )

    def test_exact_acknowledgement_is_accepted(self) -> None:
        settings = make(
            require_local_llm=True,
            llm_provider="external",
            llm_api_key="sk-test",
            allow_external_llm=EXTERNAL_LLM_ACK,
        )
        assert settings.sovereignty is Sovereignty.EXTERNAL

    def test_local_provider_pointing_to_public_host_is_detected(self) -> None:
        """Contournement à repérer : provider 'local' mais URL d'une API publique."""
        assert make(llm_base_url="https://api.openai.com/v1").sovereignty is Sovereignty.EXTERNAL
        with pytest.raises(ValidationError, match="ne pointe pas vers un hôte local"):
            make(require_local_llm=True, llm_base_url="https://api.openai.com/v1")

    def test_anthropic_requires_api_key(self) -> None:
        with pytest.raises(ValidationError, match="LLM_API_KEY"):
            make(llm_provider="external")


def test_reads_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_SOURCE_LLM_MODEL", "qwen")
    monkeypatch.setenv("RAG_SOURCE_API_TOKEN", "s3cret")
    settings = make()
    assert settings.llm_model == "qwen"
    assert settings.api_token is not None
    assert settings.api_token.get_secret_value() == "s3cret"
    assert "s3cret" not in repr(settings)


def test_rejects_too_small_context() -> None:
    with pytest.raises(ValidationError):
        make(llm_context_size=512)
