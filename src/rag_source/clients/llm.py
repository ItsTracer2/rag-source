"""Client de génération, API compatible OpenAI.

Un seul protocole pour tous les fournisseurs : ``llama-server`` en local, vLLM,
ou une API distante. Le reste du code ne sait pas qui répond — seule la
configuration le décide, et :attr:`Settings.sovereignty` dit publiquement où
partent les données.

L'implémentation d'origine passait par LangChain pour ces quelques appels, ce qui
apportait un large graphe de dépendances et des API mouvantes. Ici, deux requêtes
HTTP suffisent : une complétion, et la même en flux.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600.0

Role = Literal["system", "user", "assistant"]


class LLMError(RuntimeError):
    """Le service de génération est injoignable ou a renvoyé une réponse inattendue."""


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ChatClient(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str: ...

    def stream(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]: ...


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        *,
        model: str = "local",
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = client or httpx.Client(timeout=timeout, headers=headers)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        payload = self._payload(messages, max_tokens, temperature, stream=False)
        try:
            response = self._client.post(self._url, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"Service de génération injoignable ({self._url}) : {exc}") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Réponse de génération inexploitable : {exc}") from exc
        return str(content)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Fragments de réponse au fil de la génération.

        Sur CPU, une réponse complète demande plusieurs dizaines de secondes :
        sans flux, l'interface resterait figée sans rien montrer.
        """
        payload = self._payload(messages, max_tokens, temperature, stream=True)
        try:
            with self._client.stream("POST", self._url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    piece = _parse_sse_line(line)
                    if piece:
                        yield piece
        except httpx.HTTPError as exc:
            raise LLMError(f"Service de génération injoignable ({self._url}) : {exc}") from exc

    def _payload(
        self,
        messages: Sequence[Message],
        max_tokens: int | None,
        temperature: float | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [message.as_dict() for message in messages],
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "temperature": temperature if temperature is not None else self._temperature,
            "stream": stream,
        }

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAICompatibleClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _parse_sse_line(line: str) -> str:
    """Extrait le texte d'une ligne d'événement SSE, ou une chaîne vide."""
    if not line.startswith("data:"):
        return ""
    data = line[len("data:") :].strip()
    if not data or data == "[DONE]":
        return ""
    try:
        chunk = json.loads(data)
        delta = chunk["choices"][0].get("delta") or {}
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        logger.debug("Fragment SSE ignoré : %r", data[:120])
        return ""
    return str(delta.get("content") or "")
