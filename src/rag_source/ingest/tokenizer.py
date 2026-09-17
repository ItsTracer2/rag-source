"""Comptage de tokens.

La taille des chunks se mesure en tokens, pas en caractères : c'est la seule unité
que comprennent le modèle d'embedding (qui tronque au-delà de sa limite) et le LLM
(dont le contexte est un budget de tokens). Compter des caractères, comme le faisait
l'implémentation d'origine, revient à ignorer que « anticonstitutionnellement » et
« le chat dort » ne coûtent pas la même chose selon la langue et le vocabulaire.

Deux implémentations :

- :class:`ModelTokenCounter` utilise le tokenizer exact du modèle d'embedding. Le
  fichier est récupéré avec les modèles (``scripts/fetch-models.sh``) ;
- :class:`EstimatedTokenCounter` prend le relais quand ce fichier est absent, avec
  une estimation par caractères. Volontairement un peu pessimiste : mieux vaut un
  chunk trop court qu'un chunk tronqué à l'embedding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer

# Ratio caractères/token observé sur du texte latin ; l'anglais est plus dense que
# le français, on prend la valeur prudente.
_CHARS_PER_TOKEN = 3.0


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class EstimatedTokenCounter:
    """Estimation par le nombre de caractères, sans dépendance à un modèle."""

    exact = False

    def count(self, text: str) -> int:
        return max(1, round(len(text) / _CHARS_PER_TOKEN)) if text else 0


class ModelTokenCounter:
    """Comptage exact, avec le tokenizer du modèle d'embedding."""

    exact = True

    def __init__(self, tokenizer_path: Path) -> None:
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


def get_token_counter(tokenizer_path: Path | None) -> TokenCounter:
    """Compteur exact si le tokenizer est disponible, estimation sinon."""
    if tokenizer_path is not None and tokenizer_path.is_file():
        return ModelTokenCounter(tokenizer_path)
    return EstimatedTokenCounter()
