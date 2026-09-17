"""Registre de chargeurs, un par format de document.

Ajouter un format = créer un module dans ce paquet, décorer la classe avec
:func:`register_loader` et l'importer dans ``loaders/__init__.py``. Aucun autre
fichier du pipeline n'a besoin de changer : le reste ne manipule que des
:class:`~rag_source.domain.Section`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from rag_source.domain import DocumentFormat, Section


class LoaderError(Exception):
    """Le fichier n'a pas pu être exploité (corrompu, chiffré, vide, non supporté)."""


@dataclass(frozen=True, slots=True)
class Extracted:
    """Résultat d'un chargeur, avant calcul de l'empreinte et du chemin relatif."""

    title: str
    sections: tuple[Section, ...]
    warnings: tuple[str, ...] = field(default=())


@runtime_checkable
class Loader(Protocol):
    extensions: frozenset[str]
    """Extensions gérées, en minuscules, point compris (ex. ``{".pdf"}``)."""
    format: DocumentFormat

    def load(self, path: Path) -> Extracted: ...


_REGISTRY: dict[str, Loader] = {}


def register_loader[LoaderT: type[Loader]](cls: LoaderT) -> LoaderT:
    """Enregistre un chargeur pour chacune de ses extensions."""
    loader = cls()
    for extension in loader.extensions:
        if (previous := _REGISTRY.get(extension)) is not None:
            raise RuntimeError(
                f"L'extension {extension} est déjà gérée par {type(previous).__name__}."
            )
        _REGISTRY[extension] = loader
    return cls


def loader_for(path: Path) -> Loader | None:
    """Chargeur capable de lire ``path``, ou ``None`` si le format est inconnu."""
    return _REGISTRY.get(path.suffix.lower())


def supported_extensions() -> frozenset[str]:
    return frozenset(_REGISTRY)
