"""Chargeurs de documents, un par format.

Importer ce paquet suffit à peupler le registre : chaque module se déclare via
``@register_loader``.
"""

from rag_source.ingest.loaders import excel, markdown, pdf  # noqa: F401  (effet de bord)
from rag_source.ingest.loaders.base import (
    Extracted,
    Loader,
    LoaderError,
    loader_for,
    register_loader,
    supported_extensions,
)

__all__ = [
    "Extracted",
    "Loader",
    "LoaderError",
    "loader_for",
    "register_loader",
    "supported_extensions",
]
