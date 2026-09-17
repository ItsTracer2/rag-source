"""Contrat attendu d'un index vectoriel.

L'indexation ne dépend pas de Qdrant mais de ce protocole : il documente
exactement ce que le pipeline exige d'un index, il permet de le tester sans
conteneur, et il laisse la porte ouverte à un autre moteur sans toucher au
pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag_source.domain import Chunk
from rag_source.store.sparse import SparseVector


class ChunkStore(Protocol):
    def create(self, dimension: int) -> None:
        """Crée la collection si nécessaire."""
        ...

    def dimension(self) -> int | None:
        """Dimension des vecteurs déjà stockés, ou ``None`` si rien n'existe."""
        ...

    def indexed_versions(self) -> dict[str, str]:
        """Empreinte du fichier indexé, par document."""
        ...

    def delete_source(self, source: str) -> None:
        """Supprime tous les chunks d'un document."""
        ...

    def upsert(
        self,
        chunks: Sequence[Chunk],
        dense: Sequence[list[float]],
        sparse: Sequence[SparseVector],
    ) -> None:
        """Écrit ou remplace des chunks avec leurs deux vecteurs."""
        ...

    def count(self) -> int:
        """Nombre de chunks indexés."""
        ...
