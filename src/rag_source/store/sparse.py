"""Vecteurs creux BM25, pour la recherche par mots-clés.

Pourquoi ajouter des mots-clés à une recherche sémantique : un vecteur dense sait
qu'« éteindre l'appareil » ressemble à « couper l'alimentation », mais il est
médiocre sur les identifiants exacts — « R24 », « E01 », « QSCD », une référence de
pièce. Ces termes sont pourtant ceux que l'on tape le plus souvent.

Le principe : le client envoie la fréquence des termes (saturée façon BM25), et
Qdrant applique l'IDF de son côté, car lui seul connaît le corpus entier.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from hashlib import blake2b

# Paramètres BM25 usuels : k1 module la saturation de la fréquence, b l'effet de la
# longueur du document.
K1 = 1.2
B = 0.75
_MIN_LENGTH = 2
_MAX_LENGTH = 40
_TOKEN = re.compile(r"[0-9a-z]+(?:[-_.][0-9a-z]+)*", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SparseVector:
    indices: list[int]
    values: list[float]

    def __len__(self) -> int:
        return len(self.indices)


def tokenize(text: str) -> list[str]:
    """Termes d'un texte, normalisés.

    Les accents sont retirés (« procédure » et « procedure » doivent se rejoindre),
    la casse est ignorée, et les identifiants composés sont conservés entiers :
    « MW-104 » reste un seul terme plutôt que « mw » et « 104 ».
    """
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return [token for token in _TOKEN.findall(folded) if _MIN_LENGTH <= len(token) <= _MAX_LENGTH]


def term_id(term: str) -> int:
    """Identifiant numérique stable d'un terme.

    Qdrant indexe les dimensions creuses par entier : on hache le terme sur 32 bits.
    Une collision ferait partager un poids à deux termes ; avec 4 milliards de
    valeurs pour quelques dizaines de milliers de termes, le risque est négligeable
    et sans conséquence grave (un peu de bruit dans le classement).
    """
    return int.from_bytes(blake2b(term.encode("utf-8"), digest_size=4).digest(), "big")


def encode_document(text: str, average_length: float) -> SparseVector:
    """Vecteur creux d'un chunk, avec saturation BM25.

    ``average_length`` est la longueur moyenne des documents du corpus : c'est elle
    qui empêche un long document de dominer le classement par simple répétition.
    """
    tokens = tokenize(text)
    if not tokens:
        return SparseVector(indices=[], values=[])

    counts: dict[int, int] = {}
    for token in tokens:
        key = term_id(token)
        counts[key] = counts.get(key, 0) + 1

    length_ratio = len(tokens) / average_length if average_length > 0 else 1.0
    normalizer = K1 * (1 - B + B * length_ratio)
    indices, values = [], []
    for key, count in counts.items():
        indices.append(key)
        values.append(count * (K1 + 1) / (count + normalizer))
    return SparseVector(indices=indices, values=values)


def encode_query(text: str) -> SparseVector:
    """Vecteur creux d'une question : présence du terme, sans pondération de longueur."""
    terms = {term_id(token) for token in tokenize(text)}
    return SparseVector(indices=sorted(terms), values=[1.0] * len(terms))


def average_token_length(texts: list[str]) -> float:
    """Longueur moyenne, en termes, d'un ensemble de textes."""
    if not texts:
        return 1.0
    total = sum(len(tokenize(text)) for text in texts)
    return max(total / len(texts), 1.0)
