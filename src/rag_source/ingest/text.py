"""Utilitaires de nettoyage de texte partagés par les chargeurs."""

from __future__ import annotations

import re

# Les marqueurs de liste par lettre sont volontairement en minuscules : « a) » est
# une puce, « I. » est un titre en chiffres romains. La casse est le seul indice
# qui les distingue.
_BULLET = re.compile(r"^\s*(?:[•▪◦o\-–—*]|\(?\d+[.)]|[a-z][.)])\s+")
_MULTISPACE = re.compile(r"[ \t ]+")
_SOFT_HYPHEN = "­"


def normalize_spaces(text: str) -> str:
    """Espaces multiples et insécables réduits à une espace simple."""
    return _MULTISPACE.sub(" ", text.replace(_SOFT_HYPHEN, "")).strip()


def is_bullet(text: str) -> bool:
    """Le texte commence-t-il par une puce ou une numérotation de liste ?"""
    return bool(_BULLET.match(text))


def join_wrapped_lines(lines: list[str]) -> str:
    """Recolle les lignes coupées par la mise en page du PDF.

    Une ligne qui se termine par un tiret est recollée sans espace (césure), sinon
    avec une espace. Les puces gardent leur propre ligne : sans cela, une liste
    d'exigences devient un paragraphe illisible — c'est le défaut du nettoyage par
    expression régulière de l'implémentation d'origine.
    """
    out: list[str] = []
    for raw in lines:
        line = normalize_spaces(raw)
        if not line:
            continue
        if not out or is_bullet(line):
            out.append(line)
        elif out[-1].endswith("-") and not out[-1].endswith((" -", "--")):
            out[-1] = out[-1][:-1] + line
        else:
            out[-1] = f"{out[-1]} {line}"
    return "\n".join(out)
